"""Background tracking controller used by the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import math
import time
import traceback
from typing import Callable

from Lib.Config import Profile, Settings, load_joint_map, validate_profile, validate_settings
from Lib.Metrics import SessionMetrics, inference_over_budget
from Lib.Synchronization import FrameSynchronizer, MAX_FRAME_AGE


LogCallback = Callable[[str, str], None]
StateCallback = Callable[[str], None]
StatsCallback = Callable[[float, float, int], None]


class _LatestFrameReader:
    """Continuously capture one source without blocking the tracking loop."""

    def __init__(self, capture, label: str, on_frame=None):
        self.capture = capture
        self.label = label
        self.on_frame = on_frame or (lambda _timestamp: None)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sequence = 0
        self._frame = None
        self._captured_at = 0.0
        self.error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"vr-fbt-capture-{label}", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self.capture.isOpened():
                    self._stop.wait(0.05)
                    continue
                ok, frame = self.capture.read()
                captured_at = getattr(self.capture, "captured_at", None)
                if captured_at is None:
                    captured_at = time.monotonic()
                if not ok or frame is None:
                    self._stop.wait(0.005)
                    continue
                with self._lock:
                    self._sequence += 1
                    self._frame = frame
                    self._captured_at = captured_at
                self.on_frame(captured_at)
        except Exception as exc:
            self.error = exc
        finally:
            # The same thread owns read and release; concurrent native calls
            # can crash OpenCV rather than raise a recoverable Python error.
            try:
                self.capture.release()
            except Exception as exc:
                self.error = self.error or exc

    def latest(self, after_sequence: int):
        with self._lock:
            if self._sequence <= after_sequence or self._frame is None:
                return None
            return self._sequence, self._captured_at, self._frame

    @property
    def last_frame_at(self) -> float:
        with self._lock:
            return self._captured_at

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 0.5) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout)


@dataclass(slots=True)
class EngineCallbacks:
    log: LogCallback = lambda _level, _message: None
    state: StateCallback = lambda _state: None
    stats: StatsCallback = lambda _fps, _visibility, _frames: None
    health: Callable[[dict], None] = lambda _snapshot: None


class _PoseInference:
    """At most one inference per camera; collect healthy results independently."""

    def __init__(self, poses, metrics=None):
        self.poses = poses
        self.metrics = metrics
        self.pool = ThreadPoolExecutor(max_workers=len(poses), thread_name_prefix="vr-fbt-pose")
        self.pending = {}

    def busy(self, source_id):
        return source_id in self.pending

    def submit(self, source_id, label, frame, frame_size, captured_at):
        if self.busy(source_id):
            return False
        future = self.pool.submit(self._process, source_id, frame, captured_at)
        self.pending[source_id] = (label, frame, frame_size, captured_at, future)
        return True

    def _process(self, source_id, frame, captured_at):
        started = time.monotonic()
        result = self.poses[source_id].process(frame)
        completed = time.monotonic()
        if self.metrics is not None:
            self.metrics.inference(source_id, (completed - started) * 1000, captured_at, completed, getattr(result, "detected", False))
        return result

    def poll(self):
        completed = []
        for source_id, (label, frame, size, captured_at, future) in list(self.pending.items()):
            if not future.done():
                continue
            del self.pending[source_id]
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            completed.append((source_id, label, frame, size, captured_at, result, error))
        return completed

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class TrackingController:
    """Own one tracking worker and expose deterministic start/stop semantics."""

    def __init__(self, callbacks: EngineCallbacks | None = None):
        self.callbacks = callbacks or EngineCallbacks()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._vrchat_align_event = threading.Event()
        self._vrchat_calibrate_event = threading.Event()
        self.metrics = SessionMetrics()
        self.metrics.finish()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings: Settings, profile: Profile, remote_hub=None) -> None:
        if self.running:
            raise RuntimeError("Tracking is already running")
        validate_settings(settings)
        validate_profile(profile)
        from Lib.Config import profile_camera_sources
        self.metrics = SessionMetrics(profile_camera_sources(profile))
        self._stop_event.clear()
        self._vrchat_align_event.set()
        self._vrchat_calibrate_event.set()
        self._thread = threading.Thread(target=self._run, args=(settings, profile, remote_hub), name="vr-fbt-tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.running:
            self.callbacks.state("stopping")
            self._stop_event.set()

    def realign_vrchat(self) -> None:
        if not self.running:
            raise RuntimeError("Start tracking before aligning VRChat")
        self._vrchat_align_event.set()
        self._vrchat_calibrate_event.set()
        self.callbacks.log("INFO", "VRChat recalibration requested; stand upright, face the camera, and keep your full body visible")

    def wait(self, timeout: float = 3.0) -> bool:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.running

    def _run(self, settings: Settings, profile: Profile, remote_hub=None) -> None:
        captures = []
        capture_readers = []
        osc = None
        poses = {}
        inference_pool = None
        final_state = "stopped"
        opening_source = None
        try:
            self.callbacks.state("starting")
            self.callbacks.log("INFO", "Loading the pose model and compute backend…")
            import cv2
            from Lib import Data, OSCKit
            from Lib.Config import profile_camera_setups, profile_camera_sources
            from Lib.DebugView import DebugScene3D, fit_image_to_viewport, render_camera_mosaic, rotate_camera_frame
            from Lib.Tracking import CameraObservation, MultiCameraPoseFusion
            from Lib.Workers import CameraWorker, PoseWorker
            from Lib.VRChat import VRChatPoseSolver

            joint_map = load_joint_map(profile.joint_map)
            mapped_pose = Data.Map(joint_map)
            source_ids = profile_camera_sources(profile)
            camera_setups = profile_camera_setups(profile)
            estimated_lenses = sum(not setup.lens_intrinsics for setup in camera_setups)
            if len(source_ids) > 1 and estimated_lenses:
                self.callbacks.log("WARN", f"{estimated_lenses} camera(s) use estimated lenses; import measured lens calibration for more accurate camera rays")
            if len(source_ids) > 1 and profile.manual_camera_setup:
                defaults = profile_camera_setups(Profile(camera_source=source_ids[0], camera_sources=source_ids))
                if any(setup.position == factory.position and setup.rotation == factory.rotation
                       for setup, factory in zip(camera_setups, defaults)):
                    self.callbacks.log("WARN", "Camera positions appear to be factory defaults; measure them or switch to automatic mode")
            image_rotations = {setup.source_id: setup.image_rotation for setup in camera_setups}
            latency_offsets = {setup.source_id: setup.latency_ms / 1000 for setup in camera_setups}
            fusion = MultiCameraPoseFusion(
                source_ids,
                calibration_frames=12,
                manual_camera_setup=profile.manual_camera_setup,
                camera_setups=camera_setups,
                room_size_m=profile.room_size_m,
                calibration_duration_seconds=10.0,
            )
            vrchat_solver = VRChatPoseSolver(profile.user_height_m, calibration_frames=20)
            osc = OSCKit.Server(profile.server_ip, profile.server_port)
            self.callbacks.log("PASS", f"VRChat OSC output resolved to IPv4 {osc.target[0]}:{osc.target[1]}")
            for source_id in source_ids:
                opening_source = source_id
                if source_id.startswith("phone:"):
                    if remote_hub is None or not remote_hub.running:
                        raise RuntimeError("Start the phone camera server before using a phone source")
                    from Lib.RemoteCam import RemoteCapture
                    device_id = source_id.split(":", 1)[1]
                    capture = RemoteCapture(remote_hub.registry, device_id, cv2, timeout=0.30)
                    source_label = f"phone camera {device_id}"
                else:
                    local_index = int(source_id.split(":", 1)[1])
                    capture = CameraWorker(local_index, self._stop_event)
                    source_label = f"local camera {local_index}"
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open {source_label}")
                if hasattr(capture, "set"):
                    capture.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
                captures.append((source_id, source_label, capture))
                poses[source_id] = PoseWorker(profile.pose_quality, self._stop_event)
            opening_source = None
            for source_id, source_label, capture in captures:
                reader = _LatestFrameReader(capture, source_id.replace(":", "-"),
                    on_frame=lambda stamp, source=source_id: self.metrics.capture(source, stamp - latency_offsets.get(source, 0.0)))
                reader.start()
                capture_readers.append((source_id, source_label, reader))
            inference_pool = _PoseInference(poses, self.metrics)

            target_interval = 1.0 / settings.fps
            next_frame = started = time.monotonic()
            frames = 0
            sent_frames = 0
            had_pose = False
            last_vrchat_frame = None
            last_pose_at = 0.0
            calibration_active = True
            calibration_log_step = -1
            tracker_ids = {"2", "7", "8"} if profile.vrchat_tracker_set == "stable" else None
            source_label = ", ".join(item[1] for item in captures)
            self.callbacks.log("INFO", f"Tracking {len(captures)} camera(s): {source_label}")
            self.callbacks.log("INFO", f"MediaPipe {profile.pose_quality} output to {profile.server_ip}:{profile.server_port}")
            if len(captures) > 1:
                self.callbacks.log("INFO", "Multi-camera calibration: hold a T-pose for 10 seconds with your full body visible in every selected camera")
            rotated_sources = [f"{source_id} {degrees}°" for source_id, degrees in image_rotations.items() if degrees]
            if rotated_sources:
                self.callbacks.log("INFO", f"Clockwise camera image correction: {', '.join(rotated_sources)}")
            self.callbacks.log("INFO", "VRChat tracker set: hip + feet (stable)" if tracker_ids else "VRChat tracker set: all 8 (experimental)")
            debug_scene = DebugScene3D(cv2, room_size_m=profile.room_size_m) if profile.show_output else None
            debug_window = "VR-FBT 3D Tracking Debug"
            camera_window = "VR-FBT Camera Views"
            if debug_scene is not None:
                cv2.namedWindow(debug_window, getattr(cv2, "WINDOW_NORMAL", 0))
                cv2.resizeWindow(debug_window, debug_scene.width, debug_scene.height)
                cv2.setMouseCallback(debug_window, debug_scene.mouse_callback)
                cv2.namedWindow(camera_window, getattr(cv2, "WINDOW_NORMAL", 0))
            camera_window_sized = False
            self.callbacks.state("running")
            camera_calibrated_logged = len(captures) == 1
            last_geometry_warning = ""
            missing_sources: set[str] = set()
            last_any_frame = time.monotonic()
            capture_sequences = {source_id: 0 for source_id in source_ids}
            failed_sources: set[str] = set()
            inferred_views = {}
            synchronizer = FrameSynchronizer(source_ids)
            if len(source_ids) > 1:
                self.callbacks.log("INFO", "Camera synchronization: adaptive 100–200 ms playback horizon, 33 ms maximum view separation; late views are excluded")
            for source, delay in latency_offsets.items():
                if delay:
                    self.callbacks.log("INFO", f"{source}: subtracting {delay * 1000:.1f} ms measured residual camera delay")
            next_health = 0.0
            slow_inference_since = None
            slow_inference_warned = False

            while not self._stop_event.is_set():
                observations = []
                camera_views = []
                capture_now = time.monotonic()
                if capture_now >= next_health:
                    snapshot = self.metrics.snapshot()
                    self.callbacks.health(snapshot)
                    if len(source_ids) >= 2 and not slow_inference_warned:
                        too_slow = inference_over_budget(snapshot, settings.fps, len(source_ids))
                        slow_inference_since = (slow_inference_since or capture_now) if too_slow else None
                        if slow_inference_since is not None and capture_now - slow_inference_since >= 10:
                            self.callbacks.log("WARN", "Pose inference p95 exceeds the dual-camera frame budget; try the Lite model or lower Frame rate on the phone page")
                            slow_inference_warned = True
                    next_health = capture_now + 0.5
                for source_id, label, frame, frame_size, captured_at, camera_result, error in inference_pool.poll():
                    if error is not None:
                        failed_sources.add(source_id)
                        self.metrics.fail(source_id)
                        inferred_views.pop(source_id, None)
                        self.callbacks.log("ERROR", f"{label.capitalize()} inference failed: {error}; restart tracking to retry")
                        continue
                    inferred_views[source_id] = (
                        CameraObservation(source_id, camera_result, frame_size, captured_at), frame, label,
                    )
                    synchronizer.add(inferred_views[source_id][0], capture_now)
                for source_id, label, reader in capture_readers:
                    if source_id in failed_sources:
                        continue
                    if reader.error is not None:
                        if source_id not in failed_sources:
                            failed_sources.add(source_id)
                            self.metrics.fail(source_id)
                            self.callbacks.log("ERROR", f"{label.capitalize()} failed: {reader.error}; restart tracking to reopen it")
                        continue
                    if inference_pool.busy(source_id):
                        continue
                    latest = reader.latest(capture_sequences[source_id])
                    if latest is None:
                        silent_for = capture_now - reader.last_frame_at if reader.last_frame_at else capture_now - started
                        if silent_for >= 0.5 and source_id not in missing_sources:
                            missing_sources.add(source_id)
                            self.callbacks.log("WARN", f"{label.capitalize()} is not returning fresh frames")
                        continue
                    sequence, captured_at, frame = latest
                    captured_at -= latency_offsets.get(source_id, 0.0)
                    self.metrics.select(source_id, sequence)
                    capture_sequences[source_id] = sequence
                    if capture_now - captured_at > MAX_FRAME_AGE:
                        self.metrics.reject_stale(source_id)
                        if source_id not in missing_sources:
                            missing_sources.add(source_id)
                            self.callbacks.log("WARN", f"{label.capitalize()} frame is stale")
                        continue
                    if source_id in missing_sources:
                        missing_sources.remove(source_id)
                        self.callbacks.log("PASS", f"{label.capitalize()} resumed")
                    frame = rotate_camera_frame(frame, image_rotations.get(source_id, 0), cv2)
                    height, width = frame.shape[:2]
                    inference_pool.submit(source_id, label, frame, (width, height), captured_at)
                if self._stop_event.is_set():
                    break
                inference_now = time.monotonic()
                for source_id, (observation, frame, label) in list(inferred_views.items()):
                    if source_id in failed_sources or inference_now - observation.captured_at > MAX_FRAME_AGE:
                        self.metrics.reject_stale(source_id)
                        del inferred_views[source_id]
                        continue
                    observations.append(observation)
                    camera_views.append((frame, observation.pose, label))
                observations = synchronizer.select(inference_now, failed_sources)
                self.metrics.synchronization(observations)
                if not observations:
                    fusion.update([], cv2)
                    if time.monotonic() - last_any_frame > 3.0:
                        raise RuntimeError("No selected camera has returned a frame for three seconds")
                    self._stop_event.wait(min(target_interval, 0.05))
                    continue
                last_any_frame = time.monotonic()
                fusion_started = time.monotonic()
                fusion_result = fusion.update(observations, cv2)
                if len(source_ids) > 1 and not profile.manual_camera_setup:
                    self.metrics.calibration(fusion_result.camera_poses)
                self.metrics.stage("fusion", (time.monotonic() - fusion_started) * 1000)
                result = fusion_result.pose
                confidence = result.confidence
                now = time.monotonic()
                debug_status = None
                if self._vrchat_calibrate_event.is_set():
                    fusion.reset_calibration()
                    if debug_scene is not None:
                        debug_scene.reset_world()
                    vrchat_solver.begin_calibration()
                    self._vrchat_calibrate_event.clear()
                    last_vrchat_frame = None
                    calibration_active = True
                    calibration_log_step = -1
                    camera_calibrated_logged = len(captures) == 1
                    self.callbacks.log("INFO", "Camera T-pose calibration: keep shoulders, wrists, hips, knees and feet visible in every camera")
                    # Re-run this frame through the newly reset fusion state.
                    fusion_result = fusion.update(observations, cv2)
                    result, confidence = fusion_result.pose, fusion_result.pose.confidence
                if len(captures) > 1 and not fusion_result.calibrated:
                    current, total = fusion_result.calibration_progress
                    debug_status = f"T-POSE CALIBRATION {current:04.1f}/{total:.0f}s — {fusion_result.calibration_hint}"
                    usable_pose = False
                else:
                    usable_pose = result.detected and confidence >= 0.35
                    if not camera_calibrated_logged:
                        camera_calibrated_logged = True
                        self.callbacks.log("PASS", f"Camera geometry calibrated; fusing {fusion_result.contributing_cameras} views")
                geometry_warning = fusion_result.calibration_hint if (fusion_result.calibrated or fusion_result.safety_paused) and fusion_result.calibration_hint != "T-pose calibration complete" else ""
                if geometry_warning != last_geometry_warning:
                    if geometry_warning:
                        self.callbacks.log("WARN", geometry_warning)
                    elif last_geometry_warning:
                        self.callbacks.log("INFO", "Camera geometry checks recovered")
                    last_geometry_warning = geometry_warning
                if getattr(fusion_result, 'safety_paused', False):
                    usable_pose = False
                    last_vrchat_frame = None  # geometry faults must not resend stale trackers
                    debug_status = geometry_warning
                if usable_pose:
                    mapped_pose.Update(result.world_landmarks, smooth=profile.smooth)
                    vrchat_frame = vrchat_solver.solve(mapped_pose.KeyPoints, timestamp=now, smooth=profile.smooth)
                    if vrchat_frame is None:
                        current, total = vrchat_solver.calibration_progress
                        debug_status = "VRCHAT BODY CALIBRATION — HOLD STILL" if current == total else f"VRCHAT BODY CALIBRATION {current}/{total}"
                        log_step = current // 5
                        if log_step != calibration_log_step:
                            calibration_log_step = log_step
                            self.callbacks.log("INFO", f"VRChat body scale/forward calibration {current}/{total}")
                        had_pose = False
                    else:
                        if calibration_active:
                            calibration_active = False
                            self.callbacks.log("PASS", f"Stable calibration complete: body scale {vrchat_frame.scale:.3f}×, forward correction {vrchat_solver.neutral_yaw:+.1f}°")
                        align_head = self._vrchat_align_event.is_set() and vrchat_frame.head_confidence >= 0.5
                        send_started = time.monotonic()
                        packets = self._send_vrchat_frame(vrchat_frame, osc, OSCKit, align_head=align_head, tracker_ids=tracker_ids)
                        self.metrics.stage("osc_send", (time.monotonic() - send_started) * 1000, packets)
                        sent_frames += bool(packets)
                        last_vrchat_frame, last_pose_at = vrchat_frame, now
                        debug_status = "TRACKING"
                        if align_head:
                            self._vrchat_align_event.clear()
                            self.callbacks.log("PASS", f"VRChat space aligned for {profile.user_height_m:.2f} m user height")
                        if not had_pose:
                            self.callbacks.log("PASS", "Pose acquired; sending stabilized VRChat trackers")
                        had_pose = True
                elif last_vrchat_frame is not None and now - last_pose_at <= 0.35 and not vrchat_solver.calibrating:
                    send_started = time.monotonic()
                    packets = self._send_vrchat_frame(last_vrchat_frame, osc, OSCKit, align_head=False, tracker_ids=tracker_ids)
                    self.metrics.stage("osc_send", (time.monotonic() - send_started) * 1000, packets)
                    sent_frames += bool(packets)
                    debug_status = "HOLDING BRIEF OCCLUSION"
                else:
                    if had_pose:
                        self.callbacks.log("WARN", "Pose lost for over 350 ms; OSC output paused until tracking recovers")
                    had_pose = False
                if geometry_warning:
                    debug_status = geometry_warning
                frames += 1
                actual_fps = self.metrics.update()
                if frames == 1 or frames % max(1, settings.fps // 2) == 0:
                    self.callbacks.stats(actual_fps, confidence, sent_frames)
                if debug_scene is not None:
                    status = debug_status or ("TRACKING" if usable_pose else ("LOW CONFIDENCE" if result.detected else "NO POSE"))
                    debug_frame = debug_scene.render(
                        result.world_landmarks,
                        fusion_result.camera_poses,
                        confidence,
                        actual_fps,
                        status,
                        fusion_result.contributing_cameras,
                        floor_y=fusion_result.room_floor,
                    )
                    cv2.imshow(debug_window, debug_frame)
                    native_mosaic = render_camera_mosaic(camera_views, cv2)
                    if not camera_window_sized:
                        native_height, native_width = native_mosaic.shape[:2]
                        initial_scale = min(1.0, 1600 / native_width, 900 / native_height)
                        cv2.resizeWindow(
                            camera_window,
                            max(320, round(native_width * initial_scale)),
                            max(240, round(native_height * initial_scale)),
                        )
                        camera_window_sized = True
                    try:
                        _x, _y, viewport_width, viewport_height = cv2.getWindowImageRect(camera_window)
                        viewport = (max(1, viewport_width), max(1, viewport_height))
                    except (AttributeError, cv2.error):
                        viewport = (native_mosaic.shape[1], native_mosaic.shape[0])
                    cv2.imshow(camera_window, fit_image_to_viewport(native_mosaic, viewport, cv2))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        self._stop_event.set()
                    elif key in (ord("r"), ord("R")):
                        debug_scene.reset_view()
                next_frame += target_interval
                delay = next_frame - time.monotonic()
                if delay > 0:
                    self._stop_event.wait(delay)
                elif delay < -target_interval * 2:
                    next_frame = time.monotonic()
        except Exception as exc:
            if not self._stop_event.is_set():
                final_state = "error"
                if opening_source is not None:
                    self.metrics.fail(opening_source)
                self.callbacks.log("ERROR", f"{type(exc).__name__}: {exc}")
                self.callbacks.log("DEBUG", traceback.format_exc())
        finally:
            self._stop_event.set()
            def cleanup(label, action):
                nonlocal final_state
                try:
                    action()
                except Exception as exc:
                    final_state = "error"
                    self.callbacks.log("ERROR", f"Could not close {label}: {exc}")

            for _source_id, _label, reader in capture_readers:
                reader.stop()
            owned_captures = {id(reader.capture) for _, _, reader in capture_readers}
            for _source_id, label, capture in captures:
                if id(capture) not in owned_captures:
                    cleanup(label, capture.release)
            for _source_id, _label, reader in capture_readers:
                reader.join(2.5)
            if inference_pool is not None:
                cleanup("inference workers", inference_pool.close)
            for pose in poses.values():
                cleanup("pose model", pose.close)
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
            if osc is not None:
                cleanup("OSC socket", osc.close)
            self.metrics.finish()
            self.callbacks.health(self.metrics.snapshot())
            self.callbacks.state(final_state)
            if final_state == "stopped":
                self.callbacks.log("INFO", "Tracking stopped")

    @staticmethod
    def _send_vrchat_frame(
        frame,
        server,
        osc_module,
        align_head: bool = False,
        follow_head: bool = True,
        tracker_ids: set[str] | None = None,
    ) -> int:
        packets = 0
        for tracker_id, tracker in frame.trackers.items():
            explicit_eligibility = getattr(tracker, "output_eligible", None)
            output_eligible = tracker.confidence >= 0.5 if explicit_eligibility is None else explicit_eligibility
            if (not output_eligible or (tracker_ids is not None and tracker_id not in tracker_ids)
                    or not all(math.isfinite(value) for value in (*tracker.position, *tracker.rotation, tracker.confidence))):
                continue
            base = osc_module.Const.BasePath
            server.Send(osc_module.Phrase.direct(base, tracker_id, osc_module.Const.Type.Position, list(tracker.position)))
            server.Send(osc_module.Phrase.direct(base, tracker_id, osc_module.Const.Type.Rotation, list(tracker.rotation)))
            packets += 2
        valid_head = frame.head_confidence >= 0.5 and all(
            math.isfinite(value) for value in (*frame.head_position, *frame.head_rotation, frame.head_confidence)
        )
        if follow_head and valid_head:
            base = osc_module.Const.BasePath
            server.Send(osc_module.Phrase.direct(base, "head", osc_module.Const.Type.Position, list(frame.head_position)))
            packets += 1
        if align_head and valid_head:
            base = osc_module.Const.BasePath
            server.Send(osc_module.Phrase.direct(base, "head", osc_module.Const.Type.Rotation, list(frame.head_rotation)))
            packets += 1
        return packets
