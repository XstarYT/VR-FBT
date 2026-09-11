"""Background tracking controller used by the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import traceback
from typing import Callable

from Lib.Config import Profile, Settings, load_joint_map


LogCallback = Callable[[str, str], None]
StateCallback = Callable[[str], None]
StatsCallback = Callable[[float, float, int], None]


class _LatestFrameReader:
    """Continuously capture one source without blocking the tracking loop."""

    def __init__(self, capture, label: str):
        self.capture = capture
        self.label = label
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sequence = 0
        self._frame = None
        self._captured_at = 0.0
        self._thread = threading.Thread(
            target=self._run, name=f"vr-fbt-capture-{label}", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.capture.isOpened():
                self._stop.wait(0.05)
                continue
            ok, frame = self.capture.read()
            captured_at = time.perf_counter()
            if not ok or frame is None:
                self._stop.wait(0.005)
                continue
            with self._lock:
                self._sequence += 1
                self._frame = frame
                self._captured_at = captured_at

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


class TrackingController:
    """Own one tracking worker and expose deterministic start/stop semantics."""

    def __init__(self, callbacks: EngineCallbacks | None = None):
        self.callbacks = callbacks or EngineCallbacks()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._vrchat_align_event = threading.Event()
        self._vrchat_calibrate_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings: Settings, profile: Profile, remote_hub=None) -> None:
        if self.running:
            raise RuntimeError("Tracking is already running")
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
        try:
            self.callbacks.state("starting")
            self.callbacks.log("INFO", "Loading the pose model and compute backend…")
            import cv2
            from Lib import Data, OSCKit
            from Lib.Config import profile_camera_setups, profile_camera_sources
            from Lib.DebugView import DebugScene3D, fit_image_to_viewport, render_camera_mosaic, rotate_camera_frame
            from Lib.Tracking import CameraObservation, MultiCameraPoseFusion, Pose
            from Lib.VRChat import VRChatPoseSolver

            joint_map = load_joint_map(profile.joint_map)
            mapped_pose = Data.Map(joint_map)
            source_ids = profile_camera_sources(profile)
            camera_setups = profile_camera_setups(profile)
            image_rotations = {setup.source_id: setup.image_rotation for setup in camera_setups}
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
                if source_id.startswith("phone:"):
                    if remote_hub is None or not remote_hub.running:
                        raise RuntimeError("Start the phone camera server before using a phone source")
                    from Lib.RemoteCam import RemoteCapture
                    device_id = source_id.split(":", 1)[1]
                    capture = RemoteCapture(remote_hub.registry, device_id, cv2, timeout=0.30)
                    source_label = f"phone camera {device_id}"
                else:
                    local_index = int(source_id.split(":", 1)[1])
                    capture = cv2.VideoCapture(local_index, getattr(cv2, "CAP_DSHOW", 0))
                    source_label = f"local camera {local_index}"
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open {source_label}")
                if hasattr(capture, "set"):
                    capture.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
                captures.append((source_id, source_label, capture))
                poses[source_id] = Pose(profile.pose_quality)
            for source_id, source_label, capture in captures:
                reader = _LatestFrameReader(capture, source_id.replace(":", "-"))
                reader.start()
                capture_readers.append((source_id, source_label, reader))
            inference_pool = ThreadPoolExecutor(max_workers=len(captures), thread_name_prefix="vr-fbt-pose")

            target_interval = 1.0 / settings.fps
            next_frame = started = time.perf_counter()
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
            missing_sources: set[str] = set()
            last_any_frame = time.perf_counter()
            capture_sequences = {source_id: 0 for source_id in source_ids}

            while not self._stop_event.is_set():
                observations = []
                pending_inference = []
                camera_views = []
                capture_now = time.perf_counter()
                for source_id, label, reader in capture_readers:
                    latest = reader.latest(capture_sequences[source_id])
                    if latest is None:
                        silent_for = capture_now - reader.last_frame_at if reader.last_frame_at else capture_now - started
                        if silent_for >= 0.5 and source_id not in missing_sources:
                            missing_sources.add(source_id)
                            self.callbacks.log("WARN", f"{label.capitalize()} is not returning fresh frames")
                        continue
                    sequence, captured_at, frame = latest
                    capture_sequences[source_id] = sequence
                    if capture_now - captured_at > max(0.25, target_interval * 4):
                        if source_id not in missing_sources:
                            missing_sources.add(source_id)
                            self.callbacks.log("WARN", f"{label.capitalize()} frame is stale")
                        continue
                    if source_id in missing_sources:
                        missing_sources.remove(source_id)
                        self.callbacks.log("PASS", f"{label.capitalize()} resumed")
                    frame = rotate_camera_frame(frame, image_rotations.get(source_id, 0), cv2)
                    height, width = frame.shape[:2]
                    future = inference_pool.submit(poses[source_id].process, frame)
                    pending_inference.append((source_id, label, frame, (width, height), captured_at, future))
                for source_id, label, frame, frame_size, captured_at, future in pending_inference:
                    camera_result = future.result()
                    observations.append(CameraObservation(source_id, camera_result, frame_size, captured_at))
                    camera_views.append((frame, camera_result, label))
                if observations:
                    freshest = max(item.captured_at or 0.0 for item in observations)
                    maximum_skew = max(0.12, target_interval * 2)
                    observations = [
                        item for item in observations
                        if freshest - (item.captured_at or freshest) <= maximum_skew
                    ]
                if not observations:
                    fusion.update([], cv2)
                    if time.perf_counter() - last_any_frame > 3.0:
                        raise RuntimeError("No selected camera has returned a frame for three seconds")
                    self._stop_event.wait(min(target_interval, 0.05))
                    continue
                last_any_frame = time.perf_counter()
                fusion_result = fusion.update(observations, cv2)
                result = fusion_result.pose
                confidence = result.confidence
                now = time.perf_counter()
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
                    self.callbacks.log("INFO", "Calibrating: hold a T-pose with shoulders, wrists, hips, knees and feet visible in every camera")
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
                if usable_pose:
                    mapped_pose.Update(result.world_landmarks, smooth=profile.smooth)
                    vrchat_frame = vrchat_solver.solve(mapped_pose.KeyPoints, timestamp=now, smooth=profile.smooth)
                    if vrchat_frame is None:
                        current, total = vrchat_solver.calibration_progress
                        debug_status = "CALIBRATING — HOLD STILL" if current == total else f"CALIBRATING {current}/{total}"
                        log_step = current // 5
                        if log_step != calibration_log_step:
                            calibration_log_step = log_step
                            self.callbacks.log("INFO", f"Stable-pose calibration {current}/{total}")
                        had_pose = False
                    else:
                        if calibration_active:
                            calibration_active = False
                            self.callbacks.log("PASS", f"Stable calibration complete: body scale {vrchat_frame.scale:.3f}×, forward correction {vrchat_solver.neutral_yaw:+.1f}°")
                        align_head = self._vrchat_align_event.is_set() and vrchat_frame.head_confidence >= 0.5
                        self._send_vrchat_frame(vrchat_frame, osc, OSCKit, align_head=align_head, tracker_ids=tracker_ids)
                        sent_frames += 1
                        last_vrchat_frame, last_pose_at = vrchat_frame, now
                        debug_status = "TRACKING"
                        if align_head:
                            self._vrchat_align_event.clear()
                            self.callbacks.log("PASS", f"VRChat space aligned for {profile.user_height_m:.2f} m user height")
                        if not had_pose:
                            self.callbacks.log("PASS", "Pose acquired; sending stabilized VRChat trackers")
                        had_pose = True
                elif last_vrchat_frame is not None and now - last_pose_at <= 0.35 and not vrchat_solver.calibrating:
                    self._send_vrchat_frame(last_vrchat_frame, osc, OSCKit, align_head=False, tracker_ids=tracker_ids)
                    sent_frames += 1
                    debug_status = "HOLDING BRIEF OCCLUSION"
                else:
                    if had_pose:
                        self.callbacks.log("WARN", "Pose lost for over 350 ms; OSC output paused until tracking recovers")
                    had_pose = False
                frames += 1
                elapsed = max(time.perf_counter() - started, 1e-6)
                actual_fps = frames / elapsed
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
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    self._stop_event.wait(delay)
                elif delay < -target_interval * 2:
                    next_frame = time.perf_counter()
        except Exception as exc:
            final_state = "error"
            self.callbacks.log("ERROR", f"{type(exc).__name__}: {exc}")
            self.callbacks.log("DEBUG", traceback.format_exc())
        finally:
            for _source_id, _label, reader in capture_readers:
                reader.stop()
            for _source_id, _label, capture in captures:
                capture.release()
            for _source_id, _label, reader in capture_readers:
                reader.join()
            if inference_pool is not None:
                inference_pool.shutdown(wait=True, cancel_futures=True)
            for pose in poses.values():
                pose.close()
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
            if osc is not None:
                osc.close()
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
    ) -> None:
        for tracker_id, tracker in frame.trackers.items():
            explicit_eligibility = getattr(tracker, "output_eligible", None)
            output_eligible = tracker.confidence >= 0.5 if explicit_eligibility is None else explicit_eligibility
            if not output_eligible or (tracker_ids is not None and tracker_id not in tracker_ids):
                continue
            base = osc_module.Const.BasePath
            server.Send(osc_module.Phrase.direct(base, tracker_id, osc_module.Const.Type.Position, list(tracker.position)))
            server.Send(osc_module.Phrase.direct(base, tracker_id, osc_module.Const.Type.Rotation, list(tracker.rotation)))
        if follow_head and frame.head_confidence >= 0.5:
            base = osc_module.Const.BasePath
            server.Send(osc_module.Phrase.direct(base, "head", osc_module.Const.Type.Position, list(frame.head_position)))
        if align_head and frame.head_confidence >= 0.5:
            base = osc_module.Const.BasePath
            server.Send(osc_module.Phrase.direct(base, "head", osc_module.Const.Type.Rotation, list(frame.head_rotation)))
