"""Background tracking controller used by the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import traceback
from typing import Callable

from Lib.Config import Profile, Settings, load_joint_map


LogCallback = Callable[[str, str], None]
StateCallback = Callable[[str], None]
StatsCallback = Callable[[float, float, int], None]


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
        capture = None
        osc = None
        pose = None
        final_state = "stopped"
        try:
            self.callbacks.state("starting")
            self.callbacks.log("INFO", "Loading the pose model and compute backend…")
            import cv2
            from Lib import Data, OSCKit
            from Lib.DebugView import render_tracking_debug
            from Lib.Tracking import Pose
            from Lib.VRChat import VRChatPoseSolver

            joint_map = load_joint_map(profile.joint_map)
            mapped_pose = Data.Map(joint_map)
            pose = Pose(profile.pose_quality)
            vrchat_solver = VRChatPoseSolver(profile.user_height_m, calibration_frames=20)
            osc = OSCKit.Server(profile.server_ip, profile.server_port)
            self.callbacks.log("PASS", f"VRChat OSC output resolved to IPv4 {osc.target[0]}:{osc.target[1]}")
            if profile.camera_source.startswith("phone:"):
                if remote_hub is None or not remote_hub.running:
                    raise RuntimeError("Start the phone camera server before using a phone source")
                from Lib.RemoteCam import RemoteCapture
                device_id = profile.camera_source.split(":", 1)[1]
                capture = RemoteCapture(remote_hub.registry, device_id, cv2)
                source_label = f"phone camera {device_id}"
            else:
                local_index = int(profile.camera_source.split(":", 1)[1])
                capture = cv2.VideoCapture(local_index, getattr(cv2, "CAP_DSHOW", 0))
                source_label = f"local camera {local_index}"
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {source_label}")
            if hasattr(capture, "set"):
                capture.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)

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
            self.callbacks.log("INFO", f"Tracking {source_label} with MediaPipe {profile.pose_quality} to {profile.server_ip}:{profile.server_port}")
            self.callbacks.log("INFO", "VRChat tracker set: hip + feet (stable)" if tracker_ids else "VRChat tracker set: all 8 (experimental)")
            self.callbacks.state("running")

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"{source_label.capitalize()} stopped returning frames")
                result = pose.process(frame)
                confidence = result.confidence
                now = time.perf_counter()
                debug_status = None
                if self._vrchat_calibrate_event.is_set():
                    vrchat_solver.begin_calibration()
                    self._vrchat_calibrate_event.clear()
                    last_vrchat_frame = None
                    calibration_active = True
                    calibration_log_step = -1
                    self.callbacks.log("INFO", "Calibrating: stand upright facing the camera with shoulders, hips, knees and feet visible")
                usable_pose = result.detected and confidence >= 0.35
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
                if profile.show_output:
                    debug_frame = render_tracking_debug(
                        frame, result.image_landmarks, confidence, actual_fps, frames, source_label, cv2,
                        detected=result.detected,
                        tracking_status=debug_status,
                    )
                    cv2.imshow("VR-FBT Tracking Debug (press Q to stop)", debug_frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        self._stop_event.set()
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
            if capture is not None:
                capture.release()
            if pose is not None:
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
            if tracker.confidence < 0.5 or (tracker_ids is not None and tracker_id not in tracker_ids):
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
