# Readiness assessment — 16 September 2026

The maintained root application is suitable for a supervised multi-camera first-use trial after installation diagnostics pass. A fresh Python environment on this PC passes the software gate. Physical VRChat alignment, long-session reliability, and installation on a clean PC have not been verified in this revision. This is an experimental camera tracker, not a signed end-user release.

**Rendered-human acceptance still fails the coverage target.** The optional DWPose + Full mode and updated fusion improve three-camera confident-joint mean/p95 error to 5.4/11.1 cm, but only about 80% of evaluated joints are confident and within 20 cm (target 90%). Three concurrent CPU workers take roughly 79–82 ms per frame at the median. See the [model comparison](review/simulation/MODEL_COMPARISON.md).

**Earlier baseline:** A 32-second animated human fixture processed by the actual Full pose model calibrated with two front views or three front views, but eligible-joint mean error was 13–14 cm and p95 was 48–49 cm. A side-on third view prevented calibration. Exact-joint controls passed normal/jitter conditions. See [the simulation report](review/simulation/REPORT.md) for both layouts, camera images, thresholds, limitations and reproducible commands. These results override any interpretation of software-test success as tracking-readiness evidence.

Measured lens import/correction, a checkerboard calibration tool with held-out validation, camera-geometry fault handling and immediate stale-output suppression are now implemented. Deliberately degraded third-camera images retained about 79% reliable coverage; an incorrect camera FOV was excluded. This remains below acceptance. See [the hardening report](review/simulation/HARDENING.md) and [real-camera lens setup](CAMERA_CALIBRATION.md).

## Improvements implemented in this revision

| Area | Change and practical effect |
| --- | --- |
| Installation | `setup.bat` creates a local environment, checks 64-bit Python 3.12, installs dependencies, checks their consistency, and runs diagnostics. Failed steps stop setup. |
| Dependencies | A fresh Python environment installed all requirements and passed verification; `requirements-lock.txt` pins that complete tested package set. |
| Diagnostics | Check the selected default profile and actually import runtime dependencies, catching missing profiles and many native-library failures before tracking. GUI diagnostics run off the event thread. |
| Configuration | Reject duplicate/unknown camera setups, malformed transforms, and invalid indices in the converted 31-point skeleton. Failed atomic file replacements retain the original and remove temporary files. Save errors are shown to the user. |
| Capture | Surface capture worker exceptions with the affected source name. Read and release the native camera on its owning thread, avoiding concurrent native calls. |
| Native isolation | Local capture and pose models run in supervised subprocesses. Startup has a 15-second deadline, native requests a three-second deadline, and Stop interrupts a pending request. Unresponsive workers are terminated. |
| Multi-camera scheduling | Keep at most one inference outstanding per source; poll completions independently. Pair recent results across different source rates without blocking healthy cameras behind a slow model. |
| Synchronization/calibration | Bounded pose histories align views at an adaptive 100–200 ms playback horizon with 33 ms maximum timestamp separation; results older than 250 ms are excluded. Timestamped JPEG uses phone/PC clock estimation; WebRTC retains media timing to expose extra buffering. Measured residual device delay is configurable per camera. Reused camera frames and observed brief dropouts do not create extra calibration samples/hold time. |
| Rotated cameras | Quarter-turn image correction now uses the native image width for focal length rather than the rotated width. |
| Cleanup | A resource close failure no longer skips the remaining model/socket cleanup or the final controller state notification. |
| Numerical validity | Invalid model coordinates/confidences are withdrawn. Non-finite points cannot poison map history; a final OSC guard withholds non-finite tracker/head output. |
| Phone recovery | Incomplete offer bodies and negotiation have 10-second deadlines. WebRTC streams with no received video expire after 15 seconds, freeing their slots for fallback/reconnection. Stale peer cleanup checks ownership before cancelling tasks. |
| Phone history | Retain at most 32 offline identities, evicting the longest-disconnected entries without removing connected cameras. Reconnecting an evicted identity advances past old frame cursors. Saved profile identities remain available independently of this discovery history. |
| Error reporting | Rotating logs, token redaction for logged messages, startup error dialogs, Tk callback error dialogs, and a bounded 2,000-line on-screen log. |
| Support export | GUI and headless CLI save a local ZIP with package versions, anonymized rig configuration, saved-file diagnostics and bounded log tails. Broken configuration does not prevent export. Tokens, device IDs, hosts and user-home paths are scrubbed; video and certificates are excluded. |
| Setup usability | Start/Stop/Recalibrate/Save have a fixed footer. The setup form scrolls and changes to one column on narrow windows, and focused fields scroll into view. Native mapped-window tests cover 940×640 and 1120×760. The previous layout hid Start at both sizes. |
| Profile persistence | Save validates global settings before modifying the rig, restores the profile on a reported settings-write failure, and refreshes the in-memory saved state. Save as creates a separate startup profile while preserving the original. Windows reserved names and case-insensitive duplicates are rejected. Rollback does not provide crash atomicity across two files. |
| Runtime measurements | Bounded thread-safe per-source capture counts/rates, frame age, inference and capture-to-result percentiles, newer-frame replacement counts and failures; fusion and OSC-send timings and packet counts. The GUI and local support export retain the latest session snapshot. |
| Simulation-driven corrections | Completion jitter adapts the alignment buffer within 100–200 ms. Without a current body transform, inferred fallback points no longer carry enough confidence to emit a stale room position. This pauses output when necessary; it does not restore missing views or fix detector errors. |
| Repeatable checks | `scripts/verify.py` runs compilation, dependency checks, diagnostics, pytest, browser checks and the earlier defect reproductions. Windows CI runs the same command. |

Existing saved phone IDs, user height, room layout, model bundles, and profile settings were preserved. The existing source package already had 89 passing Python tests before these changes.

## Verification evidence

Environment: existing repository `.venv`, Windows, Python 3.12.9, OpenCV 5.0.0, MediaPipe 1.0.1; Git for Windows OpenSSL; installed Node.js. Tests use synthetic cameras/poses, real local UDP/HTTPS/WebRTC traffic, real model initialization, and actual Tk windows. Phone JavaScript checks use browser adapters, not a physical mobile browser.

Commands:

```powershell
.venv\Scripts\python.exe scripts/verify.py
git diff --check
```

Latest full gate: **167 Python tests passed in 37.82 seconds**, followed by both phone JavaScript suites, all earlier defect reproductions, compilation, dependency consistency, diagnostics and model hashes; the runner exited with code 0. GUI cases run in fresh subprocesses and a failure-propagation test verifies that a failing child cannot silently pass. An earlier fresh-environment check passed the then-current 110 tests in 32.66 seconds before the support-export and layout additions. Multi-camera tests cover independent inference scheduling, timestamps, lens geometry, calibration reuse/dropout cases, actual subprocess timeout/cancellation, real model IPC, and two/three-phone controller lifecycle checks using synthetic frames. No hardware result from an earlier audit is represented as a test of this revision.

Three concurrent Full models on 1280×720 black images, 100 measured calls per camera after five warmups, took 3.35 seconds to start. Inference plus local process communication measured p50 **21.0–22.2 ms**, p95 **25.9–27.7 ms**, p99 **27.4–28.8 ms** across the cameras. See `review/multicamera_benchmark.json`. This no-person fixture is an IPC/detector baseline, not a person-tracking or end-to-end motion latency result.

Support-export coverage verifies archive contents/redaction, input size limits, failed-write preservation, missing logs, broken configuration, headless export, cancelled GUI saves and completion feedback. Layout coverage processes actual native mapping events and checks footer visibility and keyboard-focused field bounds at both supported window sizes.

Repeated Tk creation in the mixed test process intermittently failed while reading `init.tcl` or `tk.tcl` resources. Separate probes passed 100 plain Tk lifecycles and 40 application-window lifecycles without phone startup. The root cause remains unproven. GUI tests now run once each in fresh processes, matching production launch semantics and avoiding cross-case native state; this is test isolation, not a claim to have fixed Tcl. Hosted CI and `setup.bat` on a new Windows installation have not been run here.

## Prioritized remaining improvements

This inventory covers the inspected subsystems; no finite review can establish that every possible improvement or defect has been found.

| Priority | Work | Completion evidence |
| --- | --- | --- |
| Before claiming reliable everyday tracking | Run the camera/phone/headset/avatar acceptance sequence below, first with hip + feet, then optionally all eight trackers. | Recorded device/browser/avatar versions and observed results. |
| Hardware recovery verification | Supervised native workers now pass injected startup/request hangs and cancellation checks. Verify actual webcam drivers, phone interruptions and suspend/resume on the intended rig. | Physical stop/restart and repeated reconnection evidence; no remaining worker processes. |
| Before distributing binaries | Build and smoke-test on a clean Windows account; resolve licenses and provenance, produce an installer, move mutable profiles into per-user storage with migration, and sign official releases. | Fresh-OS install evidence, license inventory, migration tests, install/uninstall results and signing identity. The Python package graph is already pinned and fresh-venv tested. |
| Tracking accuracy | Collect and import measured lens calibration for the actual cameras using the implemented tool; validate timing estimates and measured residual delays on actual cameras; validate room localization during turns and partial occlusions. | Known geometry/motion fixtures and measured multi-view position error. |
| Latency and stability | Use the implemented runtime timing/rate/age/replacement counters to measure Lite/Full/Heavy/DWPose with one, two and three real cameras. Add process-memory measurement and complete the hour-long soak. | p50/p95/p99 timing and memory over an hour on stated hardware. PC-side timings are not motion-to-avatar latency. |
| Recovery UX | Add source-specific reconnect actions, camera preview before tracking, clearer calibration instructions, and trusted-certificate troubleshooting for supported phones. | First-use sessions without developer assistance. |
| Settings UX | Add profile rename, inline validation, a first-run wizard and high-DPI layout checks. Save as now creates/duplicates rigs; scrollable setup and keyboard focus visibility are tested. | Profile migration/rename checks, first-use studies and high-DPI usability tests. |
| Diagnostics UX | Extend measured support exports with process-memory summaries and physical acceptance results. Runtime timing and source health are implemented. | Memory evidence and validation of any new exported fields. |
| Maintainability | Separate the large tracking/GUI modules, introduce an installable package and gradual type/lint checks, and retire duplicate legacy assets after checking compatibility consumers. | Package import/install tests, migration notes, and no duplicate production implementation. |
| Long-session phone service | Stress repeated joins, tab suspension, Wi-Fi changes, server stop/start, packet corruption and repeated device identities. Offline discovery history is now bounded. | Soak tests and stable task/socket/memory counts. |

The rendered accuracy failure justified adding an optional refinement model. Further GPU paths, model replacements and large rewrites should also have measured acceptance criteria. The current LAN-only phone workflow is preserved.

## Physical acceptance sequence

1. Run setup/diagnostics; launch with `run.bat`. Verify camera selection, current user height, and the saved room layout. A saved offline phone must be reconnected or explicitly replaced with an available camera.
2. Start with one camera and **Stable — hip + feet**. Keep the whole body visible. Complete application alignment and VRChat avatar calibration.
3. Check neutral standing: hip height, both feet on the floor, and left/right movement. Then test a short walk, quarter turns, crouching, and lifting each foot separately. Record drift, jitter or incorrect orientation.
4. Cover each leg briefly, then for over a second. Check bounded hold, paused unreliable output, and reacquisition without an extreme jump.
5. Stop/start at least five times. Close a debug window, disconnect the camera/phone, interrupt Wi-Fi, and close the app during startup. Record any hangs or remaining Python processes. Native workers have software deadlines, but actual device recovery still needs this check.
6. For each intended phone/browser, test permission refusal, cancelled permission, certificate handling, direct WebRTC, compatibility fallback, tab suspension, and reconnecting with the current link. Verify another healthy camera continues while a phone stops sending.
7. For multiple cameras, calibrate with a T-pose visible in every view; verify camera movement forces recalibration and that the localized body remains upright and inside the room.
8. Run a 60-minute session. Record CPU/memory, sustained FPS, visible latency, disconnects and calibration drift. Repeat with the actual headset/avatar used day to day.

Do not interpret successful UDP sends as confirmation that VRChat received or aligned trackers. A human must observe the receiver.
