# VR-FBT for VRChat

Windows camera-based full-body tracking for VRChat. VR-FBT runs Google's MediaPipe Pose Landmarker locally, turns its metric body landmarks into eight OSC trackers for the body areas supported by VRChat, and sends position and rotation messages directly to VRChat on the same PC.

> This is an experimental optical tracker. Automated tests verify the software pipeline, but final body alignment still needs to be checked in VRChat with a real person, camera, headset, and avatar.

## Requirements

- Windows 10 or 11
- 64-bit Python 3.12
- VRChat for Windows with OSC enabled
- One to three local webcams/phones; phones and PC must share the same local network
- OpenSSL (Git for Windows' bundled OpenSSL is detected automatically)

## Recommended: one or two phones with automatic geometry

Start the phone server, connect each phone on the same LAN, then select the connected phone cameras. For two phones, place one in front and one to the side and hold a full-body T-pose visible to both for 10 seconds. The app estimates their relative camera positions. With one phone, face the camera; the VRChat body scale and forward lock takes 20 tracking frames. The stock Default profile uses automatic geometry and a local-camera placeholder so it does not point to someone else's saved phone ID. Select your own sources before Start.

Keep **Anchor cameras at fixed positions in the virtual room** off unless you have measured the room and camera mounts. The room and corner coordinates shown in Default are placeholders in automatic mode. Choosing a corner preset intentionally enables fixed-room mode. Each camera's estimated horizontal FOV starts at 60°; if knees or feet drift near the image edges, try 55–70° or import a measured lens calibration. The Camera layout dialog shows a connected phone's native frame size and estimated focal length after a frame arrives. [Checkerboard lens calibration](CAMERA_CALIBRATION.md) is useful when calibration reports high reprojection error.

Saved phones that are offline are reported before tracking starts. If another selected camera is available, you can start that session with the available subset; the saved source selection remains intact. The interface labels tracking accuracy experimental and recommends the Stable hip + feet tracker set.

## Install and run

For guided source setup, install 64-bit Python 3.12 and Git for Windows, then double-click **setup.bat**. It creates the local environment if needed, installs dependencies, and runs diagnostics. It preserves an existing environment and reports an incompatible Python version instead of deleting it. Alternatively, use the commands below.

`requirements.txt` applies the tested Windows/Python 3.12 versions from `requirements-lock.txt`, including transitive dependencies. Update the lock deliberately after running verification in a fresh environment.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Main.py --check
python Main.py
```

After setup, `run.bat` starts the repository-local environment without manual activation.

On first launch, the setup assistant checks dependencies, sends an OSC test pulse, guides phone or local camera connection, asks for standing height, and saves the Stable tracker set by default. You can reopen it from **Setup assistant**. **Preview cameras** shows frames and resolution in the main window without starting pose inference; stop preview before tracking. A phone's browser camera or security error appears in Activity when the page can reach the local hub.

Start, Stop, Recalibrate and Save remain in the fixed bottom bar. Setup switches to a single column on narrower windows; use the scrollbar or mouse wheel to reach the remaining fields. Keyboard focus brings off-screen fields into view.

Use **Save as…** to keep a separate camera layout under a new profile name. It preserves the original rig and makes the new profile the startup default. Save validates the profile and global settings before writing; if the settings write fails, it attempts to restore the previous profile and reports any rollback failure explicitly.

Startup and runtime errors are saved in `%LOCALAPPDATA%\VR-FBT\logs\vr-fbt.log` (up to four files of approximately 2 MB each). Phone URL tokens in log messages are redacted. Startup failures display an error dialog even when launched without a console. If an action fails, run diagnostics and check this log. Configuration remains in `Content`; keep this source checkout in a writable folder.

To collect diagnostics, open **Activity & diagnostics → Export support ZIP…** or run `python Main.py --support-bundle support.zip`. The ZIP contains dependency versions, anonymized camera settings, saved-configuration checks, and bounded log tails. GUI exports describe the current visible settings, including unsaved edits; saved-file checks are labelled separately. Invalid configuration is reported without blocking export. No images, body landmarks, certificates, or private keys are included, and nothing is uploaded. Known tokens, camera identifiers, hosts, and user-home paths are redacted; review custom error text before sharing.

The same tab shows per-camera capture FPS, frame age, inference p95, and whether each source is waiting, has no pose, is stale/lagging, or failed. **Update FPS** measures fusion/update iterations; it can differ from camera FPS because recent results may be reused. Rates use a two-second window. Skipped captures count newer-frame replacements before inference, not network packet loss. GUI support bundles include bounded p50/p95/p99 samples for inference plus process communication, capture-to-inference completion, fusion, and OSC sending; measurements are retained after Stop. The timing origin depends on the camera transport (see synchronization below); these timings do not measure headset/avatar motion latency.

## Use with VRChat

1. Start VRChat. Open the Action Menu, choose **OSC**, and set **Enabled** on.
2. In VR-FBT select a primary camera and optionally camera 2 and camera 3, set your real height in meters, and leave the VRChat OSC destination at `127.0.0.1:9000` when VRChat is on this PC. `localhost` is also accepted and is explicitly resolved to IPv4 because VRChat's Windows listener uses IPv4.
3. Place the camera where it sees your complete body, including both feet. Stand upright and face it.
4. Keep **VRChat trackers** on **Stable — hip + feet** first. Click **Start tracking**, stand upright facing the camera with your whole body visible, and hold still until the debug overlay changes from `CALIBRATING` to `TRACKING`. The app uses 20 stable frames to lock height scale and forward direction, sends a one-shot head-rotation alignment, and then supplies a filtered head position continuously so the hip-relative camera skeleton follows the live HMD.
5. Use VRChat's normal full-body calibration for your avatar. If the tracking space becomes offset or the camera moves, return to the neutral stance and click **Recalibrate & align**.

VR-FBT can send up to eight consistently numbered virtual trackers on every confident frame. VRChat identifies their body roles during FBT calibration; its public OSC documentation does not assign fixed body roles to particular numeric IDs. VR-FBT's ordering is:

- tracker 1: chest
- tracker 2: hip
- trackers 3–4: left and right upper arm, just above the elbow
- trackers 5–6: left and right knee
- trackers 7–8: left and right foot/ankle

The recommended **Stable** mode sends only tracker 2 (hip) and trackers 7–8 (feet). This leaves VRChat's IK free to solve the chest, knees, and arms and normally avoids an imperfect camera estimate fighting the avatar. **Full — all 8 trackers** is available for a clear, evenly lit, head-to-toe camera view, but it is experimental because VRChat explicitly notes that fewer trackers may behave better when absolute pose accuracy is limited.

For each tracker it sends both `/tracking/trackers/{1-8}/position` in meters and `/tracking/trackers/{1-8}/rotation` in Euler degrees. A confident, filtered head position keeps VRChat's tracker space attached to the headset; head rotation is a single calibration pulse. Low-confidence joints are briefly held through short occlusions and then withheld instead of sending unreliable updates. See VRChat's [OSC Trackers contract](https://docs.vrchat.com/docs/osc-trackers) and [OSC overview](https://docs.vrchat.com/docs/osc-overview).

Keep the debug output enabled while positioning cameras. VR-FBT opens two windows: **3D Tracking Debug** shows a fixed virtual room, anchored camera frustums, and the wireframe moving through room coordinates; **Camera Views** renders every annotated feed from its native received resolution. This room localization also works with one camera: the first automatic pose is locked as its anchor, or a manually selected corner supplies the physical anchor. Resizing the camera window letterboxes the mosaic instead of stretching or cropping it. In the room view, drag with the left mouse button to orbit, use the mouse wheel to zoom, press R to reset the view, or Q to stop tracking.

### Pose models

- **Full — balanced (recommended):** the best default.
- **Lite — fastest:** lower CPU use and latency.
- **Heavy — larger model (slower):** optional alternative; it performed worse than Full on our rendered-human fixture.
- **DWPose + Full — experimental refinement:** improves 2D joints before multi-camera fusion; adds CPU processing per camera. Install its verified optional weights with `.venv\Scripts\python.exe scripts/install_refinement.py`, then select this mode in setup. Full remains the default.

See the [model comparison](review/simulation/MODEL_COMPARISON.md) for measured accuracy, processing time and remaining coverage failures. DWPose retains MediaPipe's coarse 3D body estimate and currently uses CPU inference.

The official Lite, Full, and Heavy bundles are stored under `Model/mediapipe`, and diagnostics verify their SHA-256 hashes. On this PC, repeated-image model-call benchmarks measured about 122 FPS Lite, 83 FPS Full, and 24 FPS Heavy; results on other PCs will differ. MediaPipe's [Pose Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) explains the image/world landmarks and temporal video mode.

Local camera capture and each pose model run in separate supervised processes. A native read/inference request that takes over three seconds fails that source; cancellation also stops its worker. A slow model does not block results from healthy cameras, and only one inference request per source is outstanding. For a repeatable synthetic multi-camera timing baseline, run `python scripts/benchmark.py --cameras 3 --quality full`; this includes local process communication but does not measure real motion-to-avatar latency.

## Use a phone camera on the local network

There is one phone connection method: direct local-network HTTPS. There is no Android developer mode, USB pairing, ADB, Cloudflare, public tunnel, or internet video relay.

1. Connect the phone and PC to the same private Wi-Fi/LAN. Guest Wi-Fi or client isolation must be off.
2. Click **Connect phone on LAN**. OpenSSL automatically generates or refreshes the local HTTPS certificate.
3. Scan the single QR code. Continue past the browser's local certificate warning if offered.
4. Press **Start camera**, allow camera access, then select the phone by its camera name under **Camera Source**.
5. If Windows Firewall asks, allow Python/VR-FBT on **Private networks** only.

The link includes a random session token and changes after restart. Keep it within your local network. WebRTC is attempted first with no STUN or TURN relay; JPEG-over-secure-WebSocket is the fallback. The receiver keeps only the newest frame so old frames do not build up into extra tracking latency. Up to three phones can connect and all three can be used by one tracking session.

## Multi-camera tracking and calibration

The desktop engine uses `Lib/Tracking.py` for T-pose calibration and fusion. The separate `vrfbt_calib` package is an experimental research path; its tests do not validate desktop behavior. Changes to desktop calibration need desktop-path tests.

For imperfect lenses, follow [measured lens calibration](CAMERA_CALIBRATION.md).
The camera-layout dialog can import a separate calibration for each camera;
fusion then corrects distortion and uses measured focal lengths. Profiles without
measured lenses keep the existing FOV estimate. Changing video aspect ratio with
a measured lens pauses tracking until the video mode or calibration is corrected.

Camera geometry is checked during tracking. A disagreeing third camera can be
excluded when the other two agree; ambiguous disagreement pauses output and
reports the reason. Persistent faults remain excluded until Recalibrate. Restore
the layout or correct its settings first: recalibration cannot repair incorrect
manually entered positions. See [imperfect-camera tests](review/simulation/HARDENING.md)
for measured fault handling and the remaining coverage failure.

Select two or three different cameras in the setup screen, then start tracking and hold a T-pose for 10 seconds with your full body visible in every view. The timer advances only while all selected cameras see the shoulders, elbows, wrists, hips, knees, and ankles in an extended-arm pose. During this phase the camera-local wireframes are compared, fixed camera transforms are estimated, and the user's neutral room location is locked. The normal 20-frame VRChat body calibration follows automatically. Place cameras at visibly different angles—front plus side is much better than putting them next to each other.

Without an imported measured lens, calibration assumes a typical 60-degree horizontal phone-camera field of view because browsers do not expose reliable lens calibration. It uses the detected T-pose skeleton as a temporary calibration target, rejects high-reprojection-error solutions, removes statistical outliers, and computes fixed camera transforms from error-weighted rotation/translation averages. A joint is triangulated when at least two views provide it. If a camera loses an entire arm or leg, the highest-confidence remaining camera takes over every joint in that limb using its calibrated wireframe transform. The existing short-occlusion hold is used only when no camera has a trustworthy observation. Camera frustums remain fixed in room coordinates while the localized skeleton moves through the room.

For the simplest physical setup, click **Camera layout…**, enter the room width/height/depth, choose one of the four corners for each camera, and enter its height in the **Height** field. Applying a corner automatically enables fixed room anchoring, puts the camera on the corresponding room corner, and aims it toward the room center. Choose **Custom coordinates** to edit X/Z/yaw/pitch/roll yourself. The same dialog has an **Image rotation** correction for a sideways or upside-down phone; choose 0°, 90°, 180°, or 270° clockwise. Rotation is applied before pose detection as well as preview rendering. Room coordinates use the center as X=0/Z=0 and the floor as Y=0.

Triangulated points are accepted only when their rays agree, remain inside the configured room, and preserve a plausible relationship to the detected torso and limb geometry. Three-camera fusion can reject a disagreeing camera using a consistent pair; nearly parallel rays are rejected. An aligned body-model fallback must agree with current image observations to remain confident. Conflicting fallback points are withheld from reliable output.

Multi-camera fusion retains bounded pose histories and selects views at a common playback horizon that adapts from 100 to 200 ms to observed completion delays. Views over 33 ms apart are not combined, and results older than 250 ms are excluded regardless of configured FPS. Missing or delayed cameras do not block healthy views; Activity marks excluded sources as out of sync. A single configured camera has no added alignment buffer. Reused results do not count as new calibration samples. Configured horizontal FOV refers to the native image before rotation correction; 90°/270° correction swaps that dimension when computing camera intrinsics.

Phone JPEG compatibility mode estimates the phone/PC monotonic clock offset using round trips of at most 50 ms, refreshes it during streaming, and stamps each frame before JPEG encoding. Where available, the browser video callback supplies capture time; otherwise presentation/draw time is used. Frames with stale, impossible, or invalid timing are withheld. Reload the phone page to use this protocol. Legacy raw JPEG senders still use PC receipt time. Clock estimates assume reasonably symmetric network delays; the 33 ms view limit is a timestamp limit, not a guarantee of exposure synchronization.

Direct WebRTC maps its relative video timestamps to the earliest observed arrival, so later network/decoder buffering no longer makes old frames look fresh. This cannot discover a constant sensor/encoder/network delay. Local webcams likewise lack an exposure timestamp. Under **Camera and room layout**, each camera has a **delay (ms)** field for a measured residual delay (0–200 ms). Positive values move its estimated capture time earlier; leave zero unless measured, and do not include delay already represented by the transport timestamp. Measure against a common filmed timer or flash, then test quick movement and occlusion. Do not tune delay by guessing from avatar smoothing. Reconnect WebRTC if its media clock resets.

Timing API references: [browser video-frame metadata](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback) and [PyAV media timestamps](https://pyav.org/docs/stable/api/time.html). Physical multi-camera timing and motion-to-avatar latency still require acceptance testing on the intended rig.

The [rendered-human simulation](review/simulation/REPORT.md) currently fails tracking acceptance despite passing exact-joint geometry controls. Two/front-three-camera rigs calibrate but show large joint errors; a side-on third camera can block T-pose calibration. Treat this as experimental software. If too few views remain to establish current body position, fallback points are withheld rather than presenting an old room position as reliable tracking.

Re-run **Recalibrate & align** whenever any camera is moved. A checkerboard/lens-calibration workflow would be required for measurement-grade triangulation; this automatic mode is designed for practical VR tracking setup without extra calibration props.

Browser camera access generally requires a secure context. Some mobile browsers may still refuse `getUserMedia` after merely continuing past a self-signed certificate warning. The app no longer exposes a certificate-download QR, as requested; if a specific phone refuses camera permission, that browser/device will need a trusted local certificate or a different locally trusted setup.

## Verification

The complete software gate is `python scripts/verify.py`. It requires Node.js for the phone-page checks, stops at the first failure, and runs from any working directory. The included Windows GitHub Actions workflow uses the same gate. A local passing run does not establish that hosted CI or a fresh installation has passed.

Native GUI cases run in individual subprocesses, matching the application's one-window-per-process launch model. Every case is executed once, with child failures propagated to the main suite. Repeated Tk creation in the mixed test process has intermittently failed to read Tcl resources on this PC; process isolation does not establish that underlying runtime issue is fixed.

Run all checks from the activated `.venv`:

```powershell
python -m compileall -q Main.py Lib vrfbt_calib tests review
python -m pytest -q
python -m unittest discover -s tests -v
python Main.py --check
node tests/phone_page.test.cjs
node tests/phone_webrtc_page.test.cjs
python review/bug_reproductions.py
node review/phone_failure_reproductions.cjs
python -m pip check
```

The suite covers VRChat path construction, all eight position/rotation transforms, head alignment packets, coordinate conversion, height validation, confidence filtering, timed T-pose validation, synthetic multi-view camera recovery and triangulation, full-limb secondary-camera takeover, impossible-ray rejection, the room and camera renderers, real MediaPipe startup, local TLS, direct WebRTC frame delivery, secure-WebSocket fallback, the three-phone connection cap, multi-phone identity/routing, GUI lifecycle, and clean shutdown.

Software tests cannot prove physical avatar alignment. The remaining acceptance test is to view the generated trackers in VRChat with the intended camera position, perform avatar calibration, and check neutral standing, turns, squats, feet, and occlusion recovery. See `AUDIT.md` for the evidence and remaining limitations.

See [READINESS.md](READINESS.md) for the current readiness assessment, the latest fixes, prioritized improvements, and the physical acceptance checklist. Older audit results describe earlier revisions.

The decisions for the supplied `to-fix` list, including deferred product work and acceptance evidence, are recorded in [FIX_STATUS.md](FIX_STATUS.md).
