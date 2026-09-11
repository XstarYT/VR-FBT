# VR-FBT for VRChat

Windows camera-based full-body tracking for VRChat. VR-FBT runs Google's MediaPipe Pose Landmarker locally, turns its metric body landmarks into eight OSC trackers for the body areas supported by VRChat, and sends position and rotation messages directly to VRChat on the same PC.

> This is an experimental optical tracker. Automated tests verify the software pipeline, but final body alignment still needs to be checked in VRChat with a real person, camera, headset, and avatar.

## Requirements

- Windows 10 or 11
- 64-bit Python 3.12
- VRChat for Windows with OSC enabled
- One to three local webcams/phones; phones and PC must share the same local network
- OpenSSL (Git for Windows' bundled OpenSSL is detected automatically)

## Install and run

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Main.py --check
python Main.py
```

After setup, `run.bat` starts the repository-local environment without manual activation.

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
- **Heavy — highest accuracy:** greater detail, but much slower.

The official Lite, Full, and Heavy bundles are stored under `Model/mediapipe`, and diagnostics verify their SHA-256 hashes. On this PC, repeated-image model-call benchmarks measured about 122 FPS Lite, 83 FPS Full, and 24 FPS Heavy; results on other PCs will differ. MediaPipe's [Pose Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) explains the image/world landmarks and temporal video mode.

## Use a phone camera on the local network

There is one phone connection method: direct local-network HTTPS. There is no Android developer mode, USB pairing, ADB, Cloudflare, public tunnel, or internet video relay.

1. Connect the phone and PC to the same private Wi-Fi/LAN. Guest Wi-Fi or client isolation must be off.
2. Click **Connect phone on LAN**. OpenSSL automatically generates or refreshes the local HTTPS certificate.
3. Scan the single QR code. Continue past the browser's local certificate warning if offered.
4. Press **Start camera**, allow camera access, then select the phone by its camera name under **Camera Source**.
5. If Windows Firewall asks, allow Python/VR-FBT on **Private networks** only.

The link includes a random session token and changes after restart. Keep it within your local network. WebRTC is attempted first with no STUN or TURN relay; JPEG-over-secure-WebSocket is the fallback. The receiver keeps only the newest frame so old frames do not build up into extra tracking latency. Up to three phones can connect and all three can be used by one tracking session.

## Multi-camera tracking and calibration

Select two or three different cameras in the setup screen, then start tracking and hold a T-pose for 10 seconds with your full body visible in every view. The timer advances only while all selected cameras see the shoulders, elbows, wrists, hips, knees, and ankles in an extended-arm pose. During this phase the camera-local wireframes are compared, fixed camera transforms are estimated, and the user's neutral room location is locked. The normal 20-frame VRChat body calibration follows automatically. Place cameras at visibly different angles—front plus side is much better than putting them next to each other.

The calibration assumes a typical 60-degree horizontal phone-camera field of view because browsers do not expose reliable lens calibration. It uses the detected T-pose skeleton as a temporary calibration target, rejects high-reprojection-error solutions, removes statistical outliers, and computes fixed camera transforms from error-weighted rotation/translation averages. A joint is triangulated when at least two views provide it. If a camera loses an entire arm or leg, the highest-confidence remaining camera takes over every joint in that limb using its calibrated wireframe transform. The existing short-occlusion hold is used only when no camera has a trustworthy observation. Camera frustums remain fixed in room coordinates while the localized skeleton moves through the room.

For the simplest physical setup, click **Camera layout…**, enter the room width/height/depth, choose one of the four corners for each camera, and enter its height in the **Height** field. Applying a corner automatically enables fixed room anchoring, puts the camera on the corresponding room corner, and aims it toward the room center. Choose **Custom coordinates** to edit X/Z/yaw/pitch/roll yourself. The same dialog has an **Image rotation** correction for a sideways or upside-down phone; choose 0°, 90°, 180°, or 270° clockwise. Rotation is applied before pose detection as well as preview rendering. Room coordinates use the center as X=0/Z=0 and the floor as Y=0.

Triangulated points are accepted only when their rays agree, remain inside the configured room, and preserve a plausible relationship to the detected torso and limb geometry. An impossible intersection is replaced with the aligned body-model point, preventing the long exploding limb lines that a low reprojection error alone can otherwise produce.

Re-run **Recalibrate & align** whenever any camera is moved. A checkerboard/lens-calibration workflow would be required for measurement-grade triangulation; this automatic mode is designed for practical VR tracking setup without extra calibration props.

Browser camera access generally requires a secure context. Some mobile browsers may still refuse `getUserMedia` after merely continuing past a self-signed certificate warning. The app no longer exposes a certificate-download QR, as requested; if a specific phone refuses camera permission, that browser/device will need a trusted local certificate or a different locally trusted setup.

## Verification

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
