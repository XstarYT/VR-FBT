# VR-FBT for VRChat

Windows camera-based full-body tracking for VRChat. VR-FBT runs Google's MediaPipe Pose Landmarker locally, turns its metric body landmarks into eight OSC trackers for the body areas supported by VRChat, and sends position and rotation messages directly to VRChat on the same PC.

> This is an experimental optical tracker. Automated tests verify the software pipeline, but final body alignment still needs to be checked in VRChat with a real person, camera, headset, and avatar.

## Requirements

- Windows 10 or 11
- 64-bit Python 3.12
- VRChat for Windows with OSC enabled
- A local webcam, or a phone and PC on the same local network
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
2. In VR-FBT select the camera, set your real height in meters, and leave the VRChat OSC destination at `127.0.0.1:9000` when VRChat is on this PC. `localhost` is also accepted and is explicitly resolved to IPv4 because VRChat's Windows listener uses IPv4.
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

Keep **Tracking debug overlay** enabled while positioning the camera. It shows the skeleton, confidence, visible points, processing FPS, camera source, and whether tracking is currently usable. Press Q in that window or click **Stop** in VR-FBT.

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

The link includes a random session token and changes after restart. Keep it within your local network. WebRTC is attempted first with no STUN or TURN relay; JPEG-over-secure-WebSocket is the fallback. The receiver keeps only the newest frame so old frames do not build up into extra tracking latency. Up to eight phones can connect and remain individually selectable, though one tracking session currently uses one selected camera.

Browser camera access generally requires a secure context. Some mobile browsers may still refuse `getUserMedia` after merely continuing past a self-signed certificate warning. The app no longer exposes a certificate-download QR, as requested; if a specific phone refuses camera permission, that browser/device will need a trusted local certificate or a different locally trusted setup.

## Verification

Run all checks from the activated `.venv`:

```powershell
python -m compileall -q Main.py Lib tests
python -m unittest discover -s tests -v
python Main.py --check
node tests/phone_page.test.cjs
node tests/phone_webrtc_page.test.cjs
python -m pip check
```

The suite covers VRChat path construction, all eight position/rotation transforms, head alignment packets, coordinate conversion, height validation, confidence filtering, real MediaPipe startup, local TLS, direct WebRTC frame delivery, secure-WebSocket fallback, multi-phone identity/routing, GUI lifecycle, and clean shutdown.

Software tests cannot prove physical avatar alignment. The remaining acceptance test is to view the generated trackers in VRChat with the intended camera position, perform avatar calibration, and check neutral standing, turns, squats, feet, and occlusion recovery. See `AUDIT.md` for the evidence and remaining limitations.
