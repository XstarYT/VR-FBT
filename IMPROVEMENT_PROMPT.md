# Implementation prompt: turn VR-FBT into a release-quality VRChat tracker

You are the senior engineer responsible for taking this Windows VRChat full-body-tracking repository from an experimental prototype to a reliable, secure, measurable desktop product. Work from the repository root, treat `AUDIT.md` as the verified starting inventory, and inspect the current code before changing it. VRChat is the only supported output target. External cameras must connect only over the local network: do not add ADB, phone developer mode, public tunnels, Cloudflare, a cloud relay, or an internet media path. Do not claim hardware behavior that you have not demonstrated. Preserve user configuration and model artifacts unless a migration is explicitly implemented and tested.

## Product goal

Deliver a polished Windows application that captures a local webcam or same-LAN phone camera, estimates body pose, transforms landmarks into correctly calibrated VRChat tracker poses, filters noise without unacceptable latency, and sends positions and rotations over VRChat's documented OSC tracker contract. A nontechnical user must be able to install it, connect a phone with one local link, complete first-run setup, calibrate in VRChat, start/stop tracking, understand system health, recover from common errors, and export useful diagnostics without opening a terminal.

## Required working method

1. Restore or initialize proper Git history and create small, reviewable commits. Preserve unrelated user work. Make the repository root the only canonical source tree; keep the nested compatibility launcher for one migration release, then remove the duplicated assets in a separately reviewed commit.
2. Reproduce the supported environment in clean 64-bit Python 3.12 on Windows. Lock compatible MediaPipe, OpenCV, python-osc, aiohttp, aiortc, cryptography and packaging versions. Verify every official Pose Landmarker bundle hash and document supported Windows/CPU combinations.
3. Treat VRChat's official OSC tracker documentation as the receiver contract. Verify `/tracking/trackers/1..8/{position,rotation}` and `/tracking/trackers/head/{position,rotation}`, three-float payloads, meters, Euler degrees, Unity left-handed axes, the Z-X-Y application order, stable sender numbering, VRChat's calibration-based role assignment, one-shot head alignment behavior, rate/dropout behavior and current calibration flow. Do not invent fixed ID-to-body-role semantics that the public documentation does not specify. Cite the official documentation in architecture notes and pin the date checked.
4. Add failing tests for each defect or requirement, implement the smallest coherent fix, then run the whole relevant suite. Keep camera, inference, transformation, filtering, transport, persistence, and UI separated behind typed interfaces so each can be tested independently.
5. Maintain a validation ledger containing the exact command, environment, fixture/hardware, result, and artifact for every claim. Clearly label simulated, bench-tested, and user-acceptance-tested behavior.

## Architecture to implement

- Create an installable `vr_fbt` package with modules for configuration/schema migration, device discovery/capture, inference backends, landmark-to-body solving, calibration, filtering, tracker pose generation, OSC transport, session orchestration, diagnostics/telemetry, and UI.
- Define typed domain objects for timestamped camera frames, landmarks with confidence, calibrated skeletons, tracker poses with position/quaternion/confidence, backend health, and tracking state. Use monotonic timestamps throughout the real-time pipeline.
- Use bounded queues and an explicit backpressure policy. Prefer dropping stale camera frames over building latency. Never touch Tk widgets outside the Tk thread. All workers must support cancellation and deterministic join/cleanup.
- Make inference backend selection explicit: DirectML, CUDA when applicable, then CPU. Show the selected device and model-load result. Validate checkpoint integrity and state-dict shapes before capture begins.
- Replace raw normalized landmark transmission with a documented transform pipeline: camera mirroring/orientation, image-to-body axes, depth/scale estimation, neutral-pose calibration, floor and height calibration, receiver origin/handedness conversion, per-tracker offsets, and final meters/quaternion output.
- Implement rotations from stable anatomical frames. Specify quaternion ordering and handedness, handle collinear/missing joints, maintain temporal continuity, prevent quaternion sign flips, and test known poses numerically.
- Implement confidence-aware filtering. Evaluate at least EMA, One Euro, and a Kalman-family option using recorded sequences. Make parameters time-step-aware. Add visibility hysteresis, stale timeouts, loss/reacquisition transitions, outlier rejection, and a receiver reset/disable message.
- Encapsulate VRChat OSC behind a transport interface. Avoid private library internals. Default to `127.0.0.1:9000`, validate destination resolution, expose send counters and last error, support a real UDP loopback test, and record packets in integration tests. Do not expose the OSC destination to LAN by default.
- Extend the maintained token/TLS `aiohttp` phone hub with explicit Private-network firewall onboarding, same-LAN checks, origin policy, global connection/rate limits, certificate rotation, mobile acceptance tests, and multi-view timestamp synchronization. Keep camera frames local and display a prominent privacy warning. Maintain one camera QR only; do not expose a CA-download QR or silently install trust. Clearly detect and explain when a mobile browser refuses camera APIs after a self-signed certificate warning.

## GUI and user experience

Keep the existing dark control-center direction but evolve it into these flows:

- First-run wizard: environment/model check, camera selection and live preview, VRChat OSC enabled/test check, height and alignment calibration, and a final readiness summary.
- Main dashboard: large start/stop action, explicit state machine, in-GUI camera/skeleton preview, backend/device, measured capture/inference/send FPS, end-to-end latency, confidence, dropped frames, OSC health, and actionable alerts.
- Devices: enumerate cameras by friendly name; preview; choose resolution/FPS/mirror/rotation/exposure; test reconnect and exclusive-use errors.
- Calibration: guided neutral pose, user height, floor/origin, camera-facing direction, VRChat head-space alignment, per-tracker offsets, quality score, redo/reset, and saved calibration version.
- Trackers: enable/disable each tracker, inspect source joints, position and rotation values, confidence, loss state, smoothing preset, and receiver ID/path.
- Profiles: create, duplicate, rename, delete with confirmation, import/export, schema-version migration, dirty-state indication, and validation before activation.
- Diagnostics: dependency/backend/model/asset/camera/receiver checks; filterable structured logs; privacy-safe support bundle; copy/export; no secrets or raw frames unless separately consented.
- Settings: start behavior, updates, logging, privacy/network exposure, theme/high contrast, and advanced performance/filter controls.

The UI must be keyboard accessible, high-DPI safe at 100–200%, usable at the minimum supported window size, responsive while inference runs, and clear about the distinction between Ready, Starting, Calibrating, Tracking, Degraded, Reconnecting, Stopping, Stopped, and Error. Buttons must not permit invalid transitions. Errors should state what failed, likely causes, and a concrete recovery action.

## Tests and evidence required

- Unit tests for schemas/migrations, transform math, rotations, filters, confidence hysteresis, tracker mapping, OSC encoding, state transitions, queue/drop policy, and error cleanup.
- Property/numerical tests for finite outputs, normalized quaternions, quaternion continuity, coordinate round trips, invalid input, missing joints, variable frame intervals, and extreme values.
- Recorded-frame inference tests with licensed fixtures and golden tolerances; corrupted/wrong checkpoint tests; DirectML-versus-CPU parity tests.
- Mock-camera and UDP-capture integration tests covering startup, steady streaming, stop, camera disconnect/reconnect, receiver errors, tracking loss/reacquisition, and application close during model load.
- Real hardware matrix covering at least two cameras, CPU fallback, representative integrated/discrete GPUs, 30/60 FPS targets, mirrored/rotated inputs, and a 60-minute soak run.
- VRChat acceptance test proving all eight tracker roles appear, and that positions and rotations match neutral standing, turns, squats, knee lifts, foot direction and elbow movement with correct scale, floor, forward direction, alignment, avatar calibration and dropout/recovery behavior.
- UI tests for save/load/migration, invalid values, profile flows, start/stop races, diagnostics, DPI/scaling, keyboard focus, and screen-reader labels where supported.
- Security tests for malformed configuration, hostile OSC strings, oversized/fragmented WebSocket frames if remote camera remains, unauthorized connections, origin rules, path handling, and support-bundle redaction.
- Performance budgets stated before optimization: model-load time, capture-to-send latency percentiles, sustained FPS, frame-drop rate, memory, CPU/GPU use, and shutdown time. Fail CI or a dedicated benchmark gate when agreed limits regress materially.

## Packaging, security, and documentation

Publish source and a reproducible build recipe for every bundled executable, or remove opaque binaries. Generate an SBOM, scan dependencies and packaged artifacts, sign Windows releases, hash model assets, and document checkpoint/data licenses and provenance. Store mutable data under a per-user application-data directory, never beside a protected install. Add CI on Windows/Python 3.12, lint/type/format gates, unit/integration suites, artifact-integrity checks, and packaging smoke tests.

Write a useful README, setup/troubleshooting guide, architecture and coordinate-system specification, calibration guide, privacy/network threat model, model card, performance report, changelog, license, security policy, and contribution guide. Replace stale/profane comments and normalize public naming while providing migrations or compatibility aliases where needed.

## Definition of done

Do not call the project complete merely because the GUI opens or packets are emitted. It is done only when a fresh Windows user can install a signed build, pass readiness checks, select a camera, enable VRChat OSC, complete alignment/avatar calibration, stream correct position and rotation poses for all eight roles, survive tracking/camera/VRChat interruptions, stop without leaked resources, restart successfully, and export a useful redacted diagnostic bundle. A phone user must be able to connect on the same LAN without developer mode or any cloud relay, with unsupported certificate-warning behavior detected honestly. All automated suites must pass, the real camera/phone/headset/avatar matrix and soak test must have recorded results, no P0/P1 audit item may remain open without an explicit product decision, and documentation must match the shipped behavior exactly.

At the end, provide: a concise change summary; architecture diagram; migration notes; exact verification commands and results; hardware/receiver evidence; performance table with p50/p95/p99 latency; security and licensing status; known limitations; and the prioritized next-release backlog. If any required proof is unavailable, state the missing evidence plainly and do not substitute assumptions.
