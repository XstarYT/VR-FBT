# VR-FBT repository audit

## Latest tracking update

The active pose backend has been replaced with MediaPipe Pose Landmarker 1.0.1 using the official Lite, Full, and Heavy detector-plus-landmarker bundles. Full is the default. The broken full-frame-to-landmark-only inference path is no longer used by the engine. MediaPipe video mode supplies person detection, aligned ROI tracking, normalized image landmarks for debug drawing, and world landmarks in meters for OSC positions. The 33-point output is explicitly converted to the repository's 31-point map, using foot-index landmarks instead of accidentally treating heels as feet.

Tracking-loss frames no longer update the mapped pose or send OSC. Capture requests a one-frame buffer, and the GUI exposes Lite/Full/Heavy quality selection. The VRChat solver converts camera-facing MediaPipe world coordinates to Unity left-handed, Y-up meters; derives anatomical rotations; scales the body using user height; and now calibrates scale and forward yaw from 20 stable, fully visible frames. Delta-time-aware One Euro filters, visibility hysteresis, and a 350 ms occlusion hold reduce jitter and tracker collapse. A filtered head position is sent continuously to keep the hip-relative camera skeleton attached to the live HMD, while head rotation remains a one-shot yaw alignment. **Recalibrate & align** resets the calibration when the camera or stance changes. The new default **Stable** tracker set sends hip plus feet because VRChat documents that fewer trackers can produce better IK when absolute tracker accuracy is limited; the all-eight set remains an experimental option. Benchmarks on this host measured about 122/83/24 FPS respectively at the MediaPipe call boundary on a repeated real image; the complete Full wrapper averaged 22 ms. The physical Webcam C110 opened at 640×480 and completed a Full inference in 15 ms, correctly returning no pose because no person was in view. Real headset/avatar acceptance testing is still required.

Verification: 56 Python tests passed, including stable multi-frame calibration, forward-yaw correction, adaptive filtering, visibility hysteresis, the recommended hip-and-feet output set, the real MediaPipe runtime, 33-to-31 landmark/world-coordinate conversion, model selection and height validation, VRChat pose/rotation math, all official OSC paths, forced IPv4 localhost handling, real UDP delivery, phone/WebRTC/TLS integration, single-instance protection and GUI lifecycle. Compilation, dependency checks, model presence, both phone-page JavaScript suites, and OpenSSL diagnostics passed. Model SHA-256 verification is included in headless diagnostics.

## Phone-connection update

The desktop now exposes exactly one external-camera path: direct HTTPS over the local network. Android debugging/ADB pairing and the Cloudflare internet tunnel were removed from the UI, source, diagnostics, tests and install scripts. Their bundled binaries were removed from the repository. The phone dialog contains a single camera QR and no trust-certificate QR or download endpoint.

The local path discovers OpenSSL automatically, generates a private CA plus LAN-IP server certificate, and refreshes the server certificate when local addresses change. The CA remains local app data and is used only to sign the server certificate; it is not served to phones. Direct WebRTC has no STUN/TURN service and the WebSocket fallback also remains on LAN. The random link token, eight-camera cap, latest-frame delivery and clean shutdown remain enforced.

Verification covers real WebRTC encode/decode into the OpenCV-compatible capture, TLS certificate generation/reuse, strict HTTPS on an active LAN address, absence of the removed certificate endpoint, multi-phone routing, authorization and shutdown. Two phone-page JavaScript tests cover permission cancellation, WSS fallback, relay-free WebRTC configuration, signaling and cleanup. Physical phone permission and browser treatment of a self-signed HTTPS warning still require target hardware; some browsers may require the certificate to be trusted at OS level before they expose camera APIs.

Audit date: 2026-09-09  
Canonical application directory: repository root (the nested `VR-FBT-main` directory is a legacy duplicate)

## Scope and verification performed

Every repository path was enumerated. All readable Python, HTML, JSON, TOML, batch, Markdown, and text files were inspected. The earlier unused PyTorch checkpoints/anchors were hashed and inspected before being removed from the active repository. The opaque `LocalServer.exe`, obsolete PyTorch modules, and duplicated nested app assets were removed after the MediaPipe/VRChat pipeline replaced them. The small nested compatibility launcher remains so old shortcuts redirect to the canonical root.

Initial audit checks plus the current implementation checks completed:

- all maintained Python source files compile successfully;
- the maintained root source compiled successfully with `compileall`;
- 56 unit/integration tests pass, including VRChat solver/OSC/UDP coverage, stable calibration and tracker-set selection, IPv4 loopback regression, real two-client WebSocket exchange, JPEG decode, direct WebRTC, generated certificate verification and local HTTPS;
- the Tk GUI was constructed, updated, and destroyed successfully at 1120×720;
- JSON and TOML configuration loaded and cross-references validated;
- all four anchor arrays loaded successfully and contained only finite values;
- the 31 landmark indices are contiguous and the eight position/rotation tracker IDs align;
- the repository-local Python 3.12 environment passes dependency, configuration, OpenSSL and model-integrity diagnostics.

The configured local `Webcam C110` was opened through DirectShow/OpenCV and returned one 640×480 BGR frame. Real MediaPipe CPU inference and actual localhost UDP OSC delivery are tested. Physical-phone browser onboarding and visual VRChat avatar interpretation are not claimed because no phone/headset/avatar fixture was available to the automated run.

## Resolved in this pass

1. Replaced the import-time CLI loop with a real Tk desktop control center. It provides profile/config editing, start/stop controls, atomic saves, validation, runtime state, FPS/confidence/frame metrics, logs, and diagnostics.
2. Added a background tracking controller with deterministic stop behavior, resource cleanup, camera-index correctness, drift-resistant frame pacing, and surfaced worker errors.
3. Removed unsafe `eval` from OSC parsing, added a safe literal parser, direct message construction, and an explicit socket close API.
4. Made expensive tracking imports lazy so the GUI and diagnostics can run even when ML packages are missing.
5. Added DirectML-to-CUDA/CPU fallback and safe CPU checkpoint loading.
6. Corrected OpenCV BGR input to RGB before model normalization.
7. Implemented profile smoothing, bounded independent history snapshots, fused-joint creation, landmark-count validation, and return values for both filters.
8. Corrected the broken timing context manager and the previous frame-rate accounting error.
9. Reworked asynchronous camera configuration so state is per instance, camera-open/read failures surface, and captures are released.
10. Corrected package-relative model imports that made four model modules fail under normal package imports.
11. Fixed remote web-camera page dimension typos, media errors, null blobs, and sends attempted before the WebSocket was open.
12. Added a standard `requirements.txt`, preserved the legacy `req.txt` entry point, documented Python 3.12 setup, and improved the restart script.
13. Added standard-library unit coverage for configuration, validation, smoothing, filters, fused joints, safe OSC parsing, and tracker message construction.
14. Converted the nested app entry point into a compatibility redirect so old shortcuts launch the maintained root GUI.
15. Replaced raw camera-index entry with DirectShow friendly-name discovery; the tested machine enumerated nine named local/virtual cameras.
16. Replaced the handwritten remote-camera transport with a token-protected `aiohttp` multi-phone hub, bounded WebSockets, heartbeat, independent device identities, latest-frame routing, and clean lifecycle behavior.
17. Added a mobile camera page with browser camera-name selection, device naming, lens switching, FPS/quality controls, backpressure, maximum resolution, clear permission errors, and stable device identity.
18. Added a persistent private local CA and LAN-address-aware server certificates. OpenSSL refreshes the server certificate when addresses change. The former public CA download endpoint and QR were later removed for the requested single-link LAN flow.
19. Fixed live VRChat delivery on Windows: `localhost` resolved to IPv6 `::1` first while VRChat listened on IPv4 `0.0.0.0:9000`. The OSC client now resolves and forces IPv4, the default profile uses `127.0.0.1`, and a regression test verifies the socket family.
20. Added a harmless **Send VRChat OSC test** pulse and resolved-target log so transport can be checked before a pose is acquired. The frames-sent metric now counts only confident pose frames rather than every captured camera frame.
21. Added a Windows named-mutex guard after finding two simultaneously running VR-FBT instances during live diagnosis. New launches now explain that the existing window must be closed instead of creating competing camera/server sessions.

## Remaining work, in priority order

### P0 — required before a trustworthy release

- **Complete physical VRChat acceptance.** The implementation follows VRChat's documented OSC paths, three-float payloads, meters, Euler degrees and Unity coordinate space, and unit tests capture every generated packet. It keeps its own numeric tracker ordering stable while VRChat assigns roles during FBT calibration. Validate the camera-facing axis choice, scale, rotations, continuous head-position and one-shot yaw alignment, floor placement, avatar calibration, dropout and recovery inside current VRChat with a real person/headset/avatar before claiming physically correct placement.
- **Establish version control and recovery.** The supplied directory has no usable `.git` repository, so changes cannot be diffed, bisected, or restored normally. Initialize or restore the real upstream Git history before further release work.
- **Retire the compatibility launcher.** Duplicated nested assets are removed. After old shortcuts have migrated, remove the remaining small redirect folder in a reviewed release.

### P1 — functional and security gaps

- **Complete receiver loss behavior.** Whole-pose detection gates OSC output, per-joint visibility uses acquire/release hysteresis with a 350 ms hold, and loss/reacquisition is reported. A receiver-specific disable/reset message still needs a defined VRChat contract.
- **Extend calibration.** Height scaling and VRChat head-space realignment are implemented. Add guided floor/forward/mirror checks, per-tracker correction, calibration quality, reset/export/import and avatar-specific presets.
- **Decide multi-camera scope.** The old `MULTI` branch was empty. The new validator deliberately blocks it. Either implement synchronization, calibration, fusion, and camera health or remove the mode from documentation/configuration entirely.
- **Tune filtering with recorded motion.** One Euro filtering is now delta-time-aware and covered by synthetic jitter tests. Record representative neutral, fast-motion, discontinuity and low-confidence sessions, then tune per-joint parameters and compare against a Kalman-family option.
- **Complete simultaneous multi-camera fusion.** The hub accepts and independently routes multiple phones, but a tracking session deliberately consumes one selected source. True multi-view tracking still needs timestamp synchronization, intrinsic/extrinsic calibration, occlusion handling, pose fusion, performance limits, and a failure policy.
- **Validate browser certificate-warning behavior.** The requested single-QR flow assumes the phone browser permits camera APIs after the user continues past the local self-signed certificate warning. Test supported iOS/Android browser versions. If a browser still treats it as an insecure context, provide an explicit opt-in trust workflow or a locally trusted alternative without restoring a cloud relay.
- **Document active model provenance and licensing.** Record the official MediaPipe task-bundle source URLs, version, license, hashes, accuracy metrics and redistribution terms in a model card.
- **Maintain model/backend compatibility evidence.** MediaPipe and OpenCV are pinned and all three official bundles are hash-checked. Add a supported CPU/Windows matrix, recorded-frame golden tolerances and regression benchmarks when CI is available.
- **Improve camera UX.** Enumerate cameras with friendly names, offer preview/test/resolution/exposure/mirroring controls, prevent simultaneous use, and explain permission/device failures.
- **Harden OSC configuration.** Resolve hostnames before starting, provide a receiver test packet, validate local/network exposure, support reconnection/error reporting, and avoid reaching into a library’s private `_sock` in the final transport abstraction.

### P2 — maintainability, UX, and delivery

- Delete or modernize unused legacy modules (`CLIKit`, `Json`, `Toml`, much of `AsyncCam`) after confirming no external consumers. They retain Windows-only imports and inconsistent naming/tab formatting.
- Split model code from application code into an installable package; add `pyproject.toml`, type checking, formatting/linting, and pre-commit hooks.
- Add CI for Python 3.12 on Windows: compile, unit tests, config-schema tests, checkpoint integrity checks, and a mocked camera/OSC integration test.
- Add structured rotating logs with a privacy-safe support bundle and GUI copy/export controls.
- Add profile creation/rename/delete/duplicate flows, unsaved-change prompts, inline validation, tooltips, keyboard navigation, high-DPI testing, and accessibility contrast/focus checks.
- Replace the external OpenCV preview window with an in-GUI preview and skeleton overlay, while keeping inference off the Tk main thread.
- Add a first-run setup wizard and clear device/backend/receiver readiness indicators.
- Package with a reproducible Windows installer, avoid writing beside a read-only installed executable, and move mutable profiles/logs to an appropriate user-data directory.
- Add release metadata: license, changelog, contribution guide, code of conduct, security policy, support boundaries, screenshots, and architecture documentation.
- Add performance measurement for cold start, model load, capture latency, inference latency, send latency, frame drops, CPU/GPU/memory usage, and long-running leaks.
- Remove profanity and stale comments, normalize names (`Phrase`, `retrieve`, snake_case), and add docstrings/types to public interfaces.

## Double-check conclusion

The GUI/configuration/data/VRChat-OSC construction paths are internally consistent under the available tests, the pinned MediaPipe runtime has completed real-image detection and a physical camera smoke test, and the model bundles pass integrity checks. Positions, rotations, all eight tracker roles, height scaling and head alignment are implemented against VRChat's published contract. It is still **not valid to call physical avatar tracking proven** until a person-in-view headset/avatar acceptance test confirms orientation, scale, floor placement, calibration and recovery. The detailed longer-term execution brief is in `IMPROVEMENT_PROMPT.md`.
