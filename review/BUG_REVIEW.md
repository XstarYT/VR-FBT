**VR-FBT bug review — 11 September 2026**

Reviewed the current working tree, including existing uncommitted changes. **14 concrete defects were identified**, with executable probes for each.

**Resolution — 11 September 2026:** F01–F14 have been implemented and the probes in this folder now assert the corrected behavior. Final verification passed 89 pytest cases, 82 unittest-discovered cases, both browser suites, both review regression scripts, compilation, application diagnostics, and `pip check`. The original findings and proposed fixes remain below as an audit trail. Automated checks cannot establish that every possible bug has been found; real phone/browser, camera, headset, and avatar testing remains necessary.

**Priority and scope**

P1 means fix before relying on the affected tracking path; P2 means a functional defect to fix next; P3 means lower urgency. IDs remain stable between this report and the reproduction output.

| ID | Priority | Defect |
|---|---|---|
| F01 | P1 | Single-camera localization mixes converted world coordinates with camera coordinates |
| F02 | P2 | Calibration credits time when every camera has lost the person |
| F03 | P2 | Solver and sender thresholds defeat brief-occlusion handling |
| F04 | P2 | Invisible fallback joints bypass room bounds and gain artificial confidence |
| F05 | P2 | Identically named webcams overwrite each other in the source selector |
| F06 | P2 | Loading a profile can silently retain the previous camera selection |
| F07 | P2 | A silent camera blocks healthy cameras and makes frames temporally inconsistent |
| F08 | P2 | Concurrent WebRTC offers bypass device ownership and connection limits |
| F09 | P2 | Nested compatibility launcher cannot import the maintained application |
| F10 | P2 | Calibration package invents baseline scale for edges without shoulder measurements |
| F11 | P2 | WebRTC setup has no complete connection deadline |
| F12 | P2 | Changing frame rate while using WebRTC has no effect |
| F13 | P2 | An empty initial frame permanently fixes the debug floor at the wrong height |
| F14 | P3 | Failed camera calibration accumulates alignment samples without a bound |

F10 concerns the standalone `vrfbt_calib` package. The desktop engine currently uses `Lib.Tracking.MultiCameraPoseFusion`; no production import of `TposeMultiCamCalibrator` was found. Passing tests for that package therefore does not validate the desktop calibration path.

**F01 — Use one explicit coordinate convention through localization**

Locations: [Tracking.py:731](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:731), [Tracking.py:251](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:251), [Tracking.py:260](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:260).

`convert_mediapipe_landmarks` negates all three world axes. `_localize_single_camera` subsequently treats those converted coordinates as camera-axis coordinates. Manual localization directly applies `camera_to_world` to them and deliberately discards the PnP rotation. PnP can absorb part of the mismatch into a rotation, but the output path then applies a different transform.

Evidence: project a known body into a front camera, provide its camera-local world landmarks through `convert_mediapipe_landmarks`, and call the real fusion path. The result has **0.727 m mean error and -1.000 vertical correlation**: higher original landmarks become lower output landmarks. The existing corner-camera test feeds unconverted camera-local coordinates directly into `PoseResult`, so it misses this interface defect. The automatic branch also consumes the already-converted coordinates and needs the same convention audit; the numeric reproduction establishes manual-mode failure.

Proposed fix: retain camera-coordinate landmarks separately from the final Unity/room representation. Solve projection geometry in a consistent camera basis, then explicitly compose the basis conversion, camera transform, and room transform exactly once. Require positive depth in PnP validation. Do not fix this by changing a sign in only one branch.

Regression: generate normalized and world landmarks together, pass them through the actual converter, and assert upright orientation, handedness, camera position, and movement for automatic and manual single-camera paths, corner poses, and multiple views. VRChat expects Y-up, left-handed world coordinates in meters; MediaPipe world landmarks use a hip-centered origin. See the [VRChat tracker contract](https://docs.vrchat.com/docs/osc-trackers) and [MediaPipe output definition](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

**F02 — Do not count missing-pose time toward T-pose calibration**

Locations: [Tracking.py:153](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:153), [Tracking.py:123](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:123), [Tracking.py:385](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:385).

When no camera has a detected pose, `update` returns before `_observe_calibration`, bypassing its missing-camera reset. Progress still uses wall time since the original start. Once poses return, that elapsed time can finish calibration immediately.

Evidence: with a three-frame minimum, valid observations at 0.0 and 0.1 seconds, empty observations at 1, 5, and 10 seconds, and one valid observation at 10.1 seconds produce `calibrated=True`. The production twelve-frame minimum does not fix this: enough samples collected before the gap remain available afterward.

Proposed fix: process invalid/empty observations in the calibration state machine before returning. Accumulate only valid elapsed intervals, cap per-frame gaps, and reset after the chosen interruption tolerance. Make displayed progress use that accumulator too.

Regression: full dropout, partial dropout, broken T-pose, and long gaps between otherwise valid observations must not advance valid hold time.

**F03 — Make output eligibility consistent with hysteresis and holding**

Locations: [VRChat.py:276](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/VRChat.py:276), [Engine.py:324](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Engine.py:324).

An active tracker accepts confidence down to 0.30 and refreshes its last-good timestamp with that sample, but the sender rejects confidence below 0.50. Therefore confidence between 0.30 and 0.50 produces neither new packets nor the intended held packet. Conversely, an inactive tracker with confidence between 0.50 and 0.55 is returned with its original confidence and can be sent before reaching the acquisition threshold.

Evidence: after a confident neutral frame, reduce left-foot confidence to 0.40 at +100 ms. Tracker 7 receives confidence 0.40 and no OSC packet, even though the 350 ms hold has not elapsed.

Proposed fix: carry explicit tracker output eligibility/held status from the solver to the sender. Define one acquisition, continuation, loss, and hold policy. Do not refresh last-good state for samples that the output layer will reject.

Regression: exercise 0.95 → 0.40 → 0.20 over time, plus reacquisition at 0.50 and 0.55, and assert actual emitted OSC messages rather than solver values alone.

**F04 — Validate model fallback geometry and preserve uncertainty**

Locations: [Tracking.py:681](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:681), [Tracking.py:694](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:694), [Data.py:102](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Data.py:102).

Room and anatomical checks are applied to triangulated candidates. The fallback `expected` point is then accepted without equivalent checks. Its confidence is floored at 0.36 even when every camera gives the joint confidence 0.05. `Data.Map.Update` accepts coordinates at 0.35, so these unseen estimates overwrite its last reliable joint position.

Evidence: make both cameras lose a wrist and place its model estimate at `(9, 9, 9)`. Fusion outputs that point in a six-meter room with confidence **0.36**. This explains a route to exploding debug limbs and poisoned smoothing state. This probe does **not** claim the 0.36-confidence wrist itself passes the OSC sender's 0.50 gate.

Proposed fix: validate both triangulated and model-derived coordinates; preserve genuinely low visibility. When no valid estimate exists, hold the last reliable point for a bounded interval or mark it unavailable. Do not raise visibility to bypass downstream filtering.

Regression: occlude the same limb in all views, inject out-of-room model estimates, and assert bounded output, no last-good overwrite, and eventual withdrawal.

**F05 — Key camera selection by source ID, not display name**

Location: [GUI.py:326](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/GUI.py:326).

The dictionary comprehension uses `LocalCamera.display_name` as a key. Two webcams with the same driver name collide, leaving only the last. The special duplicate handling applies to phones only. A configured-camera placeholder can preserve one previously saved source, but does not make all newly discovered duplicates selectable.

Evidence: discover `local:0` and `local:1`, both named `USB Camera`, while the loaded profile references `local:2`. `local:0` disappears from the selector.

Proposed fix: retain a source-ID-based model and append the local index or another stable identifier to every ambiguous label. Apply collision handling to all camera types and saved placeholders.

Regression: select each of three identically named webcams, save, refresh, and reload without changing identities.

**F06 — Rebuild source options before applying a loaded profile**

Locations: [GUI.py:205](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/GUI.py:205), [GUI.py:350](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/GUI.py:350).

Loading a profile calls `_select_sources` using the existing camera-label dictionary. If the saved source is not present, that method keeps the old primary label and sets absent secondary sources to Off. `_configuration_from_form` then reads the retained selection; Start saves it back to the newly loaded profile.

Evidence: apply a saved offline phone and an undiscovered secondary source to an existing selector. The previous primary survives and the secondary becomes Off.

Proposed fix: rebuild options with placeholders for every source in the newly loaded profile, then select its exact ordered IDs. Keep desired source IDs separately from labels so temporary unavailability cannot silently substitute another camera.

Regression: switch between profiles with disjoint offline/local/phone source sets and verify both the visible selection and the saved configuration.

**F07 — Decouple capture and timestamp observations**

Locations: [Engine.py:164](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Engine.py:164), [Engine.py:178](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Engine.py:178), [RemoteCam.py:571](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/RemoteCam.py:571).

Capture reads are serial. Each connected phone that stops producing frames waits 300 ms. Two silent phones therefore delay a healthy camera by about 600 ms per loop, reducing the whole rig to roughly 1.6 FPS before inference. The inference pool does not remove these waits. `CameraObservation` also lacks a capture timestamp, so frames acquired at materially different times are triangulated together.

Evidence: the same `RemoteCapture` calls used by the engine take approximately **0.63 seconds** for two connected registries without frames. Disconnected phones are skipped, but suspended streams whose sockets remain connected trigger this path.

Proposed fix: use one bounded latest-frame capture worker per source, timestamp frames, and consume available fresh observations without serial timeout waits. Enforce maximum age/skew before triangulation. Reacquisition must not stall the healthy views. Native capture and inference also currently have unbounded waits, so a stalled driver can prevent orderly stopping; isolate such work if bounded shutdown is required.

Regression: retain a healthy high-rate source while two other connections stop sending; verify healthy output cadence, stale-frame rejection, and stop latency.

**F08 — Reserve phone identity and capacity before awaiting request data**

Locations: [RemoteCam.py:456](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/RemoteCam.py:456), [RemoteCam.py:458](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/RemoteCam.py:458), [RemoteCam.py:469](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/RemoteCam.py:469).

`camera_identity` checks device uniqueness and the three-camera limit before `await request.json()`. The pending request reserves neither. Another request can acquire the same ID or fill the remaining capacity before the first request installs its peer. Cleanup is keyed only by device ID, so competing connections can also disconnect or remove each other's state.

Evidence: send one WebRTC offer body in two chunks; between chunks, connect a WebSocket using the same authorized device ID. Finish the offer. Both requests succeed, with the same ID simultaneously present in `_sockets` and `_peers`.

Proposed fix: atomically reserve a device ID and connection slot before the first await, release it on every failure, and attach an ownership/generation token to registration and cleanup. Replacement or fallback must explicitly transfer ownership.

Regression: delayed-body simultaneous offers, mixed WebRTC/WebSocket attempts, duplicate IDs, four competing unique IDs, and cleanup after rejected requests.

**F09 — Set the import path in the compatibility launcher**

Location: [legacy Main.py:10](C:/Users/zmuda/Documents/GitHub/VR-FBT/VR-FBT-main/Main.py:10).

Changing the working directory does not replace the initial Python module search path. `runpy.run_path` executes the root entry point while the nested script directory remains first on `sys.path`.

Evidence: `.venv\Scripts\python.exe VR-FBT-main\Main.py --check` exits with `ModuleNotFoundError: No module named 'Lib.Config'` in this checkout.

Proposed fix: insert the resolved repository root into `sys.path` before execution, or launch the root entry point with the same interpreter in a new process and propagate its exit status.

Regression: invoke the nested launcher from both the repository root and an unrelated working directory.

**F10 — Reject or resolve edges whose metric scale is unknown**

Locations: [scale.py:47](C:/Users/zmuda/Documents/GitHub/VR-FBT/vrfbt_calib/scale.py:47), [posegraph.py:167](C:/Users/zmuda/Documents/GitHub/VR-FBT/vrfbt_calib/posegraph.py:167).

When a usable camera pair has no simultaneous valid shoulder-width measurement, `relative_translation_factor` returns 1.0. Essential-matrix translation directions have independently normalized baselines, so this silently assumes the edge has the anchor's physical baseline. An edge can still have enough nose/hip/alternating-shoulder observations for pose recovery without any paired shoulder-width sample. The top-level calibrator only requires a shoulder scale on the anchor pair.

Evidence: an AB/BC graph with a measured AB scale and no BC scale still returns a concrete C translation, assuming equal baseline lengths. There is no warning or rejection for the unresolved edge scale.

Proposed fix: use a supported alternative measurement, estimate edge scales jointly where the graph and observations make them identifiable, or reject/exclude unresolved edges with a structured error. Recheck connectivity after exclusion. Do not substitute a unit factor silently.

Regression: a connected graph whose non-anchor edge has no simultaneous shoulders must either recover scale from independent evidence or report that metric translation is underdetermined.

**F11 — Bound the complete WebRTC connection attempt**

Locations: [index.html:140](C:/Users/zmuda/Documents/GitHub/VR-FBT/Content/Website/index.html:140), [index.html:175](C:/Users/zmuda/Documents/GitHub/VR-FBT/Content/Website/index.html:175), [index.html:182](C:/Users/zmuda/Documents/GitHub/VR-FBT/Content/Website/index.html:182).

The eight-second timeout covers ICE gathering only. Signaling fetch has no application deadline, and successful signaling followed by a peer stuck in `connecting` has no fallback timer. Only `failed` and `closed` events trigger fallback. If the browser or signaling request does not reach those events promptly, the page stays on CONNECTING DIRECTLY with Start disabled.

Evidence: browser adapter completes signaling and leaves the peer `connecting`; the shipped script has **zero pending timers** and never initiates fallback on its own.

Proposed fix: set one attempt deadline covering gathering, signaling, and establishment of usable video, abort the request on expiry, and enter fallback exactly once. Clear it on success/stop and coordinate peer cleanup with F08 so a lingering server peer does not reject the fallback connection.

Regression: unresolved fetch, permanently connecting ICE, disconnected transport, and Stop during each phase.

**F12 — Apply frame-rate changes to the active WebRTC track**

Location: [index.html:251](C:/Users/zmuda/Documents/GitHub/VR-FBT/Content/Website/index.html:251).

The FPS change handler only restarts WebSocket JPEG scheduling. In direct mode, the camera track keeps the constraints from Start, although the frame-rate selector remains enabled. The Quality selector also only affects JPEG encoding, so its direct-mode scope should be made explicit.

Evidence: start direct video at 30 FPS, select the available 10 FPS option, and invoke the real change handler. No track constraint update occurs; the captured maximum remains 30 FPS.

Proposed fix: update the active track's frame-rate constraints and handle failures, or disable the control while direct streaming and clearly require restarting. Either implement a direct-video quality policy or label/disable the JPEG-only quality control.

Regression: change FPS during both transports and verify the resulting track settings or JPEG scheduler interval.

**F13 — Anchor the debug floor only from valid room geometry**

Location: [DebugView.py:252](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/DebugView.py:252).

The first render locks `_world_center` and `_floor_y` even with no landmarks. It uses -0.9 m as the fallback floor and never revisits it when the first person appears. In manual room coordinates the physical floor is 0 m, but the scene receives no explicit room-floor origin. Automatic calibration can also change the coordinate frame after the initial display anchor was chosen.

Evidence: render an empty initial frame, then a confident body. `_floor_y` remains -0.9 m. The room and camera-height display can consequently be displaced from the actual configured room.

Proposed fix: pass the coordinate-system origin/floor from fusion, use exactly zero for manual room coordinates, and defer automatic floor estimation until a reliable calibrated pose exists. Re-anchor when calibration establishes a new world frame, then keep it fixed during tracking.

Regression: no-person startup followed by detection, manual floor at zero, and completion/restart of automatic calibration.

**F14 — Bound partial alignment samples during failed calibration**

Location: [Tracking.py:405](C:/Users/zmuda/Documents/GitHub/VR-FBT/Lib/Tracking.py:405).

Alignment samples are appended for each successfully localized camera before the check that every camera solved. The bound later in the function applies only to `_pose_samples`. If a visible camera continually fails PnP, the other cameras' alignment lists keep growing for as long as calibration is attempted.

Evidence: keep valid T-poses present, allow one camera's PnP solve and reject the other's, and process 650 frames. The successful camera retains all 650 alignment samples, beyond the timed session's 600-frame pose-sample limit, while calibration never finishes.

Proposed fix: bound alignment history too, preferably collecting only synchronized successful sample sets. Use a deque/window and discard both pose and alignment data together on reset.

Regression: sustained single-camera PnP failure must keep every calibration buffer bounded and recover using a recent consistent sample window.

**Original verification and remaining uncertainty**

- Compilation of the maintained Python sources, calibration package, and tests passed.
- The documented `unittest discover -s tests -v` command passed **72 tests**.
- Both existing JavaScript phone-page suites passed.
- `Main.py --check` and `pip check` passed in the repository's Python 3.12 environment.
- Full pytest discovery ran **78 tests: 77 passed, 1 failed**. The failure was `test_direct_webrtc_video_reaches_remote_capture`, which received no frame within its deadline. The same test passed under unittest and passed when rerun alone under pytest. Its root cause is unresolved; do not classify it as a deterministic transport defect based on this run.
- Pytest also reported seven warnings involving Tk variable destruction away from the Tk thread. Investigate test/application object lifetime and garbage collection before dismissing the intermittent frame timeout; no causal relationship was established here.
- Five additional two-camera synthetic seeds (2105–2109) passed a limited check: maximum rotation error 0.697 degrees and maximum relative translation-vector error about 0.008. This is additional synthetic evidence, not proof of physical calibration accuracy.
- The README's unittest-only verification command does not run the six pytest-style calibration cases. Add pytest and an explicit developer test dependency to the documented/CI verification path.
- The older audit's test totals and earlier resolved-issue claims are historical. They do not supersede the current reproductions.

The review covered the maintained launch/configuration, GUI, capture/phone transport, browser, inference/conversion, fusion, VRChat/OSC, debug rendering, standalone calibration modules, tests, scripts, and legacy helpers. Model bundle hashes were checked by diagnostics; opaque model internals and physical avatar behavior were not audited. The unused `AsyncCam` helper also warrants lifecycle hardening if restored to the engine: its background `grab` and foreground `retrieve` do not share a lock, and restart reuses an already-started thread. These are dormant-code concerns, not additional demonstrated desktop failures in the count above.

**Run the fixed-behavior regressions**

The probes now assert the desired behavior for every finding. They use synthetic poses, widget adapters, and a temporary localhost camera hub. They do not edit saved profiles, open physical cameras, or send to the configured VRChat destination.

```powershell
.\.venv\Scripts\python.exe review\bug_reproductions.py
node review/phone_failure_reproductions.cjs
```

Artifacts: [Python reproductions](C:/Users/zmuda/Documents/GitHub/VR-FBT/review/bug_reproductions.py), [browser reproductions](C:/Users/zmuda/Documents/GitHub/VR-FBT/review/phone_failure_reproductions.cjs).

The implementation followed the suggested order and each probe has been converted to an assertion of the desired behavior.
