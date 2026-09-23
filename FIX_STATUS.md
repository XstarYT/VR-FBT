# Review of `to-fix` (23 September 2026)

This file records decisions for the supplied change list. Passing software tests do not establish physical VRChat accuracy.

| Item | Decision and evidence |
| --- | --- |
| P0.1–P0.3 | Implemented. Default uses automatic geometry and a neutral local source; GUI explains the modes and checks offline saved phones before Start. Reduced-source override is session-only. GUI and configuration tests cover these paths. |
| P0.4 | Implemented estimated-FOV guidance, native phone frame size, and read-only focal length. The proposed shoulder-span estimator was omitted: shoulder pixels and an assumed shoulder width do not determine FOV without independent distance or focal data. Existing lens-geometry tests cover quarter-turn width/height handling. |
| P0.5 | Implemented high-reprojection lens hint and per-camera error in Activity. This is a diagnostic threshold, not a proof that lens calibration will correct every error. |
| P0.6 | Implemented mode-specific setup guidance and fixed-room factory-position warning. |
| P0.7 | Implemented a 0.5 m baseline and 12° subject-ray parallax guard for automatic calibration. A 30° camera-forward-vector cutoff would reject valid parallel-facing stereo rigs; synthetic tests cover both cases. Actual optical placement still needs physical validation. |
| P0.8 | Experimental-accuracy banner and explicit readiness waiver added. Rendered-human acceptance remains below 90%; it would be misleading to make the normal software gate green by weakening the acceptance threshold. |
| P1.1 | Implemented a seven-step first-run assistant for diagnostics, OSC pulse, phone/local camera connection, camera selection, height, tracker set, and summary. Completion is persisted only after a valid profile save. GUI tests cover the stable automatic profile and completion flag. |
| P1.2 | Implemented a capture-only in-window preview for selected local or phone sources, with frame-received and resolution overlays. It closes before tracking opens the cameras. Capture and GUI lifecycle tests cover release and ordering; physical device behavior remains untested. |
| P1.3 | Per-phone disconnect/reconnect controls remain open. Direct WebRTC and compatibility WebSocket have different reconnect lifecycles; closing a peer without coordinated browser state would immediately reconnect or leave a misleading UI. This needs a protocol-level design and phone tests. |
| P1.4 | Browser camera rejection codes are sent to a bounded, token-authenticated hub endpoint and mapped to desktop recovery messages. A browser that cannot load the page cannot report an error; `NotAllowedError` alone cannot distinguish permission denial from some certificate policies. |
| P1.5 | Per-source visible-joint count, accepted reprojection error, and T-pose visibility are exposed through calibration diagnostics and failure hints. |
| P1.6–P1.7 | Camera T-pose and VRChat body calibration use distinct status text. A Measure delay checklist was added; latency is never auto-written. Existing OSC gating remains in place. |
| P2.1 | The desktop calibration path is identified in README; `vrfbt_calib` is labelled experimental. No risky solver replacement was made without accuracy evidence. |
| P2.2–P2.4 | Large package split, legacy-tree removal, and installer data migration are deferred. The legacy launcher is still used by regression checks, and there is no installer in this repository to validate the Program Files migration. |
| P2.5 | Added a sustained dual-camera inference p95 warning with a Lite / lower phone-FPS action. |

Next evidence gate: repeat the rendered-human fixture with the changed calibration guard, then test one and two real phones with a headset and avatar using the physical acceptance sequence in `READINESS.md`. Keep the experimental label until coverage and alignment meet the documented targets.
