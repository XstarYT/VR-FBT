# MediaPipe Pose Landmarker model bundles

Downloaded from the official Google MediaPipe model storage documented at:
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models

| File | Mode | SHA-256 |
| --- | --- | --- |
| `pose_landmarker_lite.task` | fastest | `59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A` |
| `pose_landmarker_full.task` | balanced | `4EAA5EB7A98365221087693FCC286334CF0858E2EB6E15B506AA4A7ECDCEC4AD` |
| `pose_landmarker_heavy.task` | highest accuracy | `64437AF838A65D18E5BA7A0D39B465540069BC8AAE8308DE3E318AAD31FCBC7B` |

The `.task` bundles include both person detection and landmark tracking. They
are used locally and do not upload frames or require an online service.
