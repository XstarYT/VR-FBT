# Model and fusion comparison - 16 September 2026

DWPose + Full is implemented, installed locally and selectable in setup. Full
remains the default. The alternative improves measured accuracy, but **no tested
rendered-human configuration reaches the 90% reliable-joint coverage target**.

[Watch the full three-camera DWPose video](dwpose-three-cameras.mp4): 32 seconds,
480 frames at 15 FPS, 1440x440 H.264. Motion starts at 24 seconds. Green lines
show confident detector joints; orange rings show projected animation-rig joints.
This is the normal scenario's camera imagery with cached production 2D detections,
not fused 3D/VRChat output or a visualization of delayed frame delivery. The file
was decoded end to end, and T-pose, walk, run and turn frames were visually checked.
Recreate it with `python scripts/simulation/video.py --quality dwpose --full-duration`.

## Identical rendered motion

The fixture uses a 3D animated Soldier, two or three front views, 960x720 images
at 15 FPS, 24 seconds of T-pose followed by eight seconds of walking, running and
turning. Measurements use the production calibration, synchronization and fusion.
The DWPose result uses all 1,440 images through the application's actual `Pose`
implementation. Earlier Full/Heavy baselines preserve the fusion before these
improvements. Full with updated fusion isolates the model's additional benefit.

Error columns include joints with output confidence >=0.5, measured against rig
joint positions at the selected capture time. Coverage counts **all** evaluated
motion-phase joints and requires confidence >=0.5 plus error <=20 cm.

| Model / fusion | Cameras | Mean error | p95 error | Reliable coverage |
| --- | ---: | ---: | ---: | ---: |
| Full / earlier baseline | 2 | 13.88 cm | 48.14 cm | 73.8% |
| Full / earlier baseline | 3 | 13.18 cm | 48.84 cm | 72.8% |
| Heavy / earlier baseline | 2 | 18.29 cm | 64.29 cm | 69.1% |
| Heavy / earlier baseline | 3 | 17.87 cm | 65.21 cm | 67.5% |
| Full / updated fusion | 2 | 8.97 cm | 25.37 cm | 69.1% |
| Full / updated fusion | 3 | 9.51 cm | 29.45 cm | 79.5% |
| DWPose + Full / updated fusion | 2 | 5.66 cm | 11.66 cm | 77.9% |
| DWPose + Full / updated fusion | 3 | 5.44 cm | 11.11 cm | 80.4% |

This supports offering DWPose refinement; it does not support switching to Heavy.
Rejecting incorrect output can reduce coverage while improving confident-joint
error. Including low-confidence points, three-camera DWPose mean/p95 is
11.03/47.76 cm. The confidence threshold is therefore material to these results.

## Camera timing and interruptions

These replays inject completion delays of 10/80/40 ms and use capture timestamps.
Jitter adds 0-50 ms. Dropout removes camera 2 for two seconds during running.
Late adds 180 ms to that camera during motion. These are controlled timing inputs,
not measurements of physical camera latency or the concurrent CPU benchmark.

| DWPose scenario | 2-camera coverage | 3-camera coverage | 3-camera confident p95 |
| --- | ---: | ---: | ---: |
| Normal | 77.9% | 80.4% | 11.11 cm |
| Jitter | 78.0% | 80.4% | 11.07 cm |
| Two-second dropout | 57.4% | 79.2% | 11.25 cm |
| Additional 180 ms delay | 0% | 71.4% | 11.93 cm |
| Incorrect arrival-time timestamps | 71.1% | 0% | No calibration |

The third camera preserves useful output during one source's interruption.
Two-camera tracking cannot establish a reliable live room pose with one usable
view in the late case, and withholds confident output. Incorrect timestamps can
still break calibration; a better detector does not solve clock alignment.
All correct-timestamp scenarios selected zero-skew captures in this discrete
fixture. At live time, including all low-confidence joints and synchronization
delay, normal three-camera mean/p95 error is 20.53/52.53 cm.

## Processing cost and checks

Sequential production inference plus JPEG loading averaged 63.95 ms, p95 75.66 ms.
Three concurrent supervised CPU workers on this PC, each repeating a saved person
image, measured median 78.90-81.95 ms and p95 87.06-89.92 ms per request including
IPC. All 30 measured frames per worker detected the person; startup took 4.12 s
and the benchmark closed its workers. This is roughly 12 FPS per worker and adds
to capture/network/display delays. It is not a moving three-camera end-to-end
latency measurement. The current adapter does not use the NVIDIA GPU.

The full software gate passed 156 Python tests in 37.35 seconds, both phone-page
JavaScript suites, regression reproductions, compilation, dependency consistency
and diagnostics. New checks cover outlier cameras, degenerate baselines, partial
T-pose observability, invalid model outputs and checksum-safe installation.

## Implementation and remaining limits

- Optional checksum-pinned DWPose-l ONNX inference through the existing OpenCV
  dependency; MediaPipe Full supplies the person crop and coarse 3D body.
- Camera-pair consensus rejects inconsistent rays and near-parallel baselines.
- Strong, anatomically plausible triangulation can override an inaccurate coarse
  body estimate; fallback estimates need current image agreement for confidence.
- Known camera layouts can establish a T-pose from complementary views when each
  required joint has two usable rays. The existing Full side-view fixture still
  fails calibration; DWPose side-view acceptance has not been established.

One character, fixed lighting/background and approximate rig joint centers are
limited evidence. Real people, clothing, lens distortion, camera movement,
occlusion, the final VRChat solver and headset/avatar alignment remain unverified.

## Reproduce

Generate the frames as described in [the original report](REPORT.md), then:

```powershell
.venv\Scripts\python.exe scripts/install_refinement.py
.venv\Scripts\python.exe scripts/simulation/compare_models.py --quality dwpose
.venv\Scripts\python.exe scripts/simulation/compare_models.py --quality full --reuse-full
.venv\Scripts\python.exe scripts/benchmark.py --quality dwpose --cameras 3 --frames 30 --image review/simulation/frames/camera-0-0360.jpg --output review/simulation/dwpose-worker-benchmark.json
.venv\Scripts\python.exe scripts/verify.py
```

Raw data: [Full baseline](front/full-comparison-baseline.json),
[Heavy baseline](front/heavy-comparison-baseline.json),
[updated Full](front/full-comparison.json),
[production DWPose](front/dwpose-comparison.json),
[concurrent workers](dwpose-worker-benchmark.json).
Model provenance and installation are in [Model/dwpose](../../Model/dwpose/README.md).
