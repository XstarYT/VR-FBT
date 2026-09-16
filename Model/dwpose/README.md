# Optional DWPose refinement

Run `.venv\Scripts\python.exe scripts/install_refinement.py` from the repository,
then select **DWPose + Full — experimental refinement** in the application.
The installer checks SHA-256 before replacing an existing model. No extra Python
packages are required. Weights are downloaded separately and are not committed.

This mode runs MediaPipe Full for person localization and the coarse 3D body,
then DWPose-l for improved 2D body joints. Multi-camera fusion uses those refined
image coordinates. It currently runs on CPU through OpenCV and adds inference
cost per camera. It does not implement GPU acceleration or multi-person tracking.

Model: RTMPose-l, UBody/COCO WholeBody DWPose, 256x192, released July 2023.

- [Official model archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.zip)
- [DWPose project and licensing](https://github.com/IDEA-Research/DWPose)
- [rtmlib preprocessing reference (Apache-2.0)](https://github.com/Tau-J/rtmlib)

Expected `dwpose-l.onnx` size: 133,879,122 bytes.
SHA-256: `370c6d5777f4b496f718468ef718fe7d4d0f9acbeb7bc6757e0ba80dd692b2d1`.

See [the simulation comparison](../../review/simulation/MODEL_COMPARISON.md)
for measured accuracy and remaining limitations. Complete the dependency/model
license inventory before distributing a binary or bundled weights.
