"""Optional DWPose-l 2D refinement; retains MediaPipe's coarse 3D body estimate.

OpenCV executes the published SimCC ONNX model. Preprocessing follows the
RTMPose model configuration: 192x256 aspect-preserving crop, 1.25 padding,
BGR normalization, and a two-bin-per-pixel coordinate classification output.
Reference: https://github.com/Tau-J/rtmlib (Apache-2.0).
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / 'Model/dwpose/dwpose-l.onnx'
MODEL_SHA256 = '370c6d5777f4b496f718468ef718fe7d4d0f9acbeb7bc6757e0ba80dd692b2d1'
MODEL_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.zip'
# DWPose uses COCO-WholeBody indices. Unrefined face/finger points keep the
# original MediaPipe observation rather than claiming compatible definitions.
JOINT_MAPPING = {0:0,11:5,12:6,13:7,14:8,15:9,16:10,23:11,24:12,25:13,26:14,27:15,28:16,29:17,30:20}


def verify_model(path=MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError('DWPose refinement model is missing. Run: python scripts/install_refinement.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('DWPose model checksum mismatch. Run: python scripts/install_refinement.py')
    return f'{path.stat().st_size:,} bytes; SHA-256 verified'


class DWPoseRefiner:
    def __init__(self, model_path=MODEL_PATH):
        import cv2
        verify_model(model_path)
        self.cv2 = cv2
        self.network = cv2.dnn.readNetFromONNX(str(model_path))
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.output_names = self.network.getUnconnectedOutLayersNames()

    def refine(self, bgr_frame, pose):
        if not pose.detected:
            return pose
        height, width = bgr_frame.shape[:2]
        visible = np.asarray([point[:2] for point in pose.image_landmarks
                              if point[3] >= .3 and np.isfinite(point[:2]).all()])
        if len(visible) < 4 or min(height, width) < 2:
            return replace(pose, detected=False, confidence=0., image_landmarks=[], world_landmarks=[])
        pixels = visible * [width - 1, height - 1]
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        margin = (high - low) * .1
        low = np.maximum(0, low - margin)
        high = np.minimum([width, height], high + margin)
        center = (low + high) / 2
        scale = (high - low) * 1.25
        if min(scale) < 2:
            return replace(pose, detected=False, confidence=0., image_landmarks=[], world_landmarks=[])
        if scale[0] > scale[1] * .75:
            scale[1] = scale[0] / .75
        else:
            scale[0] = scale[1] * .75
        # Three-point transform also matches float32 coordinate rounding used
        # by the reference affine crop implementation.
        src = np.float32([center, center + [0, -scale[0]/2], center + [-scale[0]/2, -scale[0]/2]])
        dst = np.float32([[96,128],[96,32],[0,32]])
        affine = self.cv2.getAffineTransform(src, dst)
        crop = self.cv2.warpAffine(bgr_frame, affine, (192,256), flags=self.cv2.INTER_LINEAR)
        normalized = (crop - np.array([123.675,116.28,103.53])) / np.array([58.395,57.12,57.375])
        blob = np.ascontiguousarray(normalized.transpose(2,0,1)[None], dtype=np.float32)
        self.network.setInput(blob)
        outputs = self.network.forward(self.output_names)
        by_size = {value.shape[-1]: value for value in outputs}
        if 384 not in by_size or 512 not in by_size or any(value.shape != (1,133,size) for size,value in by_size.items()):
            raise RuntimeError('Unexpected DWPose output shape')
        x, y = by_size[384][0], by_size[512][0]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError('DWPose returned non-finite coordinates or confidence')
        scores = (x.max(axis=1) + y.max(axis=1)) / 2
        coordinates = np.column_stack((x.argmax(axis=1), y.argmax(axis=1))).astype(float)
        coordinates[scores <= 0] = -1
        coordinates = coordinates / 2 / [192,256] * scale + center - scale/2
        image = [point.copy() for point in pose.image_landmarks]
        for destination, source in JOINT_MAPPING.items():
            point = coordinates[source] / [width-1, height-1]
            image[destination] = [float(point[0]), float(point[1]), 0., float(np.clip(scores[source],0,1))]
        return replace(pose, image_landmarks=image)

    def close(self):
        self.network = None
