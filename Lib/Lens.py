"""Measured pinhole/Brown lens calibration, in native unrotated image pixels."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np


def validate_lens(intrinsics, distortion):
    if not intrinsics and not distortion:
        return
    if len(intrinsics) != 6 or len(distortion) != 5:
        raise ValueError('Lens calibration needs [width,height,fx,fy,cx,cy] and five distortion coefficients')
    if not np.isfinite([*intrinsics, *distortion]).all():
        raise ValueError('Lens calibration values must be finite')
    w,h,fx,fy,cx,cy = intrinsics
    if not (16 <= w <= 16384 and 16 <= h <= 16384 and w == int(w) and h == int(h)
            and .1*w <= fx <= 10*w and .1*h <= fy <= 10*h and 0 <= cx < w and 0 <= cy < h):
        raise ValueError('Lens image size, focal length or principal point is invalid')
    if max(abs(value) for value in distortion) > 5:
        raise ValueError('Lens distortion is outside the supported range')


def load_lens(path):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('Lens calibration file is too large')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'vr-fbt-lens-v1':
        raise ValueError('Unsupported lens calibration format')
    intrinsics = tuple(float(value) for value in data['intrinsics'])
    distortion = tuple(float(value) for value in data['distortion'])
    validate_lens(intrinsics, distortion)
    if not intrinsics:
        raise ValueError('Lens calibration file is empty')
    return intrinsics, distortion


def geometry(setup, size):
    w,h = size
    turn = setup.image_rotation
    nw,nh = (h,w) if turn in (90,270) else (w,h)
    rw,rh,fx,fy,cx,cy = setup.lens_intrinsics
    if abs(nw/nh-rw/rh) > .001:
        raise ValueError('Camera aspect ratio changed; restore the calibrated video mode or recalibrate its lens')
    sx,sy = nw/rw,nh/rh
    k = np.array([[fx*sx,0,(cx+.5)*sx-.5],[0,fy*sy,(cy+.5)*sy-.5],[0,0,1.]])
    transform = {
        0: np.eye(3),
        90: np.array([[0,-1,nh-1],[1,0,0],[0,0,1.]]),
        180: np.array([[-1,0,nw-1],[0,-1,nh-1],[0,0,1.]]),
        270: np.array([[0,1,0],[-1,0,nw-1],[0,0,1.]])
    }[turn]
    center = transform @ np.array([k[0,2],k[1,2],1.])
    ax,ay = (k[1,1],k[0,0]) if turn in (90,270) else (k[0,0],k[1,1])
    rotated_k = np.array([[ax,0,center[0]],[0,ay,center[1]],[0,0,1.]])
    return k, transform, rotated_k


def rectify_observation(observation, setup, cv2):
    k,transform,_ = geometry(setup, observation.frame_size)
    points = np.asarray(observation.pose.image_landmarks, dtype=float).copy()
    scale = np.asarray(observation.frame_size)-1
    pixels = np.column_stack((points[:,:2]*scale,np.ones(len(points))))
    native = (np.linalg.inv(transform) @ pixels.T).T[:,:2]
    corrected = cv2.undistortPoints(native.reshape(-1,1,2),k,np.asarray(setup.lens_distortion),P=k).reshape(-1,2)
    rotated = (transform @ np.column_stack((corrected,np.ones(len(points)))).T).T[:,:2]
    if not np.isfinite(rotated).all():
        raise ValueError('Lens correction returned invalid coordinates')
    points[:,:2] = rotated / scale
    return replace(observation, pose=replace(observation.pose,image_landmarks=points.tolist()))
