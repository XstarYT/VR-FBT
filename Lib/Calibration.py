"""Fit a lens from diverse checkerboard views with held-out validation."""
import numpy as np
from Lib.Lens import validate_lens


def calibrate_lens(corners, size, board=(9,6), square_m=.025):
    import cv2
    w,h = size
    if min(board) < 3 or max(board) > 30 or not 0 < square_m <= 1:
        raise ValueError('Invalid checkerboard dimensions')
    if len(corners) < 12:
        raise ValueError('At least 12 diverse checkerboard views are required')
    observations = [np.asarray(item,dtype=np.float32).reshape(-1,2) for item in corners]
    count = board[0]*board[1]
    if any(len(item) != count or not np.isfinite(item).all() for item in observations):
        raise ValueError('Invalid checkerboard corners')
    centers = np.array([item.mean(axis=0) for item in observations])
    spans = np.ptp(centers,axis=0)/[w,h]
    hull = cv2.convexHull(np.vstack(observations))
    coverage = cv2.contourArea(hull)/(w*h)
    if min(spans) < .2 or coverage < .35:
        raise ValueError('Move and tilt the board across the image, including corners and edges')
    # Reject repeats before splitting; identical frames cannot validate a lens.
    for i,item in enumerate(observations):
        if any(np.mean(np.linalg.norm(item-other,axis=1)) < .005*min(w,h) for other in observations[:i]):
            raise ValueError('Checkerboard views repeat; capture distinct positions and angles')
    world = np.zeros((count,3),np.float32)
    world[:,:2] = np.mgrid[0:board[0],0:board[1]].T.reshape(-1,2)*square_m
    held = [i for i in range(len(observations)) if i%4 == 0]
    train = [i for i in range(len(observations)) if i not in held]
    rms,k,d,_,_ = cv2.calibrateCamera([world]*len(train),[observations[i] for i in train],size,None,None)
    validation = []
    for i in held:
        ok,r,t = cv2.solvePnP(world,observations[i],k,d)
        if not ok:
            raise ValueError('Held-out board pose could not be solved')
        projected,_ = cv2.projectPoints(world,r,t,k,d)
        validation.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-observations[i])**2,axis=1)))))
    if not np.isfinite([rms,*validation]).all() or rms > 1.5 or max(validation) > 2.:
        raise ValueError('Lens fit failed validation; use sharper, flatter, more varied board views')
    intrinsics = (w,h,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]))
    distortion = tuple(float(value) for value in d.reshape(-1))
    validate_lens(intrinsics,distortion)
    return {'schema':'vr-fbt-lens-v1','intrinsics':intrinsics,'distortion':distortion,
            'training_rms_px':float(rms),'held_out_rms_px':validation,
            'training_views':len(train),'held_out_views':len(held),'image_coverage':coverage,
            'board_inner_corners':board,'square_m':square_m}
