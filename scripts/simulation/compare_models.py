"""Compare existing pose models on identical saved rendered images."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
from dataclasses import asdict
import hashlib
import json
import time
import cv2
import numpy as np
from Lib.Tracking import Pose,PoseResult,MultiCameraPoseFusion
from Lib.Config import CameraSetup
import evaluate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality',choices=['lite','full','heavy','dwpose'],default='heavy');parser.add_argument('--reuse-full',action='store_true');args=parser.parse_args()
    directory=ROOT/'review/simulation/front'
    manifest=json.loads((directory/'manifest.json').read_text())
    cache=directory/f'{args.quality}-comparison-observations.json'
    model_path=ROOT/f'Model/mediapipe/pose_landmarker_{"full" if args.quality == "dwpose" else args.quality}.task'
    model_bytes=model_path.read_bytes()
    if args.quality == 'dwpose':
        model_bytes+=(ROOT/'Model/dwpose/dwpose-l.onnx').read_bytes()
    fingerprint=hashlib.sha256(model_bytes+(ROOT/'Lib/Tracking.py').read_bytes()+(ROOT/'Lib/Refinement.py').read_bytes()+(directory/'manifest.json').read_bytes()).hexdigest()
    timings=[]
    if args.reuse_full:
        if args.quality!='full':raise ValueError('--reuse-full requires --quality full')
        saved=json.loads((directory/'observations.json').read_text())
        poses=[PoseResult(**item) for item in saved['poses']]
        timings=saved['inference_ms']
    elif cache.exists() and json.loads(cache.read_text()).get('fingerprint')==fingerprint:
        saved=json.loads(cache.read_text());poses=[PoseResult(**item) for item in saved['poses']];timings=saved['inference_ms']
    else:
        models=[Pose(args.quality) for _ in range(3)];poses=[]
        try:
            for index,frame in enumerate(manifest['frames']):
                start=time.perf_counter()
                poses.append(models[frame['cameraIndex']].process(cv2.imread(str(directory/frame['file'])),round(frame['time']*1000)))
                timings.append((time.perf_counter()-start)*1000)
                if index%90==0:print(f'{args.quality}: {index}/{len(manifest["frames"])}',flush=True)
        finally:
            for model in models:model.close()
        cache.write_text(json.dumps({'fingerprint':fingerprint,'poses':[asdict(pose) for pose in poses],'inference_ms':timings}))
    report={'quality':args.quality,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'inference_ms':evaluate.summary(timings),'fusion':[]}
    for cameras in (2,3):
        for scenario in ('normal','jitter','dropout','late','arrival_only'):
            result,trace=evaluate.run(manifest,poses,cameras,scenario,'rendered_detector')
            report['fusion'].append(result)
    # Unconditioned triangulation diagnoses errors introduced by body fitting.
    setups=[CameraSetup(**item) for item in manifest['setups']]
    projections=[]
    for setup in setups:
        r,t,_,_=MultiCameraPoseFusion._manual_camera_geometry(setup,np)
        f=960/(2*np.tan(np.pi/6));k=np.array([[f,0,480],[0,f,360],[0,0,1]])
        projections.append(k@np.column_stack((r,t)))
    report['joints']={}
    for joint,name in evaluate.JOINTS.items():
        pixels=[];triangulated=[]
        for tick in range(360,480):
            rows=[]
            for camera in range(3):
                index=tick*3+camera;pose=poses[index];frame=manifest['frames'][index]
                if not pose.detected or pose.image_landmarks[joint][3]<.5:continue
                point=pose.image_landmarks[joint];u,v=np.array(point[:2])*[959,719]
                pixels.append(float(np.linalg.norm([u-frame['pixels']['mixamorig'+name][0],v-frame['pixels']['mixamorig'+name][1]])))
                p=projections[camera];w=np.sqrt(point[3]);rows.extend((w*(u*p[2]-p[0]),w*(v*p[2]-p[1])))
            if len(rows)>=4:
                _,_,vt=np.linalg.svd(rows);point=vt[-1,:3]/vt[-1,3]
                expected=evaluate.truth(manifest['frames'][tick*3])[joint]
                triangulated.append(float(np.linalg.norm(point-expected)))
        report['joints'][name]={'image_error_px':evaluate.summary(pixels),'raw_triangulation_error_m':evaluate.summary(triangulated)}
    target=directory/f'{args.quality}-comparison.json';target.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
