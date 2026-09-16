"""Perturb cached production detections to isolate lens/layout guard behavior.

This is a geometry fault injection, not a new inference test on distorted images.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
import copy
import json
import cv2
import numpy as np
from Lib.Tracking import PoseResult
import evaluate


def main():
    directory=evaluate.ROOT/'review/simulation/front'
    manifest=json.loads((directory/'manifest.json').read_text())
    original=[PoseResult(**item) for item in json.loads((directory/'dwpose-comparison-observations.json').read_text())['poses']]
    focal=960/(2*np.tan(np.pi/6));k=np.array([[focal,0,480],[0,focal,360],[0,0,1.]])
    distortion=np.array([-.3,.08,.002,-.001,0.])
    cases=('unchanged','moved_third_camera','misaligned_two','wrong_fov_third','distorted_uncorrected','distorted_corrected')
    report={'method':__doc__,'results':[]}
    for case in cases:
        data=copy.deepcopy(manifest);poses=[]
        count=2 if case=='misaligned_two' else 3
        if case=='wrong_fov_third':data['setups'][2]['horizontal_fov']=80
        if case=='distorted_corrected':
            for setup in data['setups']:
                setup['lens_intrinsics']=[960,720,focal,focal,480,360]
                setup['lens_distortion']=distortion.tolist()
        for frame,pose in zip(manifest['frames'],original):
            if not pose.detected:
                poses.append(pose);continue
            points=np.asarray(pose.image_landmarks).copy()
            if case.startswith('distorted_'):
                rays=np.column_stack(((points[:,:2]*[959,719]-[480,360])/focal,np.ones(len(points))))
                pixels,_=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),k,distortion)
                points[:,:2]=pixels.reshape(-1,2)/[959,719]
            if frame['time']>=26 and ((case=='moved_third_camera' and frame['cameraIndex']==2) or (case=='misaligned_two' and frame['cameraIndex']==1)):
                rays=np.linalg.inv(k)@np.column_stack((points[:,:2]*[959,719],np.ones(len(points)))).T
                rotation,_=cv2.Rodrigues(np.array([np.deg2rad(8),0.,0.]))
                pixels=k@rotation@rays
                points[:,:2]=(pixels[:2]/pixels[2]).T/[959,719]
            poses.append(replace(pose,image_landmarks=points.tolist()))
        result,_=evaluate.run(data,poses,count,'normal','rendered_detector')
        result['fault']=case;report['results'].append(result)
        print(case,json.dumps(result),flush=True)
    (evaluate.OUT/'geometry-stress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
