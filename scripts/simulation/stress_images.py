"""Run actual DWPose inference on a distorted, blurred, noisy third camera."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
import json
import time
import cv2
import numpy as np
from Lib.Tracking import Pose, PoseResult
import evaluate


def main():
    directory=evaluate.OUT/'front'
    manifest=json.loads((directory/'manifest.json').read_text())
    poses=[PoseResult(**item) for item in json.loads((directory/'dwpose-comparison-observations.json').read_text())['poses']]
    f=960/(2*np.tan(np.pi/6));k=np.array([[f,0,480],[0,f,360],[0,0,1.]])
    d=np.array([-.3,.08,.002,-.001,0.])
    y,x=np.mgrid[0:720,0:960]
    pixels=np.column_stack((x.ravel(),y.ravel())).astype(np.float32)
    mapping=cv2.undistortPoints(pixels.reshape(-1,1,2),k,d,P=k).reshape(720,960,2)
    rng=np.random.default_rng(31);timings=[];examples=[]
    model=Pose('dwpose')
    try:
        for index,frame in enumerate(manifest['frames']):
            if frame['cameraIndex']!=2:continue
            clean=cv2.imread(str(directory/frame['file']))
            image=cv2.remap(clean,mapping[:,:,0],mapping[:,:,1],cv2.INTER_LINEAR)
            image=cv2.GaussianBlur(image,(5,5),1.2)
            image=np.clip(image.astype(float)+rng.normal(0,5,image.shape),0,255).astype(np.uint8)
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,35])
            if not ok:raise RuntimeError('JPEG encoding failed')
            image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            before=time.perf_counter()
            poses[index]=model.process(image,round(frame['time']*1000))
            timings.append((time.perf_counter()-before)*1000)
            if index//3 in (0,375,405,465):
                examples.append(np.hstack((cv2.resize(clean,(480,360)),cv2.resize(image,(480,360)))))
            if index//3%60==0:print(f'Camera 3: {index//3}/480',flush=True)
    finally:model.close()
    report={'method':__doc__,'perturbation':{'radial_tangential_coefficients':d.tolist(),'blur_sigma':1.2,'noise_std':5,'jpeg_quality':35,'seed':31},'inference_ms':evaluate.summary(timings),'results':[]}
    for corrected in (False,True):
        data=copy.deepcopy(manifest)
        if corrected:
            data['setups'][2]['lens_intrinsics']=[960,720,f,f,480,360]
            data['setups'][2]['lens_distortion']=d.tolist()
        result,_=evaluate.run(data,poses,3,'normal','rendered_detector')
        result['measured_lens_supplied']=corrected;report['results'].append(result)
        print(json.dumps(result),flush=True)
    (evaluate.OUT/'image-stress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cv2.imwrite(str(evaluate.OUT/'image-stress-examples.jpg'),np.vstack(examples))


if __name__=='__main__':main()
