"""Experimental DWPose image-landmark refinement using MediaPipe person bounds."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'review/simulation/model-libs'))
import json
from dataclasses import asdict
import time
import cv2
import numpy as np
from rtmlib import RTMPose
from Lib.Tracking import PoseResult
import evaluate

directory=ROOT/'review/simulation/front'
manifest=json.loads((directory/'manifest.json').read_text())
base=json.loads((directory/'observations.json').read_text())['poses']
cache=directory/'dwpose-comparison-observations.json'
mapping={0:0,11:5,12:6,13:7,14:8,15:9,16:10,23:11,24:12,25:13,26:14,27:15,28:16,29:17,30:20}
if cache.exists():
    saved=json.loads(cache.read_text());poses=[PoseResult(**item) for item in saved['poses']];timings=saved['inference_ms']
else:
    model=RTMPose(str(ROOT/'review/simulation/assets/dwpose.onnx'),backend='opencv',device='cpu')
    poses=[];timings=[]
    for index,(frame,item) in enumerate(zip(manifest['frames'],base)):
        pose=PoseResult(**item)
        if pose.detected:
            image=cv2.imread(str(directory/frame['file']));height,width=image.shape[:2]
            visible=np.array([p[:2] for p in pose.image_landmarks if p[3]>=.3])*[width-1,height-1]
            low=visible.min(axis=0);high=visible.max(axis=0);margin=(high-low)*.1
            bbox=[max(0,low[0]-margin[0]),max(0,low[1]-margin[1]),min(width,high[0]+margin[0]),min(height,high[1]+margin[1])]
            started=time.perf_counter();points,scores=model(image,bboxes=[bbox]);timings.append((time.perf_counter()-started)*1000)
            for destination,source in mapping.items():
                point=points[0,source];score=float(np.clip(scores[0,source],0,1))
                pose.image_landmarks[destination]=[float(point[0]/(width-1)),float(point[1]/(height-1)),0.,score]
        poses.append(pose)
        if index%90==0:print(f'DWPose {index}/{len(base)}',flush=True)
    cache.write_text(json.dumps({'poses':[asdict(p) for p in poses],'inference_ms':timings}))
report={'model':'DWPose-l image refinement + cached MediaPipe Full person bounds/world pose','extra_inference_ms':evaluate.summary(timings),'fusion':[]}
for cameras in (2,3):
    for mode in ('rendered_detector','rendered_known_geometry'):
        result,trace=evaluate.run(manifest,poses,cameras,'normal',mode);report['fusion'].append(result)
        print(json.dumps(result),flush=True)
(directory/'dwpose-comparison.json').write_text(json.dumps(report,indent=2))
