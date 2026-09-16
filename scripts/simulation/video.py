"""Export a short rendered-camera evidence clip (2D overlays, not avatar output)."""
from pathlib import Path
import argparse
import json
import av
import cv2
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'review/simulation'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--quality',choices=('full','dwpose'),default='full')
parser.add_argument('--full-duration',action='store_true',help='Include all 24 seconds of calibration and eight seconds of motion')
args=parser.parse_args()
directory=OUT/'front'
manifest=json.loads((directory/'manifest.json').read_text())
cache='dwpose-comparison-observations.json' if args.quality=='dwpose' else 'observations.json'
poses=json.loads((directory/cache).read_text())['poses']
if len(poses)!=len(manifest['frames']):raise ValueError('Pose cache and rendered frame counts differ')
indices=range(480) if args.full_duration else list(range(15))+list(range(360,480))
target=OUT/('dwpose-three-cameras.mp4' if args.quality=='dwpose' else 'rendered-cameras.mp4')
joint_names={11:'LeftArm',12:'RightArm',13:'LeftForeArm',14:'RightForeArm',15:'LeftHand',16:'RightHand',23:'LeftUpLeg',24:'RightUpLeg',25:'LeftLeg',26:'RightLeg',27:'LeftFoot',28:'RightFoot'}
links=((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28))
with av.open(str(target),'w') as output:
    stream=output.add_stream('libx264',rate=15)
    stream.width=1440;stream.height=440;stream.pix_fmt='yuv420p'
    stream.options={'crf':'23','preset':'fast'}
    for frame_index in indices:
        panels=[]
        for camera in range(3):
            index=frame_index*3+camera
            frame=manifest['frames'][index];pose=poses[index]
            image=cv2.imread(str(directory/frame['file']))
            if image is None:raise ValueError(f'Missing rendered frame: {frame["file"]}')
            for name in joint_names.values():
                point=frame['pixels']['mixamorig'+name]
                cv2.circle(image,tuple(round(v) for v in point),7,(0,130,255),2)
            if pose['detected']:
                points=pose['image_landmarks']
                for a,b in links:
                    if min(points[a][3],points[b][3])<.5:continue
                    pa=tuple(round(v*s) for v,s in zip(points[a][:2],(959,719)))
                    pb=tuple(round(v*s) for v,s in zip(points[b][:2],(959,719)))
                    cv2.line(image,pa,pb,(40,220,40),3)
            image=cv2.resize(image,(480,360))
            cv2.putText(image,f'Camera {camera+1} / {frame["time"]:.2f}s / {frame["phase"]}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),2)
            panels.append(image)
        canvas=np.full((440,1440,3),245,np.uint8);canvas[:360]=np.hstack(panels)
        label='DWPose + MediaPipe Full' if args.quality=='dwpose' else 'MediaPipe Full'
        cv2.putText(canvas,f'{label} | Green: detected joints (confidence >= 0.5) | Orange: projected animation rig joints',
                    (15,386),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,25,25),1)
        cv2.putText(canvas,'Same rendered inputs as the offline accuracy test. These are 2D overlays, not fused 3D or VRChat output.',
                    (15,417),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,25,25),1)
        video_frame=av.VideoFrame.from_ndarray(canvas,format='bgr24')
        for packet in stream.encode(video_frame):output.mux(packet)
    for packet in stream.encode():output.mux(packet)
print(target)
