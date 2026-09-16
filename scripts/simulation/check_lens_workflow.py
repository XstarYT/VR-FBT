"""Exercise checkerboard detection and the calibration CLI on rendered boards."""
from pathlib import Path
import json
import subprocess
import sys
import cv2
import numpy as np

root=Path(__file__).resolve().parents[2]
directory=root/'review/simulation/lens-fixture'
directory.mkdir(exist_ok=True)
rng=np.random.default_rng(7)
k=np.array([[820.,0,475],[0,825.,354],[0,0,1.]])
board=np.full((700,1000),255,np.uint8)
for y in range(7):
    for x in range(10):
        if (x+y)%2==0:board[y*100:(y+1)*100,x*100:(x+1)*100]=0
plane=np.array([[0,0,0],[.25,0,0],[.25,.175,0],[0,.175,0]],np.float32)
count=0
for attempt in range(2000):
    rotation=rng.uniform(-.65,.65,3)
    translation=np.array([rng.uniform(-.36,.18),rng.uniform(-.28,.14),rng.uniform(.45,.9)])
    projected,_=cv2.projectPoints(plane,rotation,translation,k,None)
    points=projected.reshape(-1,2)
    if np.any(points < [12,12]) or np.any(points > [948,708]):continue
    transform=cv2.getPerspectiveTransform(np.float32([[0,0],[1000,0],[1000,700],[0,700]]),points)
    image=cv2.warpPerspective(board,transform,(960,720),borderValue=200)
    cv2.imwrite(str(directory/f'board-{count:02}.png'),image)
    count+=1
    if count==24:break
assert count==24
# Use a new output on every invocation, preserving earlier evidence.
import tempfile
with tempfile.TemporaryDirectory() as temporary:
    target=Path(temporary)/'calibration.json'
    subprocess.run([sys.executable,str(root/'scripts/calibrate_lens.py'),'--images',str(directory),'--output',str(target)],check=True)
    result=json.loads(target.read_text())
    assert abs(result['intrinsics'][2]-820)/820 < .03,result
    assert abs(result['intrinsics'][3]-825)/825 < .03,result
    (root/'review/simulation/lens-workflow-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Checkerboard image detection, held-out fit, JSON export and focal recovery passed')
