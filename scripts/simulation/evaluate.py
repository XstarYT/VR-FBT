"""Measure production fusion on rendered images and an exact-joint control.

Run from the repository root with .venv/Scripts/python scripts/simulation/evaluate.py.
No camera, headset, network sender or saved application profile is touched.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
from dataclasses import asdict
import json
import math
import time
import cv2
import numpy as np
from Lib.Config import CameraSetup
from Lib.Synchronization import FrameSynchronizer
from Lib.Tracking import Pose, PoseResult, CameraObservation, MultiCameraPoseFusion

OUT = ROOT / 'review/simulation'
JOINTS = {11:'LeftArm',12:'RightArm',13:'LeftForeArm',14:'RightForeArm',15:'LeftHand',16:'RightHand',
          23:'LeftUpLeg',24:'RightUpLeg',25:'LeftLeg',26:'RightLeg',27:'LeftFoot',28:'RightFoot',29:'LeftToeBase',30:'RightToeBase'}
EVALUATED = (11,12,13,14,15,16,23,24,25,26,27,28)

def truth(frame):
    joints=frame['joints']
    points=np.array([joints['mixamorigHead'] for _ in range(31)])
    for index,name in JOINTS.items():
        points[index]=joints['mixamorig'+name]
    for index,name in ((17,'LeftHandPinky1'),(18,'RightHandPinky1'),(19,'LeftHandIndex1'),(20,'RightHandIndex1'),(21,'LeftHandThumb1'),(22,'RightHandThumb1')):
        points[index]=joints['mixamorig'+name]
    return points

def ideal(frame, setup):
    points=truth(frame)
    rotation,translation,_,_=MultiCameraPoseFusion._manual_camera_geometry(setup,np)
    camera=(rotation@points.T).T+translation
    focal=960/(2*math.tan(math.pi/6))
    pixels=camera[:,:2]/camera[:,2,None]*focal+np.array([480,360])
    relative=(rotation@(points-(points[23]+points[24])/2).T).T
    return PoseResult(True,.99,np.column_stack((pixels/[959,719],np.zeros(31),np.full(31,.99))).tolist(),
                      np.column_stack((-relative,np.full(31,.99))).tolist(),np.column_stack((relative,np.full(31,.99))).tolist())

def infer(manifest, refresh=False):
    cache=OUT/'observations.json'
    digest=hashlib.sha256((OUT/'manifest.json').read_bytes())
    digest.update((ROOT/'Model/mediapipe/pose_landmarker_full.task').read_bytes())
    digest.update((ROOT/'Lib/Tracking.py').read_bytes())
    for frame in manifest['frames']:
        digest.update((OUT/frame['file']).read_bytes())
    fingerprint=digest.hexdigest()
    if cache.exists() and not refresh:
        saved=json.loads(cache.read_text())
        if saved.get('fingerprint')==fingerprint:
            return [PoseResult(**item) for item in saved['poses']]
    models=[Pose('full') for _ in range(3)]
    borrowed=json.loads((OUT.parent/'observations.json').read_text())['poses'] if manifest.get('rig')=='front' else None
    poses=[];elapsed=[]
    try:
        for index,frame in enumerate(manifest['frames']):
            if borrowed is not None and frame['file'].startswith('../frames/'):
                poses.append(PoseResult(**borrowed[index]));continue
            image=cv2.imread(str(OUT/frame['file']))
            started=time.perf_counter()
            poses.append(models[frame['cameraIndex']].process(image,int(frame['time']*1000)))
            elapsed.append((time.perf_counter()-started)*1000)
            if index%90==0: print(f'Inference {index}/{len(manifest["frames"])}',flush=True)
    finally:
        for model in models:model.close()
    cache.write_text(json.dumps({'fingerprint':fingerprint,'frame_count':len(poses),'poses':[asdict(pose) for pose in poses],'inference_ms':elapsed}))
    return poses

def summary(values):
    return None if not values else {name:round(float(value),4) for name,value in zip(('mean','p50','p95','max'),(np.mean(values),*np.percentile(values,[50,95]),max(values)))}

def run(manifest, poses, count, scenario, mode):
    setups=[CameraSetup(**item) for item in manifest['setups'][:count]]
    sources=[item.source_id for item in setups]
    now=[0.0]
    fusion=MultiCameraPoseFusion(sources,manual_camera_setup=True,camera_setups=setups,room_size_m=(6,3,6),
                               calibration_frames=12,calibration_duration_seconds=None if mode=='rendered_known_geometry' else 10,clock=lambda:now[0])
    sync=FrameSynchronizer(sources)
    events=[]
    reference={frame['time']:truth(frame) for frame in manifest['frames'] if frame['cameraIndex']==0}
    reference_times=sorted(reference)
    for frame,pose in zip(manifest['frames'],poses):
        camera=frame['cameraIndex'];stamp=frame['time']
        if camera>=count:continue
        delay=(.010,.080,.040)[camera]
        if scenario=='jitter':delay+=.025*(1+math.sin(stamp*7+camera))
        if scenario=='dropout' and camera==1 and 26<=stamp<28:continue
        if scenario=='late' and camera==1 and stamp>=24:delay+=.180
        received=stamp+delay
        reported=received if scenario=='arrival_only' else stamp
        events.append((received,CameraObservation(sources[camera],pose,(960,720),reported)))
    events.sort(key=lambda item:item[0])
    cursor=0;calibrated_at=None;errors=[];live_errors=[];valid_errors=[];motion_ticks=0;detected_ticks=0
    valid_count=0;good_count=0;trace=[];per_phase={};last_hint='';max_skew=0
    safety_paused_ticks=0; warnings=set()
    for tick in range(961):
        now[0]=tick/30
        while cursor<len(events) and events[cursor][0]<=now[0]+1e-9:
            sync.add(events[cursor][1], now[0]);cursor+=1
        selected=sync.select(now[0])
        result=fusion.update(selected,cv2)
        if result.calibrated and result.calibration_hint:
            warnings.add(result.calibration_hint)
        last_hint=result.calibration_hint
        if result.calibrated and calibrated_at is None:calibrated_at=now[0]
        if len(selected)>1:max_skew=max(max_skew,max(o.captured_at for o in selected)-min(o.captured_at for o in selected))
        if now[0]<24.15:continue
        motion_ticks+=1
        safety_paused_ticks+=int(result.safety_paused)
        if not result.calibrated or not result.pose.detected:continue
        detected_ticks+=1
        estimated_stamp=float(np.mean([o.captured_at for o in selected]))
        nearest=min(reference_times,key=lambda value:abs(value-estimated_stamp))
        live_nearest=min(reference_times,key=lambda value:abs(value-now[0]))
        expected=reference[nearest];actual=np.array(result.pose.world_landmarks)
        if actual.shape!=(31,4):continue
        delta=np.linalg.norm(actual[list(EVALUATED),:3]-expected[list(EVALUATED)],axis=1)
        live_delta=np.linalg.norm(actual[list(EVALUATED),:3]-reference[live_nearest][list(EVALUATED)],axis=1)
        eligible=actual[list(EVALUATED),3]>=.5
        errors.extend(delta.tolist());live_errors.extend(live_delta.tolist());valid_errors.extend(delta[eligible].tolist())
        valid_count+=int(eligible.sum());good_count+=int(((delta<=.2)&eligible).sum())
        phase='walk' if nearest<26 else 'run' if nearest<28 else 'turn'
        per_phase.setdefault(phase,[]).extend(delta[eligible].tolist())
        trace.append({'time':round(now[0],4),'sources':len(selected),'mean_error_m':float(delta.mean()),
                      'expected_hip':((expected[23]+expected[24])/2).tolist(),'actual_hip':((actual[23,:3]+actual[24,:3])/2).tolist()})
    denominator=motion_ticks*len(EVALUATED)
    result={'camera_count':count,'scenario':scenario,'input':mode,'calibrated_at_s':calibrated_at,'calibration_hint':last_hint,
            'motion_frames':motion_ticks,'detected_fraction':detected_ticks/max(1,motion_ticks),
            'eligible_joint_fraction':valid_count/max(1,denominator),'within_20cm_fraction':good_count/max(1,denominator),
            'all_joint_error_m':summary(errors),'eligible_joint_error_m':summary(valid_errors),'live_time_error_m':summary(live_errors),
            'phase_error_m':{key:summary(value) for key,value in per_phase.items()},'max_selected_skew_ms':round(max_skew*1000,3)}
    result['acceptance_pass']=bool(calibrated_at is not None and result['within_20cm_fraction']>=.9 and valid_errors and np.percentile(valid_errors,95)<=.2)
    result['safety_paused_fraction']=safety_paused_ticks/max(1,motion_ticks)
    result['geometry_warnings']=sorted(warnings)
    result['quarantined_sources']=sorted(fusion._geometry_health.quarantined)
    return result,trace

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');parser.add_argument('--front',action='store_true');args=parser.parse_args()
    if args.front:OUT=OUT/'front'
    manifest=json.loads((OUT/'manifest.json').read_text());poses=infer(manifest,args.refresh)
    setups=[CameraSetup(**item) for item in manifest['setups']]
    perfect=[ideal(frame,setups[frame['cameraIndex']]) for frame in manifest['frames']]
    report={'description':'Rendered Soldier model, 24s T-pose then 8s walk/run/half-turn, 960x720 at 15 FPS, real Full model, production 10s calibration and synchronizer.',
            'threshold':'At least 90% of all motion-phase evaluated joints emitted at confidence >=0.5 and within 20cm; eligible-joint p95 <=20cm. Skeleton bone centers approximate anatomical landmarks.',
            'detector':{},'results':[]}
    for camera in range(3):
        selected=[pose for frame,pose in zip(manifest['frames'],poses) if frame['cameraIndex']==camera]
        report['detector'][str(camera)]={'detection_fraction':sum(p.detected for p in selected)/len(selected),'mean_confidence':float(np.mean([p.confidence for p in selected]))}
    traces={}
    for mode,data in [('exact_joints',perfect),('rendered_detector',poses),('rendered_known_geometry',poses)]:
        for count in (2,3):
            if args.front and count==2:continue  # cameras 0/1 and their observations are unchanged
            for scenario in ('normal','jitter','dropout','late','arrival_only'):
                result,trace=run(manifest,data,count,scenario,mode)
                report['results'].append(result);traces[f'{mode}-{count}-{scenario}']=trace
                print(json.dumps(result),flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    (OUT/'traces.json').write_text(json.dumps(traces))
    # A reviewed contact sheet: detector skeleton (green) and rig joint centers (orange).
    panels=[]
    for frame_index in (0,375,405,450):
        row=[]
        for camera in range(3):
            index=frame_index*3+camera;frame=manifest['frames'][index];pose=poses[index]
            picture=cv2.imread(str(OUT/frame['file']))
            for joint,name in JOINTS.items():
                p=frame['pixels']['mixamorig'+name];cv2.circle(picture,tuple(round(v) for v in p),5,(0,130,255),2)
                if pose.detected:
                    p=pose.image_landmarks[joint];cv2.circle(picture,(round(p[0]*959),round(p[1]*719)),4,(0,200,0),-1)
            cv2.putText(picture,f'Camera {camera+1} / {frame["phase"]} / {frame["time"]:.1f}s',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
            row.append(cv2.resize(picture,(480,360)))
        panels.append(np.hstack(row))
    cv2.imwrite(str(OUT/'contact-sheet.jpg'),np.vstack(panels))

if __name__=='__main__':main()
