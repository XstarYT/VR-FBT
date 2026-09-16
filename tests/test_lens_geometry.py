from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from Lib.Config import CameraSetup, Profile, save_profile, load_profile, camera_setup_for_corner
from Lib.Lens import geometry, rectify_observation, validate_lens
from Lib.Tracking import CameraObservation, PoseResult, MultiCameraPoseFusion
from Lib.GeometryHealth import GeometryHealth
from Lib.Calibration import calibrate_lens
from tests import test_triangulation_consensus as consensus_fixture


class LensTests(unittest.TestCase):
    def setUp(self):
        self.setup=CameraSetup('a',(0,1,-3),(0,0,0),lens_intrinsics=(960,720,820,825,475,354),lens_distortion=(-.18,.04,.001,-.002,0.))

    def test_lens_corrects_distortion_for_all_image_rotations_and_resize(self):
        for rotation in (0,90,180,270):
            for factor in (1.,.5):
                setup=replace(self.setup,image_rotation=rotation)
                size=tuple(int(value*factor) for value in ((720,960) if rotation in (90,270) else (960,720)))
                k,t,_=geometry(setup,size)
                body=np.array([[x,y,2.] for x,y in zip(np.linspace(-.8,.8,31),np.sin(np.arange(31))*.5)])
                distorted,_=cv2.projectPoints(body,np.zeros(3),np.zeros(3),k,np.array(setup.lens_distortion))
                ideal,_=cv2.projectPoints(body,np.zeros(3),np.zeros(3),k,None)
                def rotate(points):return (t@np.column_stack((points.reshape(-1,2),np.ones(31))).T).T[:,:2]
                image=np.column_stack((rotate(distorted)/(np.array(size)-1),np.zeros(31),np.ones(31))).tolist()
                observation=CameraObservation('a',PoseResult(True,1.,image,[[0,0,0,1.]]*31),size)
                corrected=rectify_observation(observation,setup,cv2)
                self.assertLess(np.max(np.abs(np.asarray(corrected.pose.image_landmarks)[:,:2]*(np.array(size)-1)-rotate(ideal))),.002)
                self.assertEqual(observation.pose.image_landmarks,image)

    def test_changed_aspect_ratio_pauses_instead_of_using_wrong_lens(self):
        fusion=MultiCameraPoseFusion(('a',),camera_setups=(self.setup,))
        observation=CameraObservation('a',PoseResult(True,1.,[[.5,.5,0,1.]]*31,[[0,0,0,1.]]*31),(1280,720))
        result=fusion.update([observation],cv2)
        self.assertTrue(result.safety_paused)
        self.assertFalse(result.pose.detected)
        self.assertIn('aspect ratio',result.calibration_hint)

    def test_profile_roundtrip_and_corner_change_preserve_measured_lens(self):
        import Lib.Config as config
        setup=replace(self.setup,source_id='local:0')
        with tempfile.TemporaryDirectory() as directory,patch.object(config,'PROFILES_DIR',Path(directory)):
            save_profile(Profile(camera_setups=(setup,)))
            restored=load_profile('Default').camera_setups[0]
            self.assertEqual(restored.lens_intrinsics,setup.lens_intrinsics)
            self.assertEqual(restored.lens_distortion,setup.lens_distortion)
            self.assertEqual(camera_setup_for_corner(restored,'Front left (-X, -Z)',1.,(4,3,4)).lens_intrinsics,setup.lens_intrinsics)

    def test_bad_lens_is_rejected(self):
        for intrinsics,distortion in (((960,720,0,800,480,360),(0,)*5),((960,720,800,800,480,360),(float('nan'),)*5),((),(0,)*5)):
            with self.assertRaises(ValueError):validate_lens(intrinsics,distortion)

    def test_calibration_recovers_known_lens_and_rejects_repeated_views(self):
        rng=np.random.default_rng(19)
        world=np.zeros((54,3),np.float32);world[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)*.025
        k=np.array([[820.,0,475],[0,825,354],[0,0,1]])
        corners=[]
        for index in range(24):
            rotation=rng.uniform(-.5,.5,3)
            translation=np.array([rng.uniform(-.35,.12),rng.uniform(-.26,.12),rng.uniform(.65,.95)])
            points,_=cv2.projectPoints(world,rotation,translation,k,np.array(self.setup.lens_distortion))
            corners.append(points)
        result=calibrate_lens(corners,(960,720))
        self.assertLess(abs(result['intrinsics'][2]-820),.1)
        self.assertLess(max(result['held_out_rms_px']),.01)
        with self.assertRaises(ValueError):calibrate_lens([corners[0]]*12,(960,720))


class GeometryGuardTests(unittest.TestCase):
    def fixture(self):
        fixture=consensus_fixture.ConsensusTests();fixture.setUp()
        observations=fixture.observations()
        for observation in observations:
            for point in observation.pose.image_landmarks:point[3]=.95
        return fixture,observations

    def test_moved_third_camera_is_excluded_and_latched_until_reset(self):
        fixture,observations=self.fixture();guard=GeometryHealth()
        for point in observations[2].pose.image_landmarks:point[1]+=.15
        for index in range(16):
            timestamp=index/15
            current=[replace(o,captured_at=timestamp) for o in observations]
            selected,warning,paused=guard.filter(current,fixture.projections,fixture.positions,timestamp)
            self.assertFalse(paused)
            self.assertEqual([o.source_id for o in selected],['a','b'])
        self.assertEqual(guard.quarantined,{'c'})
        selected,warning,paused=guard.filter([current[0]],fixture.projections,fixture.positions,2.)
        self.assertTrue(paused)

    def test_two_disagreeing_views_pause_without_guessing_faulty_camera(self):
        fixture,observations=self.fixture();guard=GeometryHealth()
        for point in observations[1].pose.image_landmarks:point[1]+=.15
        selected,warning,paused=guard.filter(observations[:2],fixture.projections,fixture.positions,0.)
        self.assertTrue(paused);self.assertFalse(selected)

    def test_reused_capture_cannot_latch_transient_fault(self):
        fixture,observations=self.fixture();guard=GeometryHealth()
        for point in observations[2].pose.image_landmarks:point[1]+=.15
        observations=[replace(o,captured_at=0.) for o in observations]
        for index in range(100):guard.filter(observations,fixture.projections,fixture.positions,index/30)
        self.assertEqual(guard.quarantined,set())

    def test_correct_views_are_preserved(self):
        fixture,observations=self.fixture()
        selected,warning,paused=GeometryHealth().filter(observations,fixture.projections,fixture.positions,0.)
        self.assertEqual(len(selected),3);self.assertFalse(paused);self.assertEqual(warning,'')
