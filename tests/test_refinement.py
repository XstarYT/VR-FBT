import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import zipfile
import cv2
import numpy as np
from Lib.Refinement import DWPoseRefiner, JOINT_MAPPING, verify_model
from Lib.Tracking import PoseResult


class RefinementTests(unittest.TestCase):
    def pose(self):
        image=[[.3+(i%4)*.1,.2+(i//4)*.08,0.,.9] for i in range(31)]
        world=[[i*.01,0.,0.,.9] for i in range(31)]
        return PoseResult(True,.9,image,world)

    def refiner(self):
        refiner=DWPoseRefiner.__new__(DWPoseRefiner)
        refiner.cv2=cv2;refiner.network=Mock();refiner.output_names=['x','y']
        x=np.zeros((1,133,384),np.float32);y=np.zeros((1,133,512),np.float32)
        x[:,:,192]=.8;y[:,:,256]=.6
        refiner.network.forward.return_value=[x,y]
        return refiner

    def test_refinement_preserves_input_and_world_pose(self):
        pose=self.pose();original=[point.copy() for point in pose.image_landmarks]
        result=self.refiner().refine(np.zeros((720,960,3),np.uint8),pose)
        self.assertIs(result.world_landmarks,pose.world_landmarks)
        self.assertEqual(pose.image_landmarks,original)
        self.assertEqual(result.image_landmarks[1],pose.image_landmarks[1])
        for joint in JOINT_MAPPING:
            self.assertAlmostEqual(result.image_landmarks[joint][3],.7,places=5)
            self.assertTrue(np.isfinite(result.image_landmarks[joint]).all())

    def test_missing_person_does_not_invoke_refiner(self):
        pose=PoseResult(False,0.,[],[]);refiner=self.refiner()
        self.assertIs(refiner.refine(np.zeros((10,10,3),np.uint8),pose),pose)
        refiner.network.forward.assert_not_called()

    def test_bad_native_output_is_reported_instead_of_used(self):
        refiner=self.refiner();refiner.network.forward.return_value=[np.zeros((1,17,384)),np.zeros((1,17,512))]
        with self.assertRaisesRegex(RuntimeError,'shape'):
            refiner.refine(np.zeros((720,960,3),np.uint8),self.pose())
        refiner=self.refiner();refiner.network.forward.return_value[0][0,0,0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'non-finite'):
            refiner.refine(np.zeros((720,960,3),np.uint8),self.pose())

    def test_missing_or_modified_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.onnx'
            with self.assertRaises(FileNotFoundError):verify_model(path)
            path.write_bytes(b'not the pinned weights')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_model(path)

    def test_failed_model_download_keeps_existing_weights(self):
        from scripts import install_refinement
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.onnx';path.write_bytes(b'previous file')
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,'w') as bundle:bundle.writestr('model/end2end.onnx',b'wrong weights')
            with patch.object(install_refinement,'MODEL_PATH',path), patch.object(install_refinement.urllib.request,'urlopen',return_value=io.BytesIO(archive.getvalue())):
                with self.assertRaisesRegex(ValueError,'checksum'):install_refinement.install()
            self.assertEqual(path.read_bytes(),b'previous file')
            self.assertEqual(list(Path(folder).iterdir()),[path])
