import unittest
import numpy as np
from Lib.Config import CameraSetup
from Lib.Tracking import CameraObservation, MultiCameraPoseFusion, PoseResult
from Lib.Triangulation import triangulate_consensus


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.setups=(CameraSetup('a',(-2,1.4,-3),(33.69,0,0)),
                     CameraSetup('b',(2,1.4,-3),(-33.69,0,0)),
                     CameraSetup('c',(0,1.4,-3.5),(0,0,0)))
        self.fusion=MultiCameraPoseFusion(('a','b','c'),manual_camera_setup=True,camera_setups=self.setups)
        self.projections={};self.positions={}
        for setup in self.setups:
            rotation,translation,_,position=self.fusion._manual_camera_geometry(setup,np)
            self.projections[setup.source_id]=self.fusion._camera_matrix((960,720),np,setup.source_id)@np.column_stack((rotation,translation))
            self.positions[setup.source_id]=position

    def views(self,point):
        result=[]
        for source,projection in self.projections.items():
            p=projection@np.append(point,1.)
            result.append((source,p[0]/p[2],p[1]/p[2],.95))
        return result

    def test_third_camera_outlier_does_not_pull_agreeing_views(self):
        expected=np.array((.15,1.1,.2));views=self.views(expected)
        source,u,v,_=views[2];views[2]=(source,u+180,v+90,.99)
        result=triangulate_consensus(views,self.projections,self.positions)
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result[0]-expected),.001)
        self.assertGreater(result[1],.9)

    def test_duplicate_view_has_no_depth_baseline(self):
        views=self.views((.1,1.,0))[:1]
        views.append(('duplicate',*views[0][1:]))
        projections={**self.projections,'duplicate':self.projections['a']}
        positions={**self.positions,'duplicate':self.positions['a']}
        self.assertIsNone(triangulate_consensus(views,projections,positions))

    def observations(self,arms_down=False):
        body=np.tile((0.,1.6,0.),(31,1))
        for index,value in {11:(-.2,1.4,0),12:(.2,1.4,0),13:(-.5,1.4,0),14:(.5,1.4,0),
                            15:(-.8,1.4,0),16:(.8,1.4,0),23:(-.15,.9,0),24:(.15,.9,0),
                            25:(-.15,.5,0),26:(.15,.5,0),27:(-.15,.1,0),28:(.15,.1,0)}.items():body[index]=value
        if arms_down:
            body[13]=(-.25,1.1,0);body[15]=(-.3,.8,0)
            body[14]=(.25,1.1,0);body[16]=(.3,.8,0)
        result=[]
        for setup in self.setups:
            image=[];world=[]
            for index,point in enumerate(body):
                visible=.1 if setup.source_id=='c' and index in {14,16,26,28} else .95
                p=self.projections[setup.source_id]@np.append(point,1.)
                image.append([p[0]/p[2]/959,p[1]/p[2]/719,0.,visible])
                world.append([*point,visible])
            result.append(CameraObservation(setup.source_id,PoseResult(True,.95,image,world),(960,720)))
        return result

    def test_occluded_limbs_can_be_verified_by_other_known_views(self):
        observations=self.observations()
        self.assertFalse(self.fusion._is_t_pose(observations[2]))
        self.assertTrue(self.fusion._is_multiview_t_pose(observations))
        self.assertFalse(self.fusion._is_multiview_t_pose(self.observations(arms_down=True)))
        unknown=MultiCameraPoseFusion(('a','b','c'))
        self.assertFalse(unknown._is_multiview_t_pose(observations))

    def test_every_joint_still_requires_two_visible_rays(self):
        observations=self.observations()
        observations[0].pose.image_landmarks[16][3]=.1
        self.assertFalse(self.fusion._is_multiview_t_pose(observations))
