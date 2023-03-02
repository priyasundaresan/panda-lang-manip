from typing import Any, Dict
import random
import time
import os
import open3d as o3d

import numpy as np
import cv2

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

# NEW
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cartesian import Panda

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from panda_gym.envs.contact_graspnet.contact_graspnet.grasp_inference import CGNInference

class Manipulate:
    def __init__(
        self,
        sim: PyBullet,
        robot,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self._create_scene()
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.inference_server = CGNInference()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        location, ori = self.reset_sim()
        
        distractor_names = random.choice(os.listdir('assets'))
        #self.sim.loadURDF(body_name='body', fileName='assets/%s/model.urdf'%distractor_names, basePosition=location, baseOrientation=ori, globalScaling=0.9)
        success = False
        while not success:
            try:
                self.sim.loadURDF(body_name='body', fileName='assets/%s/model.urdf'%distractor_names, basePosition=location, baseOrientation=ori, globalScaling=0.8)
                success = True
            except:
                success = False

    def reset_sim(self):
        #loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,0.1,0.075])
        loc = np.random.uniform([-0.1,-0.2,0.075], [0.0,0.2,0.075])
        ori = R.from_euler('xyz', [0,0,np.random.uniform(0,180)], degrees=True).as_quat()
        return loc, ori

    def reset_robot(self):
        self.robot.reset()
        goal_euler_xyz = np.array([180,0,0]) # standard
        self.robot.move(np.array([0,0,0.6]), goal_euler_xyz)
        self.robot.release()

    def take_rgbd(self):
        img, fmat = self.robot.sim.render(mode='rgb_array', distance=1.2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, points, colors = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
        _, points1, colors1 = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=0)
        _, points2, colors2 = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=135)
        _, points3, colors3 = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=245)
        points = np.vstack((points, points1, points2, points3))
        colors = np.vstack((colors, colors1, colors2, colors3))
        return img, points, colors, fmat


    def visualize(self, points, colors):
        pcd = o3d.geometry.PointCloud()
    
        rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
        rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
        points = (rot@points.T).T
    
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/255.)
        o3d.visualization.draw_geometries([pcd])

    def execute(self):
        self.reset_robot()
        _, points, colors, _ = self.take_rgbd()
        #self.visualize(points, colors)

        ctr = 0

        grasp_pos, (yaw,pitch,roll) = self.inference_server.run_inference(points, colors)
        
        approach_pos = grasp_pos + np.array([0,0,0.08])

        lift_pos = grasp_pos + np.array([0,0,0.15])

        grasp_euler = np.array([180.,0.,0.]) # standard
        yaw, pitch = 0, 0
            
        grasp_euler += [yaw,pitch,roll]

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%ctr, img)
        ctr += 1

        self.robot.move(approach_pos, grasp_euler)
        for i in range(100):
            self.sim.step()
            if i % 10 == 0:
                img, pcl_points, pcl_colors, fmat = self.take_rgbd()
                cv2.imwrite('preds/%05d.jpg'%ctr, img)
                ctr += 1


        self.robot.move(grasp_pos, grasp_euler)

        for i in range(500):
            self.sim.step()
            if i % 100 == 0:
                img, pcl_points, pcl_colors, fmat = self.take_rgbd()
                cv2.imwrite('preds/%05d.jpg'%ctr, img)
                ctr += 1

        self.robot.grasp()

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%ctr, img)
        ctr += 1

        self.robot.move(lift_pos, grasp_euler)

        for i in range(100):
            self.sim.step()
            if i % 10 == 0:
                img, pcl_points, pcl_colors, fmat = self.take_rgbd()
                cv2.imwrite('preds/%05d.jpg'%ctr, img)
                ctr += 1

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.mkdir('images')
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Manipulate(sim, robot)

    if not os.path.exists('preds'):
        os.mkdir('preds')
    task.execute()
