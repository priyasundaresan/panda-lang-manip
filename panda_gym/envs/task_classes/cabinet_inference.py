from typing import Any, Dict
import random
import time
import os

import numpy as np
import cv2
import open3d as o3d

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

# NEW
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cartesian import Panda
from panda_gym.envs.inference.inference_cls_off_rot import Inference

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial


class Open:
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ) -> None:
        self.robot = robot
        self.sim = sim
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.inference_server = Inference()
        self.body_id_mapping = {}

    def _create_scene(self) -> None:
        """Create the scene."""
        self.loc = self.reset_sim()

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        cabinet_ori = R.from_euler('xyz', [0,0,-90], degrees=True).as_quat()  

        self.CANONICAL_POS = [0,0.5,0]

        if 'cabinet' in self.body_id_mapping:
            self.sim.physics_client.removeBody(self.body_id_mapping['cabinet'])
            self.sim._bodies_idx.pop('cabinet')
        cabinet = self.sim.loadURDF(body_name='cabinet', fileName='cabinet_assets/cabinet/mobility.urdf', basePosition=self.loc, baseOrientation=cabinet_ori, globalScaling=0.5)
        self.body_id_mapping['cabinet'] = cabinet

    def reset_sim(self):
        loc = np.array([np.random.uniform(-0.2,0.05), np.random.uniform(0.45,0.65), 0])
        return loc

    def reset_robot(self):
        self.robot.reset()
        goal_euler_xyz = np.array([180,0,0]) # standard
        self.robot.move(np.array([0,0,0.6]), goal_euler_xyz)
        self.robot.release()

    def reset(self):
        with self.sim.no_rendering():
            self._create_scene()
        for i in range(10):
            self.sim.step()

    def take_rgbd(self):
        img, fmat = self.robot.sim.render(mode='rgb_array', distance=0.85)
        #img, fmat = self.robot.sim.render(mode='rgb_array', distance=0.6, target_position=[0,0,0.1], yaw=90)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, points, colors = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
        _, points1, colors1 = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=0)

        points = np.vstack((points, points1))
        colors = np.vstack((colors, colors1))
        return img, points, colors, fmat
        
    def execute(self, episode_idx):
        ctr = episode_idx
        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%ctr, img)
        ctr += 1

        idxs = np.random.choice(len(pcl_points), 5000)
        pcl_points = pcl_points[idxs]
        pcl_colors = pcl_colors[idxs]

        inp = np.vstack((pcl_points.T, pcl_colors.T)).T
        success = False
        grasp_pos, approach_pos, grasp_euler, _ = self.inference_server.predict(inp)
        #grasp_pos += [0,0.02,0]
        grasp_pos += [0,0.045,0]

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(ctr), img)
        ctr += 1

        self.robot.move(approach_pos, grasp_euler)
        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(ctr), img)
        ctr += 1

        for i in range(50):
            self.sim.step()

        self.robot.move(grasp_pos, grasp_euler)
        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(ctr), img)
        ctr += 1

        for i in range(50):
            self.sim.step()

        self.robot.grasp()
        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(ctr), img)
        ctr += 1

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(ctr), img)
        ctr += 1

        offset = approach_pos - grasp_pos
        num_steps = 35
        delta = offset/num_steps

        for i in range(num_steps):
            self.robot.move(grasp_pos + i*delta, grasp_euler)
            if i % 5 == 0:
                img, pcl_points, pcl_colors, fmat = self.take_rgbd()
                cv2.imwrite('preds/%05d.jpg'%(ctr), img)
                ctr += 1

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        #cv2.imwrite('preds/%05d.jpg'%(ctr), img)
        #ctr += 1

        for i in range(50):
            self.sim.step()
            #cv2.imwrite('preds/%05d.jpg'%(ctr), img)
            #ctr += 1

        return ctr + 1

    def take_rgbd(self):
        img, fmat = self.robot.sim.render(mode='rgb_array', distance=1.2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, points, colors = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
        _, points1, colors1 = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=0)
        points = np.vstack((points, points1))
        colors = np.vstack((colors, colors1))
        return img, points, colors, fmat
    
if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")

    if not os.path.exists('preds'):
        os.mkdir('preds')

    task = Open(sim, robot)
    task.reset_robot()
    start = time.time()
    ctr = 0
    for i in range(10):
        task.reset()
        ctr = task.execute(i*ctr)
        task.reset_robot()


        #success = False
        #while not success:
        #    try:
        #        task.reset()
        #        ctr = task.execute(i*ctr)
        #        task.reset_robot()
        #        success = True
        #    except:
        #        success = False
    end = time.time()

    print(end-start)
