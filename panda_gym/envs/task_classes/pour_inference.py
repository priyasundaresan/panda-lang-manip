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
#from panda_gym.envs.inference.inference import Inference
from panda_gym.envs.inference.inference_cls_off_rot import Inference

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial


class Pour:
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ) -> None:
        self.robot = robot
        self.sim = sim
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.inference_server = Inference()

    def _create_scene(self) -> None:
        """Create the scene."""
        loc1, loc2 = self.reset_sim()

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        if 'cup1' in self.sim._bodies_idx:
            self.sim.set_base_pose('cup1', loc1, np.zeros(3))
            self.sim.set_base_pose('cup2', loc2, np.zeros(3))
        else:
            cup1 = self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=loc1, globalScaling=0.5)
            cup2 = self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=loc2, globalScaling=0.75)

        size = 0.005
        n = 6
        x,y,z = loc1
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    body_name = 'droplet%d'%(i*j*k)
                    if body_name in self.sim._bodies_idx:
                        self.sim.set_base_pose(body_name, 
                                [x + size*(i-n//2),
                                y + size*(j-n//2),
                                0.1 + size*(k-n//2)], np.zeros(3))
                    else:
                        ob = self.sim.loadURDF(body_name=body_name, fileName="sphere_1cm.urdf",
                                basePosition=[
                                x + size*(i-n//2),
                                y + size*(j-n//2),
                                0.1 + size*(k-n//2)],
                                useMaximalCoordinates=True,
                                globalScaling=0.7)
                        self.sim.physics_client.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.3])
                        self.sim.physics_client.changeDynamics(ob, -1, mass=0.001)  

        self.reset_info = loc1, loc2
        self.reset_info = self.sim.get_base_position('cup1'), self.sim.get_base_position('cup2')
        return self.reset_info

    def reset_sim(self):
        loc1 = np.random.uniform([-0.2,-0.3,0.075], [0.0,-0.2,0.075])
        loc2  = np.random.uniform([-0.1,-0.1,0.075], [0.0, 0.1,0.075])
        if random.random() < 0.5:
            return loc1, loc2
        return loc2, loc1

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
        
    def parameterized_pour(self, episode_idx):
        _, _ = self.reset_info

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%episode_idx, img)

        idxs = np.random.choice(len(pcl_points), 5000)
        pcl_points = pcl_points[idxs]
        pcl_colors = pcl_colors[idxs]

        inp = np.vstack((pcl_points.T, pcl_colors.T)).T
        start_pos, end_pos, start_euler, end_euler = self.inference_server.predict(inp)

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(episode_idx+1), img)

        ## Grasp cup
        self.robot.move(start_pos + np.array([0,0,0.10]), start_euler)
        self.robot.move(start_pos, start_euler)
        self.robot.grasp()

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(episode_idx+2), img)

        ## Lift
        self.robot.move(start_pos + np.array([0,0,0.10]), start_euler)

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(episode_idx+3), img)

        # Pour
        self.robot.move(end_pos, start_euler)
        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(episode_idx+4), img)

        self.robot.move(end_pos, end_euler)

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(episode_idx+5), img)

        # Wait for pour to be done
        ctr = 0
        for i in range(50):
            if i%5 == 0:
                img, pcl_points, pcl_colors, fmat = self.take_rgbd()
                cv2.imwrite('preds/%05d.jpg'%(episode_idx+6+ctr), img)
                ctr += 1
            self.sim.step()

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        cv2.imwrite('preds/%05d.jpg'%(episode_idx+ctr+1), img)
        return  ctr + 1

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

    task = Pour(sim, robot)
    task.reset_robot()
    start = time.time()
    ctr = 0
    for i in range(10):
        task.reset()
        ctr = task.parameterized_pour(i*ctr)
        task.reset_robot()
    end = time.time()

    print(end-start)
