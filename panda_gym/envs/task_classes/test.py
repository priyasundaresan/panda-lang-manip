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
        self.obj_to_id_mapping = {}

    def _create_scene(self) -> None:
        """Create the scene."""
        loc1 = self.reset_sim()

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        for key in self.obj_to_id_mapping:
            self.sim.physics_client.removeBody(self.obj_to_id_mapping[key])
            self.sim._bodies_idx.pop(key)
        
        filename = random.choice(os.listdir('cup_assets'))
        #filename = 'cup'
        #cup1 = self.sim.loadURDF(body_name='cup1', fileName='cup_assets/%s/model.urdf'%filename, basePosition=loc1, globalScaling=0.8)
        cup1 = self.sim.loadURDF(body_name='cup1', fileName='cup_assets/%s/model.urdf'%filename, basePosition=loc1, globalScaling=np.random.uniform(0.4,0.9))

        #self.obj_to_id_mapping['cup1'] = cup1

        self.reset_info = self.sim.get_base_position('cup1')
        return self.reset_info

    def reset_sim(self):
        loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,0.1,0.075])
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
        #self.reset_robot()
        
    def parameterized_pour(self, episode_idx):
        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        self.record(img, pcl_points, pcl_colors, episode_idx, visualize=True)

    def take_rgbd(self):
        img, fmat = self.robot.sim.render(mode='rgb_array', distance=1.2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, points, colors = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
        _, points1, colors1 = self.robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=0)
        points = np.vstack((points, points1))
        colors = np.vstack((colors, colors1))
        return img, points, colors, fmat
    
    def project_waypoints(self, waypoints, fmat):
        pixels = []
        for position in waypoints:
            pixel = (fmat @ np.hstack((position, [1])))
            x,y,z,w = pixel
            x += 1
            y += 1
            x /= 2
            y /= 2
            x *= 480
            y *= 480
            y = 480 - y
            pixel = (int(x), int(y))
            pixels.append(pixel)
        return pixels

    def record(self, img, points, colors, episode_idx, visualize=True):

        # Subsample points
        #idxs = np.random.choice(len(points), min(5000, len(points)))
        #points = points[idxs]
        #colors = colors[idxs]

        # Save points, colors, offsets
        data = {'xyz':points, 'xyz_color':colors}
        np.save('dset/%d.npy'%episode_idx, data)

        if visualize:
            pcd = o3d.geometry.PointCloud()
            rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
            rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
            points = (rot@points.T).T
    
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors/255.)
            o3d.visualization.draw_geometries([pcd])

            cv2.imwrite('images/%05d.jpg'%episode_idx, img)

            #cv2.imshow('img', img)
            #cv2.waitKey(0)
    
if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    #sim = PyBullet(render=False, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")

    if not os.path.exists('dset'):
        os.mkdir('dset')
    if not os.path.exists('images'):
        os.mkdir('images')

    task = Pour(sim, robot)
    task.reset_robot()
    start = time.time()
    for i in range(10):
        print(i)
        task.reset()
        task.parameterized_pour(i)
    end = time.time()
