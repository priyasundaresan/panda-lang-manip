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

class Open:
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ) -> None:
        self.robot = robot
        self.sim = sim
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
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
        #self.reset_robot()

    def take_rgbd(self):
        img, fmat = self.robot.sim.render(mode='rgb_array', distance=0.85)
        #img, fmat = self.robot.sim.render(mode='rgb_array', distance=0.6, target_position=[0,0,0.1], yaw=90)
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

    def execute(self, episode_idx):
        TOP_LEFT_POS = np.array([-0.1,0,0.2])
        TOP_RIGHT_POS = np.array([0.1,0,0.2])
        BOTTOM_POS = np.array([0.0,0,0.1])
        #approach_pos = random.choice((TOP_LEFT_POS, TOP_RIGHT_POS, BOTTOM_POS)) 
        #approach_pos = TOP_RIGHT_POS
        #approach_pos = TOP_LEFT_POS
        approach_pos = BOTTOM_POS
        approach_pos += self.loc - self.CANONICAL_POS
        grasp_euler = np.array([-90,0,0])
        grasp_pos = approach_pos.copy()
        grasp_pos[1] = 0.36

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        waypoints = [grasp_pos, approach_pos]
        start_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat() 
        orientations = [start_ori, end_ori]

        pixels = self.project_waypoints(waypoints, fmat)

        #### Grasp cup
        #self.robot.move(approach_pos, grasp_euler)
        #for i in range(50):
        #    self.sim.step()
        #self.robot.move(grasp_pos, grasp_euler)
        #for i in range(50):
        #    self.sim.step()
        #self.robot.grasp()

        #offset = approach_pos - grasp_pos
        #num_steps = 35
        #delta = offset/num_steps

        #for i in range(num_steps):
        #    self.robot.move(grasp_pos + i*delta, grasp_euler)

        #for i in range(50):
        #    self.sim.step()

        #self.record(img, pcl_points, pcl_colors, waypoints, orientations, pixels, episode_idx, visualize=True)
        self.record(img, pcl_points, pcl_colors, waypoints, orientations, pixels, episode_idx, visualize=False)
        return waypoints, pixels

    def record(self, img, points, colors, waypoints, orientations, pixels, episode_idx, visualize=True):

        start, end = waypoints
        start_ori, end_ori = orientations

        # Subsample points
        idxs = np.random.choice(len(points), min(5000, len(points)))
        points = points[idxs]
        colors = colors[idxs]

        # Set up offsets 
        offsets = np.zeros_like(points)
        nbrs = NearestNeighbors(n_neighbors=800, algorithm='ball_tree').fit(points)

        cls = np.zeros(len(points))

        distances, indices = nbrs.kneighbors(start.reshape(1,-1))
        offsets[indices] = points[indices] - start
        cls[indices] = 1.0
        distances, indices = nbrs.kneighbors(end.reshape(1,-1))
        offsets[indices] = points[indices] - end
        cls[indices] = 2.0

        # Save points, colors, offsets
        data = {'xyz':points, 'xyz_color':colors, 'start_waypoint':start, 'end_waypoint':end, 'cls':cls, 'start_ori':start_ori, 'end_ori':end_ori}
        np.save('dset/%d.npy'%episode_idx, data)

        if visualize:
            #offsets_vis = colors.copy()
            #distances_vis = np.ones((3, len(points)))
            #distances = np.linalg.norm(offsets, axis=1)
            #distances_normalized = (distances - np.amin(distances))/(np.amax(distances) - np.amin(distances))
            #distances_vis[1] = distances_normalized
            #distances_vis[2] = distances_normalized
            #distances_vis = distances_vis.T
            #offsets_vis[indices] = (0,0,0)
            #offsets_vis += (distances_vis*255).astype(np.uint8)
    
            #cls_vis = (np.vstack((cls, cls, cls)).T)*100

            #pcd = o3d.geometry.PointCloud()
            #rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
            #rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
            #points = (rot@points.T).T
    
            #pcd.points = o3d.utility.Vector3dVector(points)
            ##pcd.colors = o3d.utility.Vector3dVector(cls_vis/255.)
            #pcd.colors = o3d.utility.Vector3dVector(offsets_vis/255.)

            #o3d.visualization.draw_geometries([pcd])

            for pixel in pixels:
                cv2.circle(img, tuple(pixel), 4, (255,0,0), -1)
            cv2.imwrite('images/%05d.jpg'%episode_idx, img)

            cv2.imshow('img', img)
            cv2.waitKey(0)

if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    #sim = PyBullet(render=False, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")

    if not os.path.exists('dset'):
        os.mkdir('dset')
    if not os.path.exists('images'):
        os.mkdir('images')

    task = Open(sim, robot)
    task.reset_robot()
    start = time.time()
    #for i in range(100):
    for i in range(20):
        print(i)
        task.reset()
        task.execute(i)
    #end = time.time()
