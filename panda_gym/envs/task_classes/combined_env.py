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

class TableTop:
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
        self.cabinet_loc, self.pour_loc, self.fill_loc = self.reset_sim()

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        # Initialize cabinet
        cabinet_ori = R.from_euler('xyz', [0,0,-90], degrees=True).as_quat()  
        self.CANONICAL_POS = [0,0.5,0]
        if 'cabinet' in self.body_id_mapping:
            self.sim.physics_client.removeBody(self.body_id_mapping['cabinet'])
            self.sim._bodies_idx.pop('cabinet')
        cabinet = self.sim.loadURDF(body_name='cabinet', fileName='cabinet_assets/cabinet/mobility.urdf', basePosition=self.cabinet_loc, baseOrientation=cabinet_ori, globalScaling=0.5)
        self.body_id_mapping['cabinet'] = cabinet

        # Initialize pouring
        if 'cup1' in self.sim._bodies_idx:
            self.sim.set_base_pose('cup1', self.pour_loc, np.zeros(3))
            self.sim.set_base_pose('cup2', self.fill_loc, np.zeros(3))
        else:
            cup1 = self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=self.pour_loc, globalScaling=0.5)
            cup2 = self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=self.fill_loc, globalScaling=0.75)

        ## Make water
        size = 0.005
        n = 6
        x,y,z = self.pour_loc
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

    def reset_sim(self):
        cabinet_loc = np.array([np.random.uniform(-0.2,0.05), np.random.uniform(0.45,0.65), 0])
        pour_cup_loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,-0.2,0.075])
        fill_cup_loc  = np.random.uniform([-0.1,-0.1,0.075], [0.0, 0.0,0.075])
        if random.random() < 0.5:
            pour_cup_loc, fill_cup_loc = fill_cup_loc, pour_cup_loc
        return cabinet_loc, pour_cup_loc, fill_cup_loc

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

    def pix2point_neighborhood(self, img, waypoint_proj, pixels_2d, points):
        height, width, _ = img.shape
        
        img_masked = np.zeros((height,width)).astype(np.uint8)
        noise = np.random.randint(-15,15,2)
        waypoint_proj += noise

        cv2.circle(img_masked, tuple(waypoint_proj), 10, (255,255,255), -1)
        img_masked_vis = np.repeat(img_masked[:, :, np.newaxis], 3, axis=2)
        #cv2.imshow('img', np.hstack((img, img_masked_vis)))
        #cv2.waitKey(0)
        ys, xs = np.where(img_masked > 0)

        masked_2d = np.vstack((xs, ys)).T.astype(np.uint8)
        pixels_2d = pixels_2d.astype(np.uint8)

        idxs = np.in1d(pixels_2d, masked_2d).reshape(pixels_2d.shape)
        idxs = np.all(idxs, axis=1).squeeze()
        idxs = np.where(idxs == True)[0]

        return points[idxs], idxs

    def point2point_neighborhood(self, source_points, target_points):
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(source_points)
        distances, idxs = nbrs.kneighbors(target_points)
        #idxs_thresh = np.where(distances < 5e-4)[0] 
        #return idxs[idxs_thresh]
        return idxs

    def take_rgbd(self, waypoints):
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        img, depth, points, colors, pixels_2d, waypoints_proj = self.robot.sim.render(distance=1.2, yaw=45, pitch=-30, waypoints=waypoints)
        #img, depth, points, colors, pixels_2d, waypoints_proj = self.robot.sim.render(distance=0.8, yaw=90, pitch=-85, waypoints=waypoints)

        idxs = np.where(points[:,2] < 0.3)[0]
        points = points[idxs]
        colors = colors[idxs]
        pixels_2d = pixels_2d[idxs]

        H,W,_ = img.shape
        points_start, idxs_start = self.pix2point_neighborhood(img, waypoints_proj[0], pixels_2d.copy(), points.copy())

        #points = points[idxs_start]
        #colors = colors[idxs_start]

        _, _, points1, colors1, pixels1_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=0)
        _, _, points2, colors2, pixels_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=90)

        points = np.vstack((points1, points2))
        colors = np.vstack((colors1, points2))

        start_idxs = self.point2point_neighborhood(points, points_start)
        #end_idxs = self.point2point_neighborhood(points, points_end)
        #colors[start_idxs] = (0,0,255)

        keypoint_cond = np.zeros(len(points))
        keypoint_cond[start_idxs] = 1.0

        return img, points, colors, waypoints_proj, keypoint_cond

    def execute(self, episode_idx):
        TOP_LEFT_POS = np.array([-0.1,0,0.2])
        TOP_RIGHT_POS = np.array([0.1,0,0.2])
        BOTTOM_POS = np.array([0.0,0,0.1])
        approach_pos = random.choice((TOP_LEFT_POS, TOP_RIGHT_POS, BOTTOM_POS)) 
        #approach_pos = TOP_RIGHT_POS
        #approach_pos = TOP_LEFT_POS
        #approach_pos = BOTTOM_POS
        approach_pos += self.cabinet_loc - self.CANONICAL_POS
        grasp_euler = np.array([-90,0,0])
        grasp_pos = approach_pos.copy()
        grasp_pos[1] += 0.36

        waypoints = [grasp_pos, approach_pos]

        img, pcl_points, pcl_colors, pixels, pcl_kpt_cond = self.take_rgbd(waypoints)

        #print(pcl_points.shape, pcl_kpt_cond.shape)

        start_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat() 
        orientations = [start_ori, end_ori]

        #### Open cabinet
        self.reset_robot()
        self.robot.move(approach_pos, grasp_euler)
        for i in range(50):
            self.sim.step()
        self.robot.move(grasp_pos, grasp_euler)
        for i in range(50):
            self.sim.step()
        self.robot.grasp()

        offset = approach_pos - grasp_pos
        num_steps = 35
        delta = offset/num_steps

        for i in range(num_steps):
            self.robot.move(grasp_pos + i*delta, grasp_euler)

        self.robot.release()
        for i in range(num_steps):
            self.robot.move(approach_pos - i*delta, grasp_euler)
            if i == 15:
                self.robot.grasp()

        for i in range(50):
            self.sim.step()

        self.record(img, pcl_points, pcl_colors, pcl_kpt_cond, waypoints, orientations, pixels, episode_idx, visualize=False)
        return waypoints, pixels

    def record(self, img, points, colors, kpt_cond, waypoints, orientations, pixels, episode_idx, visualize=True):

        start, end = waypoints
        start_ori, end_ori = orientations

        # Subsample points
        idxs = np.random.choice(len(points), min(5000, len(points)))
        points = points[idxs]
        colors = colors[idxs]
        kpt_cond = kpt_cond[idxs]

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
        data = {'xyz':points, 'xyz_color':colors, 'xyz_kpt':kpt_cond, 'start_waypoint':start, 'end_waypoint':end, 'cls':cls, 'start_ori':start_ori, 'end_ori':end_ori}
        np.save('dset/%d.npy'%episode_idx, data)

        if visualize:
            offsets_vis = colors.copy()
            distances_vis = np.ones((3, len(points)))
            distances = np.linalg.norm(offsets, axis=1)
            distances_normalized = (distances - np.amin(distances))/(np.amax(distances) - np.amin(distances))
            distances_vis[1] = distances_normalized
            distances_vis[2] = distances_normalized
            distances_vis = distances_vis.T
            offsets_vis[indices] = (0,0,0)
            offsets_vis += (distances_vis*255).astype(np.uint8)
    
            cls_vis = (np.vstack((cls, cls, cls)).T)*100

            pcd = o3d.geometry.PointCloud()
            rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
            rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
            points = (rot@points.T).T
    
            pcd.points = o3d.utility.Vector3dVector(points)
            #pcd.colors = o3d.utility.Vector3dVector(cls_vis/255.)
            #pcd.colors = o3d.utility.Vector3dVector(offsets_vis/255.)
            pcd.colors = o3d.utility.Vector3dVector(colors/255.)

            o3d.visualization.draw_geometries([pcd])

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

    task = TableTop(sim, robot)
    task.reset_robot()
    start = time.time()
    #for i in range(40):
    for i in range(10):
        print(i)
        task.reset()
        task.execute(i)
