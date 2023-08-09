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

class PixelSelector:
    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img

    def mouse_callback(self, event, x, y, flags, param):
        print(self.img)
        print("HERE")
        cv2.imshow("pixel_selector", self.img)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            print(x, y)
            cv2.circle(self.img, (x, y), 3, (255, 255, 0), -1)

    def run(self, img):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        # print("HI", self.img)
        while True:
            # cv2.imshow('pixel_selector',self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        print(self.clicks)
        return self.clicks

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
        self.inference_server = Inference(checkpoint_path='log_cabinet_kpt_cond/part_seg/2023-03-02_17-12', inp_dim=7)
        #self.inference_server = Inference(checkpoint_path='log_cabinet_kpt_cond_rand/part_seg/2023-03-02_19-58', inp_dim=7)
        self.pixel_selector = PixelSelector()

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

    def pix2point_neighborhood(self, img, waypoint_proj, pixels_2d, points):
        height, width, _ = img.shape
        
        img_masked = np.zeros((height,width)).astype(np.uint8)
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
        #img, depth, points, colors, pixels_2d, waypoints_proj = self.robot.sim.render(distance=1.2, yaw=45, pitch=-30, waypoints=waypoints)
        img, depth, points, colors, pixels_2d, _ = self.robot.sim.render(distance=1.2, yaw=45, pitch=-30)
        waypoints_proj = self.pixel_selector.run(img)

        idxs = np.where(points[:,2] < 0.3)[0]
        points = points[idxs]
        colors = colors[idxs]
        pixels_2d = pixels_2d[idxs]

        H,W,_ = img.shape
        points_start, idxs_start = self.pix2point_neighborhood(img, waypoints_proj[0], pixels_2d.copy(), points.copy())

        _, _, points1, colors1, pixels1_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=0)
        _, _, points2, colors2, pixels_2d, _ = self.robot.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=90)

        points = np.vstack((points1, points2))
        colors = np.vstack((colors1, points2))

        start_idxs = self.point2point_neighborhood(points, points_start)
        keypoint_cond = np.zeros(len(points))
        keypoint_cond[start_idxs] = 1.0

        return img, points, colors, waypoints_proj, keypoint_cond

    def execute(self, episode_idx):
        TOP_LEFT_POS = np.array([-0.1,0,0.2])
        TOP_RIGHT_POS = np.array([0.1,0,0.2])
        BOTTOM_POS = np.array([0.0,0,0.1])
        approach_pos = random.choice((TOP_LEFT_POS, TOP_RIGHT_POS, BOTTOM_POS)) 
        approach_pos += self.loc - self.CANONICAL_POS
        grasp_euler = np.array([-90,0,0])
        grasp_pos = approach_pos.copy()
        grasp_pos[1] += 0.36

        waypoints = [grasp_pos, approach_pos]

        img, pcl_points, pcl_colors, pixels, pcl_kpt_cond = self.take_rgbd(waypoints)

        for pixel in pixels:
            cv2.circle(img, tuple(pixel), 4, (255,0,0), -1)

        #cv2.imshow('img', img)
        #cv2.waitKey(0)

        idxs = np.random.choice(len(pcl_points), 5000)
        pcl_points = pcl_points[idxs]
        pcl_colors = pcl_colors[idxs]
        pcl_kpt_cond = pcl_kpt_cond[idxs]

        inp = np.vstack((pcl_points.T, pcl_colors.T, pcl_kpt_cond.T)).T
        grasp_pos, approach_pos, grasp_euler, _ = self.inference_server.predict(inp)

        start_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat() 
        orientations = [start_ori, end_ori]

        ##### Open cabinet
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

        for i in range(50):
            self.sim.step()

        #self.record(img, pcl_points, pcl_colors, pcl_kpt_cond, waypoints, orientations, pixels, episode_idx, visualize=False)

if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")

    if not os.path.exists('preds'):
        os.mkdir('preds')

    task = Open(sim, robot)
    task.reset_robot()
    start = time.time()
    for i in range(10):
        task.reset()
        task.execute(i)
        task.reset_robot()
    end = time.time()

    print(end-start)