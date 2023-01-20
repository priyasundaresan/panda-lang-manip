from typing import Any, Dict
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
        #self.reset_info = self._create_scene()
        #with self.sim.no_rendering():
        #    self._create_scene()
        #    self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

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
        cup1_loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,-0.2,0.075])
        cup2_loc  = np.random.uniform([-0.1,-0.1,0.075], [0.0, 0.1,0.075])
        return cup1_loc, cup2_loc

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
        #self.reset_robot()

        pos, final_pos = self.reset_info

        grasp_pos = pos - np.array([0,0,0.045])
        pour_pos = final_pos + np.array([0,-0.05,0.15])

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        #waypoints = [pos, final_pos, grasp_pos, pour_pos]
        waypoints = [grasp_pos, pour_pos]
        pixels = self.project_waypoints(waypoints, fmat)

        ## Grasp cup
        #goal_euler_xyz = np.array([180,-35,90]) # standard
        #self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)
        #self.robot.move(pos - np.array([0,0,0.045]), goal_euler_xyz)
        #self.robot.grasp()

        ## Lift
        #self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

        ## Pour into other cup
        #self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)
        #goal_euler_xyz = np.array([180,85,90])
        #self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)

        # Wait for pour to be done
        #for i in range(50):
        #    self.sim.step()

        self.record(img, pcl_points, pcl_colors, waypoints, pixels, episode_idx, visualize=False)
        #self.record(img, pcl_points, pcl_colors, waypoints, pixels, episode_idx, visualize=True)
        return waypoints, pixels

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

    def record(self, img, points, colors, waypoints, pixels, episode_idx, visualize=True):

        start, end = waypoints

        # Subsample points
        idxs = np.random.choice(len(points), min(5000, len(points)))
        points = points[idxs]
        colors = colors[idxs]

        # Set up offsets 
        offsets = np.zeros_like(points)
        nbrs = NearestNeighbors(n_neighbors=800, algorithm='ball_tree').fit(points)

        #norm = np.linalg.norm(offsets, axis=1)

        cls = np.zeros(len(points))

        distances, indices = nbrs.kneighbors(start.reshape(1,-1))
        offsets[indices] = points[indices] - start
        cls[indices] = 1.0
        distances, indices = nbrs.kneighbors(end.reshape(1,-1))
        offsets[indices] = points[indices] - end
        cls[indices] = 2.0

        #cls[np.where(np.linalg.norm(offsets, axis=1) > 0)] = 1.0

        # Save points, colors, offsets
        data = {'xyz':points, 'xyz_color':colors, 'start_waypoint':start, 'end_waypoint':end, 'cls':cls}
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
            pcd.colors = o3d.utility.Vector3dVector(cls_vis/255.)
            o3d.visualization.draw_geometries([pcd])

            #for pixel in pixels:
            #    cv2.circle(img, tuple(pixel), 4, (255,0,0), -1)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
    
if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")

    #if not os.path.exists('images'):
    #    os.mkdir('images')

    if not os.path.exists('dset'):
        os.mkdir('dset')

    task = Pour(sim, robot)
    task.reset_robot()
    start = time.time()
    for i in range(200):
    #for i in range(50):
        print(i)
        task.reset()
        task.parameterized_pour(i)
    end = time.time()

    print(end-start)

    #img = robot.sim.render(mode='depth')
    #img, pointcloud = robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
    #pcd = o3d.geometry.PointCloud()
    #points = pointcloud[:,:,:3].reshape(-1,3)
    #pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([pcd])


    #img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #img = (img*255).astype(np.uint8)
    #cv2.imwrite('images/%05d_depth.jpg'%0, img)

    ##img = robot.sim.render(mode='rgb_array')
    #img = robot.sim.render(mode='rgb_array', distance=0.6, target_position=[0,0,0.1], yaw=90)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('images/%05d_rgb.jpg'%0, img)
