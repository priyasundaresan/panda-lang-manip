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

class Pour:
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ) -> None:
        self.robot = robot
        self.sim = sim
        #self.reset_info = self._create_scene()
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        loc1, loc2 = self.reset_sim()

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        #if loc1 is None:
        #    loc1 = [0,-0.2,0.075]
        #if loc2 is None:
        #    loc2 = [0,-0.1,0.075]

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
        
    def parameterized_pour(self, episode_idx):
        #self.robot.reset()
        #self.robot.release()
        self.reset_robot()

        pos, final_pos = self.reset_info

        grasp_pos = pos - np.array([0,0,0.045])
        pour_pos = final_pos + np.array([0,-0.05,0.15])

        img, pcl_points, pcl_colors, fmat = self.take_rgbd()
        #waypoints = [pos, final_pos, grasp_pos, pour_pos]
        waypoints = [pos, final_pos]

        ## Grasp cup
        #goal_euler_xyz = np.array([180,-35,90]) # standard
        #self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)
        #self.robot.move(pos - np.array([0,0,0.045]), goal_euler_xyz)
        #self.robot.grasp()

        #waypoints.append(self.robot.get_ee_position())

        ## Lift
        #self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

        ## Pour into other cup
        #self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)
        #goal_euler_xyz = np.array([180,85,90])
        #self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)

        #waypoints.append(self.robot.get_ee_position())
        ##img, pcl_points, pcl_colors, fmat = self.take_rgbd()

        pixels = self.project_waypoints(waypoints, fmat)

        self.visualize(img, pcl_points, pcl_colors, pixels)

        # Wait for pour to be done
        for i in range(50):
            self.sim.step()

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

    def visualize(self, img, points, colors, pixels):
        #pcd = o3d.geometry.PointCloud()
    
        #rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
        #rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
        #points = (rot@points.T).T
    
        #pcd.points = o3d.utility.Vector3dVector(points)
        #pcd.colors = o3d.utility.Vector3dVector(colors/255.)
        #o3d.visualization.draw_geometries([pcd])

        for pixel in pixels:
            cv2.circle(img, tuple(pixel), 4, (255,0,0), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    if not os.path.exists('images'):
        os.mkdir('images')

    start = time.time()
    for i in range(4):
        task = Pour(sim, robot)
        task.parameterized_pour(i*8)
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
