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
        loc1, loc2 = self.reset_sim()
        #self._create_scene()
        self._create_scene(loc1, loc2)
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self, loc1=None, loc2=None) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        if loc1 is None:
            loc1 = [0,-0.2,0.075]
        if loc2 is None:
            loc2 = [0,-0.1,0.075]

        self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=loc1, globalScaling=0.5)
        self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=loc2, globalScaling=0.75)

        #self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=[0,-0.2,0.075], globalScaling=0.5)
        #self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=[0,-0.1,0.075], globalScaling=0.75)
        #self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=[0,0.0,0.075], globalScaling=0.75)
        #self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=[0,0.1,0.075], globalScaling=0.75)

        # Create water
        size = 0.005
        n = 4
        #n = 6

        x,y,z = loc1
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ob = self.sim.loadURDF(body_name='droplet%d'%(i*j*k), fileName="sphere_1cm.urdf",
                            basePosition=[
                            x + size*(i-n//2),
                            y + size*(j-n//2),
                            0.1 + size*(k-n//2)],
                            useMaximalCoordinates=True,
                            globalScaling=0.7)

                    #ob = self.sim.loadURDF(body_name='droplet%d'%(i*j*k), fileName="sphere_1cm.urdf",
                    #        basePosition=[
                    #        size*(i-n//2),
                    #        -0.2 + size*(j-n//2),
                    #        0.1 + size*(k-n//2)],
                    #        useMaximalCoordinates=True,
                    #        globalScaling=0.7)
                    self.sim.physics_client.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.5])

    def reset_sim(self):
        cup1_loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,-0.2,0.075])
        cup2_loc  = np.random.uniform([-0.2,-0.1,0.075], [0.0, 0.1,0.075])

        #cup1_loc = [0,-0.2,0.075]
        #cup2_loc = [0,-0.1,0.075]
        #cup2_loc = [-0.2,-0.1,0.075]

        return cup1_loc, cup2_loc
        
    def reset(self) -> None:
        return self.sim.get_base_position('cup1'), self.sim.get_base_position('cup2')

    def parameterized_pour(self):
        self.robot.reset()
        self.robot.release()

        pos, final_pos = self.reset()

        goal_euler_xyz = np.array([180,-35,90]) # standard

        # Grasp cup
        self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)
        #self.robot.move(pos - np.array([0,0,0.0175]), goal_euler_xyz)
        self.robot.move(pos - np.array([0,0,0.02]), goal_euler_xyz)
        self.robot.grasp()

        # Lift
        self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

        # Pour into other cup
        self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)
        #goal_euler_xyz = np.array([180,80,90])
        goal_euler_xyz = np.array([180,85,90])
        self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)
        
        for i in range(50):
            self.sim.step()

if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Pour(sim, robot)
    task.parameterized_pour()

    #img = robot.sim.render(mode='depth')
    img, pointcloud = robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
    pcd = o3d.geometry.PointCloud()
    points = pointcloud[:,:,:3].reshape(-1,3)
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)
    cv2.imwrite('images/%05d_depth.jpg'%0, img)

    #img = robot.sim.render(mode='rgb_array')
    img = robot.sim.render(mode='rgb_array', distance=0.6, target_position=[0,0,0.1], yaw=90)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('images/%05d_rgb.jpg'%0, img)
