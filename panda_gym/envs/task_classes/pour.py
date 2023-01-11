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
        self.reset_info = self._create_scene()
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        loc1, loc2 = self.reset_sim()

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        if loc1 is None:
            loc1 = [0,-0.2,0.075]
        if loc2 is None:
            loc2 = [0,-0.1,0.075]

        if 'cup1' in self.sim._bodies_idx:
            self.sim.set_base_pose('cup1', loc1, np.zeros(3))
            self.sim.set_base_pose('cup2', loc2, np.zeros(3))
        else:
            cup1 = self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=loc1, globalScaling=0.5)
            #self.sim.physics_client.changeDynamics(cup1, -1, mass=0.03)  
            cup2 = self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=loc2, globalScaling=0.75)
            #self.sim.physics_client.changeDynamics(cup2, -1, mass=1.0)  

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
                        self.sim.physics_client.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.5])
                        self.sim.physics_client.changeDynamics(ob, -1, mass=0.001)  

        self.reset_info = self.sim.get_base_position('cup1'), self.sim.get_base_position('cup2')
        return self.reset_info

    def reset_sim(self):
        cup1_loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,-0.2,0.075])
        cup2_loc  = np.random.uniform([-0.1,-0.1,0.075], [0.0, 0.1,0.075])
        return cup1_loc, cup2_loc
        
    def parameterized_pour(self, episode_idx):
        self.robot.reset()
        self.robot.release()
        pos, final_pos = self.reset_info

        goal_euler_xyz = np.array([180,-35,90]) # standard

        # Grasp cup
        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%episode_idx, img)

        self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+1), img)

        #self.robot.move(pos - np.array([0,0,0.02]), goal_euler_xyz)
        self.robot.move(pos - np.array([0,0,0.045]), goal_euler_xyz)

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+2), img)

        self.robot.grasp()

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+3), img)

        # Lift
        self.robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+4), img)

        # Pour into other cup
        self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+5), img)

        goal_euler_xyz = np.array([180,85,90])
        self.robot.move(final_pos + np.array([0,-0.05,0.15]), goal_euler_xyz)

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+6), img)
        
        for i in range(50):
            self.sim.step()

        img, fmat = robot.sim.render(mode='rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%05d.jpg'%(episode_idx+7), img)

if __name__ == '__main__':
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    if not os.path.exists('images'):
        os.mkdir('images')

    for i in range(2):
        task = Pour(sim, robot)
        task.parameterized_pour(i*8)

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
