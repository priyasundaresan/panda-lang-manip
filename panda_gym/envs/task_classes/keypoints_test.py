from typing import Any, Dict
import random
import time
import os
import open3d as o3d

import numpy as np
import cv2

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

# NEW
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cartesian import Panda

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


class RealSenseD415:
    """Default configuration with 3 RealSense RGB-D cameras."""
    def __init__(self):
        # Mimic RealSense D415 RGB-D camera parameters.
        image_size = (480, 640)
        intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

        # Set default camera poses.
        #front_position = (0.68, 0, 0.75)
        #front_position = (0.5, 0, 0.75)
        front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
        front_rotation = R.from_euler('xyz', front_rotation).as_quat()

        # Default camera configs. (Daniel: setting Noise=True based on Andy advice)
        # Daniel: actually, getting some errors; let's revisit later.
        self.CONFIG = [
            {
                'image_size': image_size,
                'intrinsics': intrinsics,
                'position': front_position,
                'rotation': front_rotation,
                'zrange': (0.01, 10.),
                'noise': False
            },
        ]

class Manipulate:
    def __init__(
        self,
        sim: PyBullet,
        robot,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.object_size = 0.04
        self._create_scene()
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.config = RealSenseD415().CONFIG[0]
        print(self.config)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        location, ori = self.reset_sim()
        
        if 'object' in self.sim._bodies_idx:
            self.sim.set_base_pose('object', location, np.zeros(3))
        else:
            self.sim.create_box(
                body_name="object",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=1.0,
                position=np.array([location[0], location[1], self.object_size / 2]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )

    def reset(self):
        self._create_scene()
        for i in range(10):
            self.sim.step()

    def reset_sim(self):
        loc = np.random.uniform([-0.2,-0.2,0.075], [0.0,0.2,0.075])
        #loc = np.random.uniform([-0.1,-0.2,0.075], [0.0,0.2,0.075])
        #loc = random.choice([[-0.55,0.3,0], [-0.55,-0.3,0], [0.2,0.3,0], [0.2,-0.3,0]])
        #loc = random.choice([[-0.35,0.3,0], [-0.35,-0.3,0], [0.2,0.3,0], [0.2,-0.3,0]])
        ori = R.from_euler('xyz', [0,0,np.random.uniform(0,180)], degrees=True).as_quat()
        return loc, ori

    def reset_robot(self):
        self.robot.reset()
        goal_euler_xyz = np.array([180,0,0]) # standard
        self.robot.move(np.array([0,0,0.6]), goal_euler_xyz)
        self.robot.release()

    def take_rgbd(self):
        img, points, colors = self.robot.sim.render_fixed(self.config)
        img = np.array(img)
        object_position = self.sim.get_base_position("object")

        pixels = self.robot.sim.project3d2pix(self.config, [object_position])
        for pixel in pixels:
            print(pixel)
            cv2.circle(img, tuple(pixel), 4, (255,0,0), -1)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)

        pcd = o3d.geometry.PointCloud()

        #rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
        #rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
        #points = (rot@points.T).T

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/255.)
        o3d.visualization.draw_geometries([pcd])

    def execute(self):
        self.reset_robot()
        for i in range(1000):
            self.robot.sim.step()
        self.take_rgbd()

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.mkdir('images')
    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Manipulate(sim, robot)
    for i in range(10):
        task.reset()
        task.execute()
