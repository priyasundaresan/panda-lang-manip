import os
import open3d as o3d
import cv2

import pybullet as p

import time
from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task

# NEW
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cartesian import Panda

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from pynput import keyboard

class Teleop:
    def __init__(
        self,
        sim: PyBullet,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:

        sim.physics_client.setGravity(0,0,-9.8)
        sim.timestep = 1./240.

        self.sim = sim

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=90, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        #cabinet_ori = R.from_euler('xyz', [0,0,90], degrees=True).as_quat()  
        cabinet_ori = R.from_euler('xyz', [0,0,-90], degrees=True).as_quat()  
        #cabinet = self.sim.loadURDF(body_name='cabinet', fileName='cabinet_assets/black_cabinet/mobility.urdf', basePosition=[0,0.6,0], baseOrientation=cabinet_ori, globalScaling=0.5)
        #cabinet = self.sim.loadURDF(body_name='cabinet', fileName='cabinet_assets/cabinet3/mobility.urdf', basePosition=[0,0.5,0], baseOrientation=cabinet_ori, globalScaling=0.5)
        cabinet = self.sim.loadURDF(body_name='cabinet', fileName='cabinet_assets/cabinet/mobility.urdf', basePosition=[0,0.5,0], baseOrientation=cabinet_ori, globalScaling=0.5)
    

        self.object_size = 0.04
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def reset(self) -> None:
        return np.zeros(3), np.zeros(3)

    def teleop(self, robot):
        pos, goal = task.reset()
        grasped = False
        last = ''
        vel = 1
        offset = 0.02
        curr_euler = robot.get_ee_orientation()
        reset_pos = robot.get_ee_position()

        robot.move(pos + np.array([0,0,0.02]), curr_euler)
        while True:
            curr_pos = robot.get_ee_position()
            curr_euler = robot.get_ee_orientation()
            with keyboard.Events() as events:
                event = events.get(10.0)
                if event is None:
                    robot.sim.step()
                else:
                    if 'char' in dir(event.key):
                        if event.key.char == last:
                            last = 0
                            vel *= 1.1
                            if vel > 1.5:
                                vel = 1.
                        else:
                            vel = 1.
                            offset = 0.02

                        new_offset = offset * vel
                        offset = new_offset
                        if event.key.char == 'j':
                            robot.move(curr_pos - np.array([0,0,offset]), curr_euler)
                        elif event.key.char == 'k':
                            robot.move(curr_pos + np.array([0,0,offset]), curr_euler)
                        elif event.key.char == 'o':
                            robot.move(curr_pos + np.array([offset,0,0]), curr_euler)
                        elif event.key.char == 'i':
                            robot.move(curr_pos + np.array([-offset,0,0]), curr_euler)
                        elif event.key.char == 'l':
                            robot.move(curr_pos + np.array([0,offset,0]), curr_euler)
                        elif event.key.char == 'h':
                            robot.move(curr_pos + np.array([0,-offset,0]), curr_euler)
                        elif 'q' in str(event.key):
                            break
                        elif 'r' in str(event.key):
                            curr_euler = np.array([-180,0,0])
                            robot.move(reset_pos, curr_euler)
                        elif 'a' in str(event.key):
                            curr_euler = np.array([-90,0,0])
                            robot.move(curr_pos, curr_euler)
                        elif 's' in str(event.key):
                            curr_euler = np.array([90,0,0])
                            robot.move(curr_pos, curr_euler)
                        elif 'd' in str(event.key):
                            curr_euler = np.array([180,0,90])
                            robot.move(curr_pos, curr_euler)
                        elif 'f' in str(event.key):
                            robot.move(reset_pos, curr_euler)
                        last = event.key.char

                    elif event.key == keyboard.Key.space:
                        if not grasped:
                            robot.grasp()
                            grasped = True
                        else:
                            robot.release()
                            grasped = False

                    elif event.key == keyboard.Key.esc:
                        break
                    elif event.key == keyboard.Key.right:
                        offset = np.array([0,0,10])
                        robot.move(curr_pos, curr_euler + offset)
                    elif event.key == keyboard.Key.left:
                        offset = np.array([0,0,-10])
                        robot.move(curr_pos, curr_euler + offset)
                    elif event.key == keyboard.Key.up:
                        offset = np.array([0,10,0])
                        robot.move(curr_pos, curr_euler + offset)
                    elif event.key == keyboard.Key.down:
                        offset = np.array([0,-10,0])
                        robot.move(curr_pos, curr_euler + offset)

        #robot.move(goal, curr_euler)
        #for _ in range(50):
        #    robot.sim.step()

def visualize(img, points, colors):
    pcd = o3d.geometry.PointCloud()

    rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
    rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
    points = (rot@points.T).T

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.)
    o3d.visualization.draw_geometries([pcd])

    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)
    cv2.imwrite('images/%05d_depth.jpg'%0, img)

    img, _ = robot.sim.render(mode='rgb_array', distance=0.6, target_position=[0,0,0.1], yaw=90)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('images/%05d_rgb.jpg'%0, img)

if __name__ == '__main__':
    sim = PyBullet(render=True)
    task = Teleop(sim)
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    robot.reset()
    robot.release()
    task.teleop(robot)

    #img, points, colors = robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)
    #data = {'xyz':points, 'xyz_color':colors}
    #np.save('0.npy', data)

    #visualize(img, points, colors)
