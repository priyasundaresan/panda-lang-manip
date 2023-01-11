from typing import Any, Dict
import time

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

from pynput import keyboard

class Teleop:
    def __init__(
        self,
        sim: PyBullet,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        #sim.timestep = 1./240.
        self.sim = sim
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        #self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, np.random.uniform(0,90), 1.0]))
        return object_position, self.goal

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        if np.random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = np.random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    
    
    def record(self, robot):
        #img = robot.sim.render(mode='rgb_array', distance=0.6, target_position=[0,0,0.1], yaw=90)
        img, fmat = robot.sim.render(mode='rgb_array')
        H,W,C = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        position = robot.get_ee_position()
        print(position)
        pixel = (fmat @ np.hstack((position, [1])))[:3]*np.array([240, 240, 0]) + np.array([120, 120, 0])
        pixel = pixel[:-1].astype(int)
        pixel[1] = 240 - pixel[1]
        print(pixel)
        cv2.circle(img, tuple(pixel), 4, (255,255,0), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def teleop(self, robot):
        pos, goal = task.reset()
        grasped = False
        last = ''
        vel = 1
        offset = 0.015
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
                            vel *= 1.05
                            if vel > 1.5:
                                vel = 1.
                        else:
                            vel = 1.
                            offset = 0.015

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

        for _ in range(100):
            robot.sim.step()

        self.record(robot)

if __name__ == '__main__':
    sim = PyBullet(render=True)
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    robot.reset()
    robot.release()
    task = Teleop(sim)
    task.teleop(robot)
