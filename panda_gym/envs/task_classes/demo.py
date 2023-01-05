from typing import Any, Dict
import time

import numpy as np

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
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        return object_position

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

    def teleop(self, robot):
        task.reset()
        grasped = False
        last = ''
        vel = 1
        offset = 0.02
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
                            vel *= 1.03
                        else:
                            vel = 1
                            offset = 0.02

                        offset *= vel
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


if __name__ == '__main__':
    sim = PyBullet(render=True)
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    robot.reset()
    robot.release()
    task = Teleop(sim)
    task.teleop(robot)

