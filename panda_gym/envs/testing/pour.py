from typing import Any, Dict
import time
import os

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

class Manipulate(Task):
    def __init__(
        self,
        sim: PyBullet,
    ) -> None:
        super().__init__(sim)
        self._create_scene()
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=[0,-0.2,0.075], globalScaling=0.5)
        self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=[0,-0.1,0.075], globalScaling=0.75)

        # Create water
        size = 0.005
        n = 4
        #n = 6
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ob = self.sim.loadURDF(body_name='droplet%d'%(i*j*k), fileName="sphere_1cm.urdf",
                            basePosition=[
                            size*(i-n//2),
                            -0.2 + size*(j-n//2),
                            0.1 + size*(k-n//2)],
                            useMaximalCoordinates=True,
                            globalScaling=0.7)
                    self.sim.physics_client.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.5])

    def get_obs(self) -> np.ndarray:
        return np.zeros(3)

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(3)

    def reset(self) -> None:
        return self.sim.get_base_position('cup1'), self.sim.get_base_position('cup2')

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        return np.zeros(3)

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        return np.zeros(3)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return True

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        return 0

if __name__ == '__main__':
    if not os.path.exists('images'):
        os.mkdir('images')

    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Manipulate(sim)
    robot.reset()

    pos, final_pos = task.reset()

    goal_euler_xyz = np.array([180,-35,90]) # standard
    robot.release()
    robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)
    robot.move(pos - np.array([0,0,0.055]), goal_euler_xyz)
    robot.grasp()
    robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

    robot.move(final_pos + np.array([0,0,0.15]), goal_euler_xyz)
    #goal_euler_xyz = np.array([180,30,90]) # standard
    #robot.move(final_pos + np.array([0,0,0.10]), goal_euler_xyz)
    print('done')
    
    
