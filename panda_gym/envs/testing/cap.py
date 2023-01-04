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
from panda_gym.envs.robots.panda_ori import Panda

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
        #with self.sim.no_rendering():
        #    self._create_scene()
        #    self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=[0,-0.2,0.075], globalScaling=0.5)
        self.sim.loadURDF(body_name='cap1', fileName='assets/cap.urdf', basePosition=[0.0,0.0,0.075], globalScaling=0.5)

        # Create water
        #size = 0.005
        size = 0.005
        n = 4
        #n = 6
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ob = self.sim.loadURDF(body_name='droplet%d'%(i*j*k), fileName="sphere_1cm.urdf", basePosition=[size*(i-n//2), -0.2 + size*(j-n//2), 0.1 + size*(k-n//2)], useMaximalCoordinates=True, globalScaling=0.7)
                    self.sim.physics_client.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.5])

        #self.sim.physics_client.setRealTimeSimulation(1)

        #self.sim.create_box(
        #    body_name="object",
        #    half_extents=np.ones(3) * self.object_size / 2,
        #    mass=1.0,
        #    position=np.array([0.0, 0.0, self.object_size / 2]),
        #    rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        #)

    def get_obs(self) -> np.ndarray:
        return np.zeros(3)

    def get_achieved_goal(self) -> np.ndarray:
        return np.zeros(3)

    def reset(self) -> None:
        return self.sim.get_base_position('cap1'), self.sim.get_base_position('cup1')

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
    thresh = 5e-3

    time.sleep(10)

    euler_xyz = np.array([-180,0,0])
    pos, final_pos = task.reset()

    pos -= np.array([0,0,0.02])
    da = pos - robot.get_ee_position()
    ctr = 0
    while np.linalg.norm(da) > thresh and ctr < 150:
        da = pos - robot.get_ee_position()
        action = da.tolist() + [0.5]
        robot.set_action(action, euler_xyz)
        robot.sim.step()
        ctr += 1

    pos -= np.array([0,0,0.05])
    ctr = 0
    da = pos - robot.get_ee_position()
    while np.linalg.norm(da) > thresh and ctr < 100:
        da = pos - robot.get_ee_position()
        action = da.tolist() + [0.5]
        robot.set_action(action, euler_xyz)
        robot.sim.step()
        ctr += 1

    print('here')
    # Grasp
    robot.block_gripper = True
    for i in range(10):
        robot.set_action(action, euler_xyz)
        robot.sim.step()

    ctr = 0
    pos = final_pos + np.array([0,0,0.05])
    da = pos - robot.get_ee_position()
    while np.linalg.norm(da) > thresh and ctr < 100:
        da = pos - robot.get_ee_position()
        action = da.tolist() + [0.5]
        robot.set_action(action, euler_xyz)
        robot.sim.step()
        ctr += 1

    ctr = 0
    pos = final_pos + np.array([0,0,0.02])
    da = pos - robot.get_ee_position()
    while np.linalg.norm(da) > thresh and ctr < 100:
        da = pos - robot.get_ee_position()
        action = da.tolist() + [0.5]
        robot.set_action(action, euler_xyz)
        robot.sim.step()
        ctr += 1

    robot.block_gripper = False
    for i in range(100):
        robot.set_action(action, euler_xyz)
        robot.sim.step()
