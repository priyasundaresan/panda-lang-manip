from typing import Any, Dict
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
        self.sim.loadURDF(body_name='plate', fileName='assets/blue_plate/model.urdf', basePosition=[0,-0.1,0.075], globalScaling=0.75)
        self.sim.loadURDF(body_name='banana', fileName='assets/plastic_banana/model.urdf', basePosition=[0,0.1,0.0], globalScaling=0.65)
        self.sim.loadURDF(body_name='apple', fileName='assets/plastic_apple/model.urdf', basePosition=[0,0.2,0.0], globalScaling=0.75)
        self.sim.loadURDF(body_name='plum', fileName='assets/plastic_plum/model.urdf', basePosition=[0,-0.2,0.0], globalScaling=0.75)
        self.sim.loadURDF(body_name='plate_holder', fileName='assets/plate_holder/model.urdf', basePosition=[-0.2,-0.2,0.0], globalScaling=0.75)

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
        return self.sim.get_base_position('banana'), self.sim.get_base_position('plate')

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
    if not os.path.exists('images'):
        os.mkdir('images')

    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Manipulate(sim)
    robot.reset()

    pos, final_pos = task.reset()

    for i in range(100):
        robot.sim.step()
    img, points, colors = robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)

    #pcd = o3d.geometry.PointCloud()

    #rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
    #rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
    #points = (rot@points.T).T

    data = {'xyz':points, 'xyz_color':colors}
    np.save('0.npy', data)

    visualize(img, points, colors)
