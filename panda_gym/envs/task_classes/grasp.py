from typing import Any, Dict
import cv2
import os
import time
import open3d as o3d

import numpy as np

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
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
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
        self.object_position = object_position
        #self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", self.goal, object_position)
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

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return True

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        return 0

    def record(self, robot):
        img, fmat = robot.sim.render(mode='rgb_array', distance=1.2)
        #img, fmat = robot.sim.render(mode='rgb_array', distance=0.5, yaw=270)
        H,W,C = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        position = robot.get_ee_position()
        #position = self.goal
        print('POS', position)
        pixel = (fmat @ np.hstack((position, [1])))
        print(pixel)
        
        x,y,z,w = pixel
        #x /= w
        #y /= w
        print(x,y)
        x += 1
        y += 1
        print(x,y)
        x /= 2
        y /= 2
        print(x,y)
        #print(x,y)

        x *= 480
        y *= 480
        y = 480 - y
    
        pixel = (int(x), int(y))

        #pixel = (fmat @ np.hstack((position, [1])))[:3]*np.array([480, 480, 0]) + np.array([240, 240, 0])
        #pixel = pixel[:-1].astype(int)
        #pixel[1] = 480 - pixel[1]
        #print(pixel)

        cv2.circle(img, tuple(pixel), 4, (255,255,0), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)

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

    sim = PyBullet(render=True)
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = Manipulate(sim)
    robot.reset()
    thresh = 5e-3

    pos = task.reset()

    goal_euler_xyz = np.array([180,0,0]) # standard
    robot.reset()
    robot.move(np.array([0,0,0.6]), goal_euler_xyz)
    robot.release()

    #robot.move(pos, goal_euler_xyz)
    #robot.grasp()
    #robot.move(pos + np.array([0,0,0.15]), goal_euler_xyz)

    #for i in range(100):
    #    robot.sim.step()

    task.record(robot)

    img, points, colors = robot.sim.render(mode='depth', distance=0.6, target_position=[0,0,0.1], yaw=90)

    #pcd = o3d.geometry.PointCloud()
    #rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
    #rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
    #points = (rot@points.T).T

    data = {'xyz':points, 'xyz_color':colors}
    np.save('0.npy', data)

    visualize(img, points, colors)
