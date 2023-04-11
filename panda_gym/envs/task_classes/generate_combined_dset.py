from typing import Any, Dict
from collections.abc import Iterable
import random
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

from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial
from sklearn.metrics.pairwise import euclidean_distances

class TableTop:
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ) -> None:
        self.robot = robot
        self.sim = sim
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.body_id_mapping = {}

    def _create_scene(self) -> None:
        """Create the scene."""
        num_objs = np.random.randint(3,5)

        self.cabinet_loc, object_locs = self.reset_sim(num_objs)
        self.pour_loc = object_locs[0]
        self.fill_loc = object_locs[1]
        grasping_locs = object_locs[2:]

        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        # Delete graspable objects
        for k in self.body_id_mapping:
            if 'body' in k:
                self.sim.physics_client.removeBody(self.body_id_mapping[k])
                try:
                    self.sim._bodies_idx.pop(k)
                except:
                    continue

        # Delete cabinet
        if 'cabinet' in self.body_id_mapping:
            self.sim.physics_client.removeBody(self.body_id_mapping['cabinet'])
            self.sim._bodies_idx.pop('cabinet')

        # Initialize cabinet
        cabinet_ori = R.from_euler('xyz', [0,0,-90], degrees=True).as_quat()  
        self.CANONICAL_POS = [0,0.5,0]
        cabinet = self.sim.loadURDF(body_name='cabinet', fileName='cabinet_assets/cabinet/mobility.urdf', basePosition=self.cabinet_loc, baseOrientation=cabinet_ori, globalScaling=0.5, useMaximalCoordinates=False)
        self.body_id_mapping['cabinet'] = cabinet

        # Initialize pouring
        if 'cup1' in self.sim._bodies_idx:
            self.sim.set_base_pose('cup1', self.pour_loc, np.zeros(3))
            self.sim.set_base_pose('cup2', self.fill_loc, np.zeros(3))
        else:
            cup1 = self.sim.loadURDF(body_name='cup1', fileName='assets/cup.urdf', basePosition=self.pour_loc, globalScaling=0.5)
            cup2 = self.sim.loadURDF(body_name='cup2', fileName='assets/cup.urdf', basePosition=self.fill_loc, globalScaling=0.75)

        ## Make water
        size = 0.005
        n = 6
        x,y,z = self.pour_loc
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
                        self.sim.physics_client.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.3])
                        self.sim.physics_client.changeDynamics(ob, -1, mass=0.001)  

        # Put graspable objects in scene
        #success = False

        self.graspable_obj_names = []
        self.graspable_objs = []
        for i in range(len(grasping_locs)):
            fn = random.choice(os.listdir('grasping_assets'))
            self.graspable_obj_names.append(fn)
                
            ori = R.from_euler('xyz', [0,0,np.random.uniform(0,180)], degrees=True).as_quat()
            obj = self.sim.loadURDF(body_name='body%d'%i, fileName='grasping_assets/%s/model.urdf'%fn, basePosition=grasping_locs[i], baseOrientation=ori, globalScaling=0.85)

            self.graspable_objs.append('body%d'%i)
            self.body_id_mapping['body%d'%i] = obj
        self.graspable_obj_names = self.filter_names(self.graspable_obj_names)

    def filter_names(self, fns):
        filtered_names = []
        for fn in fns:
            words = fn.split('_')
            if len(words) == 1:
                filtered_names.append(words[0])
            else:
                if words[-1].isdigit():
                    words = words[:-1]
                if words[0] in ['plastic']:
                    words = words[1:]
                filtered_names.append(' '.join(words))
        return filtered_names

    def dist(self, new_point, points, r_threshold):
        for point in points:
            dist = np.sqrt(np.sum(np.square(new_point-point)))
            if dist < r_threshold:
                return False
        return True
    
    
    def RandX(self,N, r_threshold):
        points = []
        while len(points) < N:
            #new_point = np.random.uniform([-0.3,-0.3,0.075], [0.1,0.1,0.075])
            new_point = np.random.uniform([-0.3,-0.3,0.075], [0.0,0.1,0.075])
            if self.dist(new_point, points, r_threshold):
                points.append(new_point)
        return points

    def reset_sim(self, num_locs=3):
        cabinet_loc = np.array([np.random.uniform(-0.2,0.05), np.random.uniform(0.45,0.65), 0])
        #filtered_points = self.RandX(num_locs, 0.075)
        filtered_points = self.RandX(num_locs, 0.125)

        return cabinet_loc, filtered_points

    #def reset_sim(self):
    #    cabinet_loc = np.array([np.random.uniform(-0.2,0.05), np.random.uniform(0.45,0.65), 0])
    #    pour_cup_loc = np.random.uniform([-0.2,-0.3,0.075], [0.0,-0.2,0.075])
    #    fill_cup_loc  = np.random.uniform([-0.1,-0.1,0.075], [0.0, 0.0,0.075])
    #    if random.random() < 0.5:
    #        pour_cup_loc, fill_cup_loc = fill_cup_loc, pour_cup_loc
    #    return cabinet_loc, pour_cup_loc, fill_cup_loc

    def reset_robot(self):
        self.robot.reset()
        goal_euler_xyz = np.array([180,0,0]) # standard
        self.robot.move(np.array([0,0,0.6]), goal_euler_xyz)
        self.robot.release()

    def reset(self):
        with self.sim.no_rendering():
            self._create_scene()
        self.reset_robot()
        for i in range(10):
            self.sim.step()

    def pix2pix_neighborhood(self, img, waypoint_proj):
        height, width, _ = img.shape

        pixels = []
        for i in range(width):
            for j in range(height):
                pixels.append([i,j])

        pixels = np.array(pixels)

        nbrs = NearestNeighbors(radius=3).fit(pixels)
        dists, idxs = nbrs.radius_neighbors(np.reshape(waypoint_proj, (-1,2)))
        
        pixels = pixels[idxs[0]]

        #img_masked = np.zeros((height,width)).astype(np.uint8)
        #noise = np.random.randint(-15,15,2)
        #waypoint_proj += noise

        #cv2.circle(img_masked, tuple(waypoint_proj), 2, (255,255,255), -1)
        #img_masked_vis = np.repeat(img_masked[:, :, np.newaxis], 3, axis=2)
        #cv2.imshow('img', np.hstack((img, img_masked_vis)))
        #cv2.waitKey(0)
        #ys, xs = np.where(img_masked > 0)

        #masked_2d = np.vstack((xs, ys)).T.astype(np.uint8)
        #return masked_2d
        return pixels


    def point2point_neighborhood(self, source_points, target_points):
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
        dists, idxs  = nbrs.kneighbors(source_points, return_distance=True)
        thresh = 0.1
        idxs = np.where(dists < thresh)[0]
        return idxs

    def filter_outliers(self, points):
        z_low = np.where(points[:,2] > 0.002)
        z_high = np.where(points[:,2] < 0.67)
        idxs = np.intersect1d(z_low, z_high)
        return points[idxs]

    def take_rgbd(self, waypoints):
        self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        
        _, _, cam2world = self.sim.get_cam2world_transforms(target_position=np.array([-0.1,0.1,0]), distance=0.9, yaw=90, pitch=-70)
        img, depth, points, colors, pixels_2d, waypoints_proj = self.sim.render(target_position=np.array([-0.1,0.1,0]), distance=0.9, yaw=90, pitch=-70, waypoints=waypoints)

        #_, _, cam2world = self.sim.get_cam2world_transforms(target_position=np.array([-0.1,0.2,0]), distance=0.9, yaw=90, pitch=-60)
        #img, depth, points, colors, pixels_2d, waypoints_proj = self.sim.render(target_position=np.array([-0.1,0.2,0]), distance=0.9, yaw=90, pitch=-60, waypoints=waypoints)

        pixels_start = self.pix2pix_neighborhood(img, waypoints_proj[0])
        pixels_end = self.pix2pix_neighborhood(img, waypoints_proj[1])
        #print(pixels_start, waypoints_proj[0])
        #print(pixels_end, waypoints_proj[1])
        #pixels_start = np.array([waypoints_proj[0]])
        #pixels_end = np.array([waypoints_proj[1]])


        points_start = self.sim.deproject(depth, pixels_start, cam2world)
        points_end = self.sim.deproject(depth, pixels_end, cam2world)
    
        #points_start, _ = self.pix2point_neighborhood(img, waypoints_proj[0], pixels_2d.copy(), points.copy())
        #points_end, _ = self.pix2point_neighborhood(img, waypoints_proj[1], pixels_2d.copy(), points.copy())
        #points_start = self.filter_outliers(points_start)
        #points_end = self.filter_outliers(points_end)

        _, _, points1, colors1, _, _ = self.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=0)
        _, _, points2, colors2, _, _ = self.sim.render(distance=0.6, target_position=[0,0,0.1], yaw=90)
        points = np.vstack((points, points1, points2))
        colors = np.vstack((colors, colors1, colors2))

        start_idxs = self.point2point_neighborhood(points, points_start)
        end_idxs = self.point2point_neighborhood(points, points_end)

        #print(len(start_idxs), len(end_idxs))
        #colors[start_idxs] = (255,0,0)
        #colors[end_idxs] = (0,255,0)

        keypoint_cond = np.zeros(len(points))
        keypoint_cond[start_idxs] = 1.0
        keypoint_cond[end_idxs] = 2.0

        #keypoint_cond[start_idxs] = 1.0
        #keypoint_cond[end_idxs] = 2.0

        return img, points, colors, waypoints_proj, keypoint_cond

    def open(self, episode_idx, prompt, drawer_position='top_left'):
        drawer_to_pos = {'top_left': np.array([-0.1,0.2,0.2]), \
                         'top_right': np.array([0.1,0.2,0.2]), \
                         'middle': np.array([0.0,0.2,0.1])}

        approach_pos = drawer_to_pos[drawer_position]
        approach_pos += self.cabinet_loc - self.CANONICAL_POS
        grasp_euler = np.array([-90,0,0])
        grasp_pos = approach_pos.copy()
        grasp_pos[1] += 0.16

        self.active_open_drawer = drawer_position
        self.active_open_drawer_position = grasp_pos.copy()
        self.active_open_drawer_position[1] -= 0.05

        waypoints = [grasp_pos, approach_pos]

        img, pcl_points, pcl_colors, pixels, pcl_kpt_cond = self.take_rgbd(waypoints)

        start_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat() 
        orientations = [start_ori, end_ori]

        #### Open cabinet
        self.reset_robot()
        self.robot.move(approach_pos, grasp_euler)
        for i in range(50):
            self.sim.step()
        self.robot.move(grasp_pos, grasp_euler)
        for i in range(50):
            self.sim.step()
        self.robot.grasp()

        offset = approach_pos - grasp_pos
        num_steps = 15
        delta = offset/num_steps

        for i in range(num_steps):
            self.robot.move(grasp_pos + i*delta, grasp_euler)

        self.robot.release()
        self.reset_robot()

        #for i in range(50):
        #    self.sim.step()

        self.record(img, pcl_points, pcl_colors, pcl_kpt_cond, waypoints, orientations, pixels, episode_idx, prompt, 'open', visualize=False)
        return waypoints, pixels

    def close(self, episode_idx, prompt, drawer_position='top_left'):
        drawer_to_pos = {'top_left': np.array([-0.1,0.2,0.2]), \
                         'top_right': np.array([0.1,0.2,0.2]), \
                         'middle': np.array([0.0,0.2,0.1])}
        approach_pos = drawer_to_pos[drawer_position]
        approach_pos += self.cabinet_loc - self.CANONICAL_POS
        grasp_euler = np.array([-90,0,0])
        grasp_pos = approach_pos.copy()
        #grasp_pos[1] += 0.36
        grasp_pos[1] += 0.16
        waypoints = [approach_pos, grasp_pos]

        img, pcl_points, pcl_colors, pixels, pcl_kpt_cond = self.take_rgbd(waypoints)

        start_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat() 
        orientations = [start_ori, end_ori]

        #offset = approach_pos - grasp_pos
        #num_steps = 15
        #delta = offset/num_steps

        #for i in range(num_steps):
        #    self.robot.move(approach_pos - i*delta, grasp_euler)
        #    if i == 3:
        #        self.robot.grasp()

        self.record(img, pcl_points, pcl_colors, pcl_kpt_cond, waypoints, orientations, pixels, episode_idx, prompt, 'close', visualize=False)
        return waypoints, pixels

    def pick_place(self, episode_idx, prompt, place_position='cabinet'):
        grasp_pos = self.sim.get_base_position(self.graspable_objs[self.grasp_obj_idx])
        place_pos = self.active_open_drawer_position

        waypoints = [grasp_pos, place_pos]

        img, pcl_points, pcl_colors, pixels, pcl_kpt_cond = self.take_rgbd(waypoints)

        reset_euler= np.array([180,0,0]) # standard
        start_ori = R.from_euler('xyz', reset_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', reset_euler, degrees=True).as_quat() 
        orientations = [start_ori, end_ori]

        ### Grasp cup
        #self.robot.move(approach_pos, grasp_euler)
        #self.robot.move(grasp_pos, grasp_euler)
        #self.robot.grasp()

        ### Lift
        #self.robot.move(approach_pos, grasp_euler)

        ### Pour into other cup
        #self.robot.move(pour_pos, grasp_euler)
        #self.robot.move(pour_pos, pour_euler)

        ### Wait for pour to be done
        #for i in range(50):
        #    self.sim.step()

        self.record(img, pcl_points, pcl_colors, pcl_kpt_cond, waypoints, orientations, pixels, episode_idx, prompt, 'pick_place', visualize=False)
        return waypoints, pixels

    
    def pour(self, episode_idx, prompt):
        pos, final_pos = self.pour_loc, self.fill_loc
        if pos[1] - final_pos[1] < 0:
            alpha = 1
        else:
            alpha = -1

        approach_pos = pos + np.array([0,0,0.15])
        grasp_pos = pos - np.array([0,0,0.045])
        pour_pos = final_pos + np.array([0,alpha*(-0.05),0.10])
        grasp_euler = np.array([180,alpha*(-35),90])
        pour_euler = np.array([180,alpha*85,90])

        waypoints = [grasp_pos, pour_pos]

        img, pcl_points, pcl_colors, pixels, pcl_kpt_cond = self.take_rgbd(waypoints)

        start_ori = R.from_euler('xyz', grasp_euler, degrees=True).as_quat()  
        end_ori = R.from_euler('xyz', pour_euler, degrees=True).as_quat() 

        print(grasp_euler, alpha, start_ori)
        print(pour_euler, alpha, end_ori)

        orientations = [start_ori, end_ori]

        ### Grasp cup
        #self.robot.move(approach_pos, grasp_euler)
        #self.robot.move(grasp_pos, grasp_euler)
        #self.robot.grasp()

        ### Lift
        #self.robot.move(approach_pos, grasp_euler)

        ### Pour into other cup
        #self.robot.move(pour_pos, grasp_euler)
        #self.robot.move(pour_pos, pour_euler)

        ### Wait for pour to be done
        #for i in range(50):
        #    self.sim.step()

        self.record(img, pcl_points, pcl_colors, pcl_kpt_cond, waypoints, orientations, pixels, episode_idx, prompt, 'pour', visualize=False)
        return waypoints, pixels


    def record(self, img, points, colors, kpt_cond, waypoints, orientations, pixels, episode_idx, prompt, primitive_label, visualize=True):
        print('in record', episode_idx)
        cv2.imwrite('dset/images/%05d.jpg'%(episode_idx), img)
        np.save('dset/keypoints/%05d.npy'%(episode_idx), pixels)
        np.save('dset/lang/%05d.npy'%(episode_idx), prompt)
        np.save('dset/primitive_labels/%05d.npy'%(episode_idx), primitive_label)

        start, end = waypoints
        start_ori, end_ori = orientations

        # Subsample points
        #idxs = np.random.choice(len(points), min(5000, len(points)))
        idxs = np.where(points[:,2] < 0.5)[0]
        points = points[idxs]
        colors = colors[idxs]
        kpt_cond = kpt_cond[idxs]

        idxs = np.where(points[:,1] < 0.45)[0]
        points = points[idxs]
        colors = colors[idxs]
        kpt_cond = kpt_cond[idxs]

        idxs = points[:,2] > 0.002
        points = points[idxs]
        colors = colors[idxs]
        kpt_cond = kpt_cond[idxs]

        print('here', points.shape)

        #nbrs = NearestNeighbors(n_neighbors=800, algorithm='ball_tree').fit(points)
        #nbrs = NearestNeighbors(n_neighbors=2000, algorithm='ball_tree').fit(points)
        #nbrs = NearestNeighbors(n_neighbors=2000, algorithm='ball_tree').fit(points)
        #nbrs = NearestNeighbors(radius=0.1).fit(points)
        nbrs = NearestNeighbors(radius=0.125).fit(points)

        #distances, start_indices = nbrs.kneighbors(start.reshape(1,-1))
        #distances, end_indices = nbrs.kneighbors(end.reshape(1,-1))

        distances, start_indices = nbrs.radius_neighbors(start.reshape(1,-1), return_distance=True)
        distances, end_indices = nbrs.radius_neighbors(end.reshape(1,-1), return_distance=True)
        start_indices = start_indices[0]
        end_indices = end_indices[0]

        print(len(start_indices), len(end_indices), len(points), (len(start_indices) + len(end_indices))/len(points))

        # Set up cls
        cls = np.zeros(len(points))
        both_indices = np.intersect1d(start_indices, end_indices)

        start_indices = np.setdiff1d(start_indices, both_indices)
        end_indices = np.setdiff1d(end_indices, both_indices)
        cls[start_indices] = 1.0
        cls[end_indices] = 2.0
        cls[both_indices] = 3.0

        # Set up offsets 
        offsets = np.zeros_like(points)
        offsets[start_indices] = points[start_indices] - start
        offsets[end_indices] = points[end_indices] - end

        # Save points, colors, offsets
        data = {'xyz':points, 'xyz_color':colors, 'xyz_kpt':kpt_cond, 'start_waypoint':start, 'end_waypoint':end, 'cls':cls, 'start_ori':start_ori, 'end_ori':end_ori, 'label':primitive_label}
        np.save('dset/%05d.npy'%episode_idx, data)

        if visualize:
        #if True:
            cv2.putText(img, prompt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1, 2)
            for pixel in pixels:
                cv2.circle(img, tuple(pixel), 4, (255,0,0), -1)
            cv2.imshow('img', img)
            cv2.waitKey(0)

            offsets_vis = colors.copy()

            distances_vis = np.ones((3, len(points)))
            distances = np.linalg.norm(offsets, axis=1)
            distances_normalized = (distances - np.amin(distances))/(np.amax(distances) - np.amin(distances))

            distances_vis[1] = distances_normalized
            distances_vis[2] = distances_normalized
            distances_vis = distances_vis.T

            offsets_vis[both_indices] = (0,0,0)

            offsets_vis += (distances_vis*255).astype(np.uint8)
    
            cls_vis = (np.vstack((cls, cls, cls)).T)*50

            pcd = o3d.geometry.PointCloud()
            rot = R.from_euler('yz', [90,90], degrees=True).as_matrix()
            rot = R.from_euler('y', 180, degrees=True).as_matrix()@rot
            points = (rot@points.T).T
    
            pcd.points = o3d.utility.Vector3dVector(points)
            #pcd.colors = o3d.utility.Vector3dVector(cls_vis/255.)
            pcd.colors = o3d.utility.Vector3dVector(offsets_vis/255.)
            #pcd.colors = o3d.utility.Vector3dVector(colors/255.)

            o3d.visualization.draw_geometries([pcd])

    def get_pick_place_promp(self):
        prompts = ['Pick up the %s and put it in the %s', \
                   'Get the %s and put it in the %s', \
                   'Put the %s in the %s']
        prompt = random.choice(prompts)
        grasp_obj_idx = random.choice(np.arange(len(self.graspable_objs)))
        grasp_obj_name = self.graspable_obj_names[grasp_obj_idx]
        prompt = prompt%(grasp_obj_name, 'cabinet')
        return prompt, grasp_obj_idx, grasp_obj_name

    def get_demo_sequences(self):
        #language_to_primitive_map = [[['Open the top left cabinet'], [('open', 'top_left')]],
        #                         [['Open the top right cabinet'], [('open', 'top_right')]],
        #                         [['Open the middle cabinet'], [('open', 'middle')]],
        #                         [['Open a drawer'], [('open')]],
        #                         [['Open a cabinet'], [('open')]],
        #                         [['Open the cupboard'], [('open')]],
        #                         [['Pour me a glass'], [('pour')]],
        #                         [['Fill up my glass'], [('pour')]],
        #                         [['Fill up the empty cup'], [('pour')]],
        #                         [['Pour the full glass into the empty one'], [('pour')]],
        #                         [['Fill up the big cup'], [('pour')]],
        #                         [['Fill up my glass', 'Open a drawer', 'Close the cabinet'], [('pour'), ('open', 'top_left'), ('close', 'top_left')]],
        #                         [['Fill up my glass', 'Open a drawer', 'Close the cabinet'], [('pour'), ('open', 'top_right'), ('close', 'top_right')]],
        #                         [['Fill up my glass', 'Open a drawer', 'Close the cabinet'], [('pour'), ('open', 'middle'), ('close', 'middle')]]]*15

        #language_to_primitive_map = [
        #                         [['Pour me a glass'], [('pour')]],
        #                         [['Fill up my glass'], [('pour')]],
        #                         [['Fill up the empty cup'], [('pour')]]]

        language_to_primitive_map = [[['Open the top left cabinet'], [('open', 'top_left')]],
                                 [['Open the top right cabinet'], [('open', 'top_right')]],
                                 [['Open the middle cabinet'], [('open', 'middle')]],
                                 [['Open a drawer', 'Close the cabinet'], [('open', 'top_left'), ('close', 'top_left')]],
                                 [['Open a drawer', 'Close the cabinet'], [('open', 'top_right'), ('close', 'top_right')]],
                                 [['Open a drawer', 'Close the cabinet'], [('open', 'middle'), ('close', 'middle')]]]

        #language_to_primitive_map = [
        #                         [['Pick up the %s and put it in the %s'], [('pick_place')]],
        #                         [['Get the %s and put it in the %s'], [('pick_place')]],
        #                         [['Put the %s in the %s'], [('pick_place')]]]


        #for i in range(100):
        #    prompt, grasp_obj_idx, grasp_obj_name = self.get_pick_place_promp()
        #    rand_cabinet = random.choice(['top_left', 'top_right', 'middle'])
        #    language_to_primitive_map += [
        #                                [[int(grasp_obj_idx), prompt, prompt], [grasp_obj_name, ('open', rand_cabinet), ('pick_place')]],
        #                                ]
                                 
        return language_to_primitive_map

if __name__ == '__main__':
    dset_dir = 'dset'
    images_dir = os.path.join(dset_dir, 'images')
    lang_dir = os.path.join(dset_dir, 'lang')
    keypoints_dir = os.path.join(dset_dir, 'keypoints')
    primitive_labels_dir = os.path.join(dset_dir, 'primitive_labels')
    for d in [dset_dir, images_dir, lang_dir, keypoints_dir, primitive_labels_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    sim = PyBullet(render=True, background_color=np.array([255,255,255]))
    robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
    task = TableTop(sim, robot)

    episode = 0

    while episode < 350:
    #while episode < 70:
        if episode%5 == 0:
            task.sim.close()
            sim = PyBullet(render=True, background_color=np.array([255,255,255]))
            robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
            task = TableTop(sim, robot)

        success = False
        while not success:
            try:
                print('resetting')
                task.reset()
                language_to_primitive = task.get_demo_sequences()
                demo_seq = random.choice(language_to_primitive)
                instructions, actions = demo_seq
                for instr, action in zip(instructions, actions):
                    print(instr, action)
                    if isinstance(instr, int):
                        task.grasp_obj_idx = instr
                        continue
                    if isinstance(action, str):
                        if action == 'open':
                            task.open(episode, instr)
                        elif action == 'pour':
                            task.pour(episode, instr)
                        elif action == 'close':
                            task.close(episode, instr)
                        elif action == 'pick_place':
                            task.pick_place(episode, instr)
                    else:
                        if action[0] == 'open':
                            task.open(episode, instr, drawer_position=action[1])
                        elif action[0] == 'close':
                            task.close(episode, instr, drawer_position=action[1])
                    episode += 1
                success = True
            except:
                continue
