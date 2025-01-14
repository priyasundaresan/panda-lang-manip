"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
#from data_utils.PourDataLoader_cls_off import PourDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d

class Inference:
    def __init__(self, ROOT_DIR='/home/priya/iliad/panda-lang-manip/panda_gym/envs/inference'):
        sys.path.insert(0, ROOT_DIR)
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.num_outputs = 3
        self.inp_dim = 6
        self.num_classes = 3
        self.MODEL = importlib.import_module('model_cls_off')
        self.classifier = self.MODEL.get_model(self.num_outputs, self.inp_dim, self.num_classes).cuda()
        checkpoint = torch.load('/home/priya/iliad/panda-lang-manip/panda_gym/envs/inference/log_l1/part_seg/2023-01-19_15-54/checkpoints/best_model.pth')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.eval()
        
    def visualize(self, pcl_input, offsets):
        pcl_input = pcl_input.squeeze().cpu().numpy().T
        points = pcl_input[:,:3]
        colors = pcl_input[:,3:]
        offsets = offsets.squeeze()
    
        # Set up offsets 
        distances_vis = np.zeros((len(points), 3))
        distances = np.linalg.norm(offsets, axis=1)
        distances_normalized = (distances - np.amin(distances))/(np.amax(distances) - np.amin(distances))
        distances_vis[:, 2] = distances_normalized
        colors += (distances_vis*255).astype(np.uint8)
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/255.)
    
        o3d.visualization.draw_geometries([pcd]) # with display

    def pc_normalize(self, pc, centroid=None, m=None):
        if centroid is None:
            centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if m is None:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc, centroid, m

    def predict(self, points):
        points_xyz = points.T[:3,:].T
        colors_xyz = points.T[3:,:].T
        points_xyz, centroid, m = self.pc_normalize(points_xyz)
        inp = np.vstack((points_xyz.T, colors_xyz.T))
        points_t = torch.from_numpy(inp).cuda().float().unsqueeze(0)

        with torch.no_grad():
            pred, offsets = self.classifier(points_t)
            pred = pred.contiguous().view(-1, self.num_classes)
            cls  = pred.data.max(1)[1].cpu().numpy()

            offsets = offsets.squeeze().detach().cpu().numpy()

            start_idxs = np.where(cls == 1)
            end_idxs = np.where(cls == 2)
            
            start_offsets = offsets[start_idxs]
            end_offsets = offsets[end_idxs]
            
            start_waypts = points_xyz[start_idxs] - start_offsets
            end_waypts = points_xyz[end_idxs] - end_offsets
            
            start_waypt = np.mean(start_waypts, axis=0)
            end_waypt = np.mean(end_waypts, axis=0)
    
            start_waypt *= m
            start_waypt += centroid
            end_waypt *= m
            end_waypt += centroid

            return start_waypt, end_waypt
