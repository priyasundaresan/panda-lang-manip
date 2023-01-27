# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class PourDataset(Dataset):
    def __init__(self,root = './data/pour_dset', npoints=2048, split='train'):
        self.npoints = npoints
        self.root = root
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        self.datapath = {}
        for idx, fn in enumerate(sorted(os.listdir('%s/%s'%(self.root, split)))):
            self.datapath[idx] = os.path.join('%s/%s/%s'%(self.root, split, fn))

    def __getitem__(self, index):
        if index in self.cache:
            point_set, colors, cls_labels, start_waypt, end_waypt = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.load(fn, allow_pickle=True).item()

            point_set = data['xyz']
            colors = data['xyz_color']
            start_waypt = data['start_waypoint']
            end_waypt = data['end_waypoint']
            cls_labels = data['cls'].reshape(-1,1)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, colors, cls_labels, start_waypt, end_waypt)

        point_set[:, 0:3], centroid, m = pc_normalize(point_set[:, 0:3])
        start_waypt, _, _ = pc_normalize(start_waypt, centroid, m)
        end_waypt, _, _ = pc_normalize(end_waypt, centroid, m)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)

        point_set = point_set[choice, :]
        colors = colors[choice, :]
        cls_labels = cls_labels[choice, :].T

        inp = np.vstack((point_set.T, colors.T)).T

        return inp, cls_labels, start_waypt, end_waypt

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    dset = PourDataset('data/pour_dset4', npoints=5000, split='test')
    inp, offsets, start, end = dset[0]
    print(inp.shape, offsets.shape)
