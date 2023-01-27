# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

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
            point_set, colors, offset_labels = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.load(fn, allow_pickle=True).item()

            point_set = data['xyz']
            colors = data['xyz_color']
            offset_labels = data['gripper_offsets']
            #cls_labels = np.zeros((len(point_set), 1))
            idxs = np.where(np.linalg.norm(offset_labels, axis=1) > 0)
            #cls_labels[idxs] = 1
            cls_labels = data['cls'].reshape(-1,1)
            if len(self.cache) < self.cache_size:
                #self.cache[index] = (point_set, colors, offset_labels)
                self.cache[index] = (point_set, colors, cls_labels)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        point_set = point_set[choice, :]
        colors = colors[choice, :]
        #offset_labels = offset_labels[choice, :]
        cls_labels = cls_labels[choice, :].T

        inp = np.vstack((point_set.T, colors.T)).T
        #return inp, offset_labels
        return inp, cls_labels

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    dset = PourDataset()
    print(len(dset))
    #pts, colors, offsets = dset[0]
    inp, offsets = dset[0]
    print(inp.shape, offsets.shape)
