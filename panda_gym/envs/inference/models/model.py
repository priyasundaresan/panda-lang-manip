import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '..')
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_output_channels, num_input_channels, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], num_input_channels, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        #self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        #self.fp1 = PointNetFeaturePropagation(in_channel=131+num_classes+num_input_channels, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=131+num_input_channels, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_output_channels+num_classes, 1)
        self.num_classes = num_classes

    #def forward(self, xyz, cls_label):
    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        #if self.normal_channel:
        #    l0_points = xyz
        #    l0_xyz = xyz[:,:3,:]
        #else:
        #    l0_points = xyz
        #    l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        #cls_label_one_hot = cls_label.view(B,self.num_classes,1).repeat(1,1,N)
        #l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)

        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1) # comment out
        x = x.permute(0, 2, 1)
        return x, l3_points

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        #self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss

if __name__ == '__main__':
    #num_outputs = 3
    num_outputs = 0
    num_classes = 2
    num_inputs = 6 # xyz + color
    model = get_model(num_outputs, num_inputs, num_classes) 
    xyz = np.zeros((6,2048))
    
    xyz_tensor = torch.from_numpy(xyz).unsqueeze(0).float()
    print(xyz_tensor.shape)

    cls_tensor = to_categorical(torch.Tensor([1]), 2)
    result, _ = model(xyz_tensor, cls_tensor)
    print(result.shape, result[0])
