#coding=utf-8
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import os
class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, rgb,t, mask):
        for op in self.ops:
            rgb,t, mask = op(rgb,t, mask)
        return rgb,t, mask

class Normalize(object):
    def __call__(self, rgb,t, mask):
        rgb = rgb/255
        t = t/ 255
        mask /= 255
        return rgb,t, mask

class Minusmean(object):
    def __init__(self, mean1,mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, rgb,t, mask):
        rgb = rgb - self.mean1
        t = t - self.mean2
        mask /= 255
        return rgb,t, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb,t, mask):
        rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        t = cv2.resize(t, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return rgb,t, mask

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb,t, mask):
        H,W,_ = rgb.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        rgb = rgb[ymin:ymin+self.H, xmin:xmin+self.W, :]
        t = t[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return rgb,t, mask

class RandomHorizontalFlip(object):
    def __call__(self, rgb,t, mask):
        if np.random.randint(2)==1:
            rgb = rgb[:,::-1,:].copy()
            t = t[:, ::-1, :].copy()
            mask  =  mask[:,::-1,:].copy()
        return rgb,t, mask

class ToTensor(object):
    def __call__(self, rgb,t, mask):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)
        t = torch.from_numpy(t)
        t = t.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        return rgb.float(),t.float(),mask.mean(dim=0,keepdim=True).float()

class RandomSpitalTransformation(object):
    def __call__(self, rgb,t, mask):
        RTI= 10*torch.rand(1)+2 # Random Transformation Intensity
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        dtheta = 2*(torch.rand(6)-0.5)/RTI
        theta = dtheta + identity_theta
        resampling_grid = F.affine_grid(theta.view(-1,2, 3), rgb.unsqueeze(0).size(), align_corners=True)
        option = 10*torch.rand(1)
        if option < 5:
            rgb_wrap = F.grid_sample(rgb.unsqueeze(0).float(), resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            mask_wrap = F.grid_sample(mask.unsqueeze(0).float(), resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            return rgb_wrap.squeeze(0) , t, mask_wrap.squeeze(0)
        else:
            t_wrap = F.grid_sample(t.unsqueeze(0).float(), resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            return rgb, t_wrap.squeeze(0), mask

class RST_Test(object):
    def __call__(self, rgb,t, mask):
        RTI= 10*torch.rand(1)+2 # Random Transformation Intensity
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        dtheta = 2*(torch.rand(6)-0.5)/RTI
        theta = dtheta + identity_theta
        resampling_grid = F.affine_grid(theta.view(-1,2, 3), rgb.unsqueeze(0).size(), align_corners=True)
        t_wrap = F.grid_sample(t.unsqueeze(0).float(), resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return rgb, t_wrap.squeeze(0), mask
