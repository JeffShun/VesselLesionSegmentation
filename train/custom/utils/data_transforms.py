
import random
import torch
import math
from torch.nn import functional as F
import numpy as np
"""
数据预处理工具
1、所有数据预处理函数都包含两个输入: img 、label
2、img、label的输入维度为3维[C,H,W]，第一个维度是通道数
"""

class TransformCompose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label, other):
        for t in self.transforms:
            img, label, other = t(img, label, other)
        return img, label, other

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class normlize(object):
    def __init__(self, win_clip=False):
        self.win_clip = win_clip

    def __call__(self, img, label, other): 
        img_o, label_o, other_o = img, label, other 
        if self.win_clip:
            min_val, max_val = self._get_auto_windowing(img, lower_percentile=0.1, upper_percentile=99.9)
            img = np.clip(img, min_val, max_val)
        img_o = self._norm(img)
        return img_o, label_o, other_o
    
    def _get_auto_windowing(self, image, lower_percentile=0.1, upper_percentile=99.9):
        
        """
        计算图像直方图的低和高百分位数。
        """
        hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # 计算累计直方图
        cumulative_hist = np.cumsum(hist)
        cumulative_hist = cumulative_hist / cumulative_hist[-1]  # 归一化
        
        # 计算低和高百分位数
        lower_threshold = np.percentile(image, lower_percentile)
        upper_threshold = np.percentile(image, upper_percentile)

        # 返回窗位和窗宽
        return lower_threshold, upper_threshold
    

    def _norm(self, img):
        ori_shape = img.shape
        img_flatten = img.reshape(ori_shape[0], -1)
        img_min = img_flatten.min(axis=-1,keepdims=True)[0]
        img_max = img_flatten.max(axis=-1,keepdims=True)[0]
        img_norm = (img_flatten - img_min)/(img_max - img_min)
        img_norm = img_norm.reshape(ori_shape)
        return img_norm   

class to_tensor(object):
    def __init__(self, use_gpu=False):
        self.device = "cuda:0" if use_gpu else "cpu"
    def __call__(self, img, label, other):
        img_o = torch.from_numpy(img).to(self.device)
        label_o = torch.from_numpy(label).to(self.device)
        other_o = torch.from_numpy(other).to(self.device)
        return img_o, label_o, other_o


class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label, other):
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="trilinear")[0]
        label_o = torch.nn.functional.interpolate(label[None], size=self.size, mode="trilinear")[0]
        other_o = torch.nn.functional.interpolate(other[None], size=self.size, mode="trilinear")[0]
        label_o = (label_o > 0.4).float()
        return img_o, label_o, other_o


class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, img, label, other):
        img_o, label_o, other_o = img, label, other
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
        return img_o, label_o, other_o
    

class random_add_gaussian_noise(object):
    def __init__(self, prob=0.2, mean=0, std=1):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img, label, other):
        img_o, label_o, other_o = img, label, other
        if random.random() < self.prob:
            noise = torch.randn_like(img, device=img.device) * self.std + self.mean
            noisy_image = img + noise
            img_o = torch.clip(noisy_image, 0 , 1)
        return img_o, label_o, other_o
    

class random_flip(object):
    def __init__(self, axis=0, prob=0.5):
        assert isinstance(axis, int) and axis in [1,2,3]
        self.axis = axis
        self.prob = prob
    def __call__(self, img, label, other):
        img_o, label_o, other_o = img, label, other
        if random.random() < self.prob:
            img_o = torch.flip(img, [self.axis])
            label_o = torch.flip(label, [self.axis])
            other_o = torch.flip(other, [self.axis])
        return img_o, label_o, other_o
    

class random_rotate3d(object):
    def __init__(self,
                x_theta_range=[-180,180], 
                y_theta_range=[-180,180], 
                z_theta_range=[-180,180],
                prob=0.5, 
                ):
        self.prob = prob
        self.x_theta_range = x_theta_range
        self.y_theta_range = y_theta_range
        self.z_theta_range = z_theta_range
    
    def __call__(self, img, label, other):
        img_o, label_o, other_o = img, label, other
        if random.random() < self.prob:
            random_angle_x = random.uniform(self.x_theta_range[0], self.x_theta_range[1])
            random_angle_y = random.uniform(self.y_theta_range[0], self.y_theta_range[1])
            random_angle_z = random.uniform(self.z_theta_range[0], self.z_theta_range[1]) 
            img_o = self._rotate3d(img,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")
            label_o = self._rotate3d(label,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")
            other_o = self._rotate3d(other,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")

            img_o = torch.clip(img_o, 0 , 1)
            label_o = (label_o > 0.4).float()

        return img_o, label_o, other_o

    def _rotate3d(self, data, angles=[0,0,0], itp_mode="bilinear"): 
        alpha, beta, gama = [(angle/180)*math.pi for angle in angles]
        transform_matrix = torch.tensor([
            [math.cos(beta)*math.cos(gama), math.sin(alpha)*math.sin(beta)*math.cos(gama)-math.sin(gama)*math.cos(alpha), math.sin(beta)*math.cos(alpha)*math.cos(gama)+math.sin(alpha)*math.sin(gama), 0],
            [math.cos(beta)*math.sin(gama), math.cos(alpha)*math.cos(gama)+math.sin(alpha)*math.sin(beta)*math.sin(gama), -math.sin(alpha)*math.cos(gama)+math.sin(gama)+math.sin(beta)*math.cos(alpha), 0],
            [-math.sin(beta), math.sin(alpha)*math.cos(beta),math.cos(alpha)*math.cos(beta), 0]
            ])
        # 旋转变换矩阵
        transform_matrix = transform_matrix.unsqueeze(0)
        # 为了防止形变，先将原图padding为正方体，变换完成后再切掉
        data = data.unsqueeze(0)
        data_size = data.shape[2:]
        pad_x = (max(data_size)-data_size[0])//2
        pad_y = (max(data_size)-data_size[1])//2
        pad_z = (max(data_size)-data_size[2])//2
        pad = [pad_z,pad_z,pad_y,pad_y,pad_x,pad_x]
        pad_data = F.pad(data, pad=pad, mode="constant",value=0).to(torch.float32)
        grid = F.affine_grid(transform_matrix, pad_data.shape).to(data.device)
        output = F.grid_sample(pad_data, grid, mode=itp_mode)
        output = output.squeeze(0)
        output = output[:,pad_x:output.shape[1]-pad_x, pad_y:output.shape[2]-pad_y, pad_z:output.shape[3]-pad_z]
        return output







