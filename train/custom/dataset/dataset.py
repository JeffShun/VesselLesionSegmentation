"""data loader."""

import numpy as np
from torch.utils import data
import torch
from scipy.ndimage import filters

class MyDataset(data.Dataset):
    def __init__(
        self,
        dst_list_file,
        transforms,
        gaussian_sigma
        ):
        self.gaussian_sigma = gaussian_sigma
        self.data_lst = self._load_files(dst_list_file)
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        img = data['img']
        seg = data['seg']
        vesselmask = data['vesselmask']

        # vessel mask转化为vessel hint
        vessel_hint = filters.gaussian_filter(vesselmask.astype("float"), self.gaussian_sigma)
        
        # transform前，数据必须转化为[C,H,W]的形状
        img = img[np.newaxis,:,:,:].astype(np.float32)
        label = seg[np.newaxis,:,:,:].astype(np.float32)
        vessel_hint = vessel_hint[np.newaxis,:,:,:].astype(np.float32)

        if self._transforms:
            img, label, vessel_hint = self._transforms(img, label, vessel_hint)

        img = torch.cat([img, vessel_hint], 0)
        vesseldomin = (vessel_hint > 0).float()
        
        return img, label, vesseldomin

