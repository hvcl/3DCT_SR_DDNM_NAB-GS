import os

from PIL import Image
import numpy as np
import cv2

from torchvision import transforms
from torch.utils.data import Dataset
import pickle

import torch

normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))

class UHRCTDataset(Dataset):
    def __init__(self, args, degraded_path, gt_path): #, warped_noise_path):
        self.args = args
        self.degraded_path = degraded_path
        self.gt_path = gt_path

        with open(degraded_path, 'rb') as handle:
            degraded_data = pickle.load(handle)
        with open(gt_path, 'rb') as handle:
            gt_data = pickle.load(handle)

        self.degraded_data_list = degraded_data['train']['projections'] # length: 100
        self.gt_data_list = gt_data['train']['projections'] # length: 100
        self.transform = transforms.ToTensor()

        if "param_search" in self.args.image_folder:
            self.degraded_data_list = self.degraded_data_list[::20]
            self.gt_data_list = self.gt_data_list[::20]

    def __len__(self):
        return len(self.gt_data_list)

    def __getitem__(self, idx):
        degraded_image = normalize_array(self.degraded_data_list[idx])
        gt_image = normalize_array(self.gt_data_list[idx])

        # if "up2x_sr4x_down2x" in self.args.image_folder:
        #     # upsample degraded image 2x, bilinear
        #     # degraded_image = cv2.resize(degraded_image, (degraded_image.shape[1]*2, degraded_image.shape[0]*2), interpolation=cv2.INTER_LINEAR) # 32.50
        #     degraded_image = cv2.resize(degraded_image, (degraded_image.shape[1]*2, degraded_image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        # elif "pad_sr4x_unpad" in self.args.image_folder:
        #     # zero pad degrade_image to have a shape of 128x128, 
        #     degraded_image = np.pad(degraded_image, ((32, 32), (32, 32)), mode='constant')

        degraded_image = np.array(Image.fromarray(degraded_image*255).convert('RGB'))
        gt_image = np.array(Image.fromarray(gt_image*255).convert('RGB'))

        degraded_image = self.transform(degraded_image)
        gt_image = self.transform(gt_image)

        data = {
            "deg_img": degraded_image,
            "gt_img": gt_image,
        }

        return data