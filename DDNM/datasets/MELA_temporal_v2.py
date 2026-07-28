import os

from PIL import Image
import numpy as np
import cv2

from torchvision import transforms
from torch.utils.data import Dataset
import pickle

import torch
import yaml
from datasets.generateData import ConeGeometry_special, loadImage
import scipy.io
import tigre
from tqdm import tqdm

def create_angle_list(yaml_data, args):
    # 기존 angles 생성
    original_angles = np.linspace(0, yaml_data["totalAngle"] / 180 * np.pi, yaml_data["numTrain"]+1)[:-1] + yaml_data["startAngle"] / 180 * np.pi
    
    # 새로운 angles 리스트 초기화
    new_angles = []
    
    for i in range(len(original_angles)):
        if i == 0 or i == len(original_angles) - 1:
            # 첫 번째와 마지막 각도는 그대로 유지
            new_angles.append(original_angles[i])
        else:
            temp_angles = []
            for j in range(-(args.N_proj_per_iter // 2), args.N_proj_per_iter // 2 + 1):
                # if j != 0:  # 중앙 각도(j=0)를 제외
                angle = original_angles[i] + j * args.degree_interval * np.pi / 180
                temp_angles.append(angle)
            # new_angles.append([original_angles[i]] + temp_angles)  # 중앙 각도를 먼저 추가
            new_angles.append(temp_angles)  # 중앙 각도를 먼저 추가
    
    return new_angles



class MELADataset_Temporal_v2(Dataset):
    def __init__(self, args, data_path, config_path, gt_path):
        self.normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))
        self.args = args
        self.data_path = data_path
        self.config_path = config_path
        self.gt_path = gt_path

        # Load configuration
        with open(config_path, "r") as handle:
            yaml_data = yaml.safe_load(handle)
        with open(gt_path, 'rb') as handle:
            gt_data = pickle.load(handle)
        self.gt_data_list = gt_data['train']['projections']

        # Load CT image
        geo = ConeGeometry_special(yaml_data)
        self.geo = geo
        gt, lr = loadImage(data_path, 
                            yaml_data['lrVol'], 
                            yaml_data["nVoxel"], 
                            yaml_data["convert"],
                            yaml_data["rescale_slope"], 
                            yaml_data["rescale_intercept"], 
                            yaml_data["normalize"]
                            )
        yaml_data["image"] = gt.copy()

        if lr is None:
            print("Target shape is same with original volume.")
            img = gt
        else:
            scale = np.array(gt.shape) / np.array(lr.shape)    
            upsample = 'trilinear'
            print(f"Volume {lr.shape} is upsampled into {gt.shape} by {upsample}.")
            if upsample == 'trilinear':
                img = scipy.ndimage.zoom(lr, scale, order=1) # Linear interpolation
            elif upsample == 'cubic':
                img = scipy.ndimage.zoom(lr, scale, order=3, prefilter=False) # Cubic interpolation
            elif upsample == 'prefilter':
                img = scipy.ndimage.zoom(lr, scale, order=3)
            else:
                print('No upsample method was selected')
                img = scipy.ndimage.zoom(lr, scale, order=0)
            yaml_data["upsampled"] = img.copy()
                
        yaml_data["train"] = {"angles": 
                              np.linspace(0, yaml_data["totalAngle"] / 180 * np.pi, yaml_data["numTrain"]+1)[:-1] 
                              + yaml_data["startAngle"]/ 180 * np.pi
                              } # angles: 0 to pi, 100 angles in total
        yaml_data["train"]["angles_list"] = create_angle_list(yaml_data, args)

        self.img = np.transpose(img, (2, 0, 1)).copy()
        self.yaml_data = yaml_data
        self.N_proj_per_iter = args.N_proj_per_iter
        self.degree_interval = args.degree_interval

        self.transform = transforms.ToTensor()

        self.N_prev = self.args.N_proj_per_iter // 2
        self.N_next = self.args.N_proj_per_iter // 2
        self.generate_projections()

    def generate_projections(self):
        self.projections = []
        proj = tigre.Ax(self.img, self.geo, self.yaml_data["train"]["angles"])[:, ::-1, :]
        
        return
        for angles in tqdm(self.yaml_data["train"]["angles_list"], desc="Generating projections"):
            if isinstance(angles, (float, np.float64)):
                flat_angles = np.array([angles])
            elif isinstance(angles, list):
                flat_angles = np.array([angle for sublist in angles for angle in (sublist if isinstance(sublist, list) else [sublist])])
            else:
                flat_angles = np.array(angles)
            
            proj = tigre.Ax(self.img, self.geo, flat_angles)[:, ::-1, :]
            
            # Downsample projections
            downsample_factor = self.args.deg_scale
            downsampled_proj = scipy.ndimage.zoom(proj, (1, 1/downsample_factor, 1/downsample_factor), order=1)
            
            self.projections.append(downsampled_proj)

            if len(self.projections) == 2:
                break

    def __len__(self):
        return len(self.projections)
        return len(self.gt_data_list)

    def __getitem__(self, idx):
        data = {}
        # data['angles'] = self.yaml_data["train"]["angles_list"][idx]
        
        projections = self.projections[idx]
        for i, proj in enumerate(projections):
            norm_proj = self.normalize_array(proj)
            data[f'deg_img_{i}'] = self.transform(np.array(Image.fromarray(norm_proj*255).convert('RGB')))

        gt_image = self.normalize_array(self.gt_data_list[idx])
        gt_image = np.array(Image.fromarray(gt_image*255).convert('RGB'))
        data['gt_img'] = self.transform(gt_image)

        return data

    '''
    def __getitem__(self, idx):
        angles = self.yaml_data["train"]["angles"][idx]
        if isinstance(angles, (float, np.float64)):
            flat_angles = np.array([angles])
        elif isinstance(angles, list):
            flat_angles = np.array([angle for sublist in angles for angle in (sublist if isinstance(sublist, list) else [sublist])])
        else:
            flat_angles = np.array(angles)  # angles가 이미 numpy 배열인 경우

        projections = tigre.Ax(self.img, self.geo, flat_angles)[:, ::-1, :]

        # Normalize and reshape projections
        data = {}
        for i, proj in enumerate(projections):
            norm_proj = normalize_array(proj)
            data[f'deg_img_{i}'] = self.transform(np.array(Image.fromarray(norm_proj*255).convert('RGB')))

        gt_image = normalize_array(self.gt_data_list[idx])
        gt_image = np.array(Image.fromarray(gt_image*255).convert('RGB'))
        gt_image = self.transform(gt_image)

        data['gt_img'] = gt_image
        
        # degraded_image = self.transform(degraded_image)

        # degraded_image_prev = torch.zeros_like(degraded_image)
        # degraded_image_next = torch.zeros_like(degraded_image)

        # warped_noise_prev = np.zeros_like(warped_noise)
        # warped_noise_next = np.zeros_like(warped_noise)

        # if idx > 0:
        #     # warped_noise_prev = self.warped_noise_data[idx - 1]
        #     degraded_image_prev = normalize_array(self.degraded_data_list[idx - 1])
        #     degraded_image_prev = self.transform(np.array(Image.fromarray(degraded_image_prev*255).convert('RGB')))

        # if idx < len(self.gt_data_list) - 1:
        #     # warped_noise_next = self.warped_noise_data[idx + 1]
        #     degraded_image_next = normalize_array(self.degraded_data_list[idx + 1])
        #     degraded_image_next = self.transform(np.array(Image.fromarray(degraded_image_next*255).convert('RGB')))

        
        # data['deg_img'] = degraded_image
        # data['class'] = 0
        return data
        # return degraded_image, gt_image, degraded_image_prev, degraded_image_next, 0, # warped_noise, warped_noise_prev, warped_noise_next, 0
    '''
