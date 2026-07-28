import os

from PIL import Image
import numpy as np
import cv2

from torchvision import transforms
from torch.utils.data import Dataset
import pickle
    
# class MELADataset(Dataset):
#     def __init__(self, path):
#         self.path = path
#         self.data_list = [file for file in os.listdir(path) if file.lower().endswith('.png')]
#         with open(path, 'rb') as handle:
#             data = pickle.load(handle)
#         self.data_list = data['train']['projections']
#         self.transform = transforms.ToTensor()  # Converts PIL image in the range [0, 255] to float tensor in the range [0., 1.]

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         file_name = self.data_list[idx]

#         # Load the image
#         image_path = os.path.join(self.path, file_name)
#         image = Image.open(image_path).convert('L')

#         # Apply transformations (resize and convert to tensor)
#         image = self.transform(image)

#         return image, 0

normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))

class MELADataset(Dataset):
    def __init__(self, degraded_path, gt_path): # , optical_flow_path):
        self.degraded_path = degraded_path
        self.gt_path = gt_path
        # self.warped_noise_path = warped_noise_path
        # self.optical_flow_path = optical_flow_path

        # self.path = path
        with open(degraded_path, 'rb') as handle:
            degraded_data = pickle.load(handle)
        with open(gt_path, 'rb') as handle:
            gt_data = pickle.load(handle)

        # warped_noise: (num_frames, height, width, channels) = (100, 512, 512, 3)
        # self.warped_noise_data = np.load(warped_noise_path, allow_pickle=True) # .permute(0, 3, 1, 2)
        # self.warped_noise_data = np.load(warped_noise_path, allow_pickle=True).transpose(0, 3, 1, 2) # shape: (100, 3, 512, 512)
        # print(self.warped_noise_data.shape, self.optical_flow_data.shape)
        # (100, 128, 128, 3) (99, 2, 128, 128)

        self.degraded_data_list = degraded_data['train']['projections']
        self.gt_data_list = gt_data['train']['projections']

        # self.data_list = [file for file in os.listdir(path) if file.lower().endswith('.png')]
        # with open(path, 'rb') as handle:
        #     data = pickle.load(handle)
        # self.data_list = data['train']['projections']
        self.transform = transforms.ToTensor()  # Converts PIL image in the range [0, 255] to float tensor in the range [0., 1.]

    def __len__(self):
        return len(self.gt_data_list)

    def __getitem__(self, idx):
        degraded_image = normalize_array(self.degraded_data_list[idx])
        gt_image = normalize_array(self.gt_data_list[idx])
        # warped_noise = self.warped_noise_data[idx] # shape: (H, W, 3)
        # print(np.min(warped_noise), np.max(warped_noise)) #, np.min(optical_flow), np.max(optical_flow))
        # exit()
        degraded_image = np.array(Image.fromarray(degraded_image*255).convert('RGB'))
        gt_image = np.array(Image.fromarray(gt_image*255).convert('RGB'))

        degraded_image = self.transform(degraded_image)
        gt_image = self.transform(gt_image)

        return degraded_image, gt_image, 0 #, warped_noise
        # file_name = self.data_list[idx]

        # Load the image
        # image_path = os.path.join(self.path, file_name)
        # image = Image.open(image_path).convert('L')
        # image = normalize_array(self.data_list[idx])
        # if image.shape != (512, 512):
        #     print(f"Image is resized from {image.shape} to (512, 512)")
        #     image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        
        # # Apply transformations (resize and convert to tensor)
        # image = np.array(Image.fromarray(image*255).convert('RGB'))
        # image = self.transform(image)

        # return image, 0

class MELADatasetWarped(Dataset):
    def __init__(self, degraded_path, gt_path, warped_noise_path): # , optical_flow_path):
        self.degraded_path = degraded_path
        self.gt_path = gt_path
        self.warped_noise_path = warped_noise_path
        # self.optical_flow_path = optical_flow_path

        # self.path = path
        with open(degraded_path, 'rb') as handle:
            degraded_data = pickle.load(handle)
        with open(gt_path, 'rb') as handle:
            gt_data = pickle.load(handle)

        # warped_noise: (num_frames, height, width, channels) = (100, 512, 512, 3)
        self.warped_noise_data = np.load(warped_noise_path, allow_pickle=True) # .permute(0, 3, 1, 2)
        self.warped_noise_data = np.load(warped_noise_path, allow_pickle=True).transpose(0, 3, 1, 2) # shape: (100, 3, 512, 512)
        # print(self.warped_noise_data.shape, self.optical_flow_data.shape)
        # (100, 128, 128, 3) (99, 2, 128, 128)

        self.degraded_data_list = degraded_data['train']['projections']
        self.gt_data_list = gt_data['train']['projections']

        # self.data_list = [file for file in os.listdir(path) if file.lower().endswith('.png')]
        # with open(path, 'rb') as handle:
        #     data = pickle.load(handle)
        # self.data_list = data['train']['projections']
        self.transform = transforms.ToTensor()  # Converts PIL image in the range [0, 255] to float tensor in the range [0., 1.]

    def __len__(self):
        return len(self.gt_data_list)

    def __getitem__(self, idx):
        degraded_image = normalize_array(self.degraded_data_list[idx])
        gt_image = normalize_array(self.gt_data_list[idx])
        warped_noise = self.warped_noise_data[idx] # shape: (H, W, 3)
        # print(np.min(warped_noise), np.max(warped_noise)) #, np.min(optical_flow), np.max(optical_flow))
        # exit()
        degraded_image = np.array(Image.fromarray(degraded_image*255).convert('RGB'))
        gt_image = np.array(Image.fromarray(gt_image*255).convert('RGB'))

        degraded_image = self.transform(degraded_image)
        gt_image = self.transform(gt_image)

        return degraded_image, gt_image, warped_noise, 0