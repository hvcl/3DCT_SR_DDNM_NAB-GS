import os

from PIL import Image
import numpy as np
import pickle

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class Chest50Dataset(Dataset):
    def __init__(self):
        with open('/workspace/miccai2024/data/NAF/chest_50.pickle', "rb") as handle:
            data = pickle.load(handle)
        self.data = data['train']['projections']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = np.array(self.data[idx]).repeat(2, axis=0).repeat(2, axis=1)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        image = torch.Tensor(d).unsqueeze(0)

        return image, 0
    
    
class CXRDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data_list = [file for file in os.listdir(path) if file.lower().endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        # Load the image
        image_path = os.path.join(self.path, file_name)
        image = Image.open(image_path)

        # Apply transformations (resize and convert to tensor)
        image = self.transform(image)

        return image, 0