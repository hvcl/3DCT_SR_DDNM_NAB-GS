import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))

class MELADataset_canDenorm(Dataset):
    def __init__(self, args, degraded_path, gt_path): #, warped_noise_path):
        self.args = args
        self.degraded_path = degraded_path
        self.gt_path = gt_path

        with open(degraded_path, 'rb') as handle:
            degraded_data = pickle.load(handle)
        with open(gt_path, 'rb') as handle:
            gt_data = pickle.load(handle)

        self.degraded_data = degraded_data['train']['projections'] # length: 100
        self.gt_data = gt_data['train']['projections'] # length: 100

        # print(self.degraded_data.shape) # (100, 512, 512)

        # # save deg projections after min-max normalization
        # import os
        # os.makedirs("./temp", exist_ok=True)
        # os.makedirs("./temp/deg", exist_ok=True)
        # os.makedirs("./temp/gt", exist_ok=True)
        # for idx in range(len(self.degraded_data)):
        #     deg_img = normalize_array(self.degraded_data[idx]) * 255.0
        #     deg_img = deg_img.astype(np.uint8)
        #     deg_pil = Image.fromarray(deg_img)
        #     deg_pil.save(f"./temp/deg/deg_{idx:03d}.png")
        # for idx in range(len(self.gt_data)):
        #     gt_img = normalize_array(self.gt_data[idx]) * 255.0
        #     gt_img = gt_img.astype(np.uint8)
        #     gt_pil = Image.fromarray(gt_img)
        #     gt_pil.save(f"./temp/gt/gt_{idx:03d}.png")
        # exit()

        # cache volume-level arrays and compute volume-level normalization functions
        self.deg_data_all = np.array(self.degraded_data).astype(np.float32)  # [N, h, w]
        self.gt_data_all = np.array(self.gt_data).astype(np.float32)         # [N, H, W]

        # volume-level min/max (fall back to safe eps to avoid div0)
        self.deg_vol_min = float(np.nanmin(self.deg_data_all))
        self.deg_vol_max = float(np.nanmax(self.deg_data_all))
        self.gt_vol_min = float(np.nanmin(self.gt_data_all))
        self.gt_vol_max = float(np.nanmax(self.gt_data_all))
        self._eps = 1e-12

        def _make_norm_fn(vmin, vmax):
            denom = (vmax - vmin) if (vmax - vmin) > self._eps else 1.0
            return lambda img: ((img.astype(np.float32) - vmin) / denom).astype(np.float32)

        self.deg_norm_vol_function = _make_norm_fn(self.deg_vol_min, self.deg_vol_max)
        self.gt_norm_vol_function = _make_norm_fn(self.gt_vol_min, self.gt_vol_max)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.gt_data)

    # 기존 MELADataset_Temporal 클래스 내부의 __getitem__ 메서드를 수정합니다.
    def __getitem__(self, idx):
        # per-projection (local) normalization - keeps previous behavior
        deg_local = normalize_array(self.degraded_data[idx])
        gt_local = normalize_array(self.gt_data[idx])

        # volume-level normalization images (use cached volume-level min/max)
        deg_vol_norm = self.deg_norm_vol_function(self.degraded_data[idx])
        gt_vol_norm = self.gt_norm_vol_function(self.gt_data[idx])

        # Inputs are already normalized to 0..1 floats.
        # Create tensors directly to avoid PIL/ToTensor scaling issues.
        deg_tensor = torch.from_numpy(deg_local.astype(np.float32)).unsqueeze(0)       # (1,H,W)
        gt_tensor = torch.from_numpy(gt_local.astype(np.float32)).unsqueeze(0)
        deg_vol_tensor = torch.from_numpy(deg_vol_norm.astype(np.float32)).unsqueeze(0)
        gt_vol_tensor = torch.from_numpy(gt_vol_norm.astype(np.float32)).unsqueeze(0)

        # If downstream expects 3-channel RGB, expand channel dim by repeating
        deg_tensor = deg_tensor.repeat(3, 1, 1)
        gt_tensor = gt_tensor.repeat(3, 1, 1)
        deg_vol_tensor = deg_vol_tensor.repeat(3, 1, 1)
        gt_vol_tensor = gt_vol_tensor.repeat(3, 1, 1)

        data = {
            "deg_img": deg_tensor,                       # per-projection normalized tensor
            "gt_img": gt_tensor,                         # per-projection normalized tensor
            # "deg_min_max": (float(np.min(self.degraded_data[idx])), float(np.max(self.degraded_data[idx]))),
            # "gt_min_max": (float(np.min(self.gt_data[idx])), float(np.max(self.gt_data[idx]))),
            # volume-normalized projection (tensor, 0..1)
            # "deg_norm_vol": deg_vol_tensor,
            # "gt_norm_vol": gt_vol_tensor,
            # also expose the volume-level mins/maxs if caller needs them
            # "deg_vol_min_max": (self.deg_vol_min, self.deg_vol_max),
            # "gt_vol_min_max": (self.gt_vol_min, self.gt_vol_max)
        }

        return data
