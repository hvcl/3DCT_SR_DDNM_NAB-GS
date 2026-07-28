import os

from PIL import Image
import numpy as np
import cv2

from torchvision import transforms
from torch.utils.data import Dataset
import pickle

import torch
from pathlib import Path


normalize_array = lambda image: (image.astype(np.float32) - np.min(image)) / (np.max(image) - np.min(image))


def _load_projection_from_dir(output_dir, proj_idx):
    if not output_dir:
        return None

    output_dir = Path(output_dir)
    if output_dir.is_file():
        if output_dir.suffix in [".npy"]:
            arr = np.load(output_dir).astype(np.float32)
            return np.clip(arr, 0.0, 1.0)
        if output_dir.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            arr = np.array(Image.open(output_dir).convert("L"), dtype=np.float32) / 255.0
            return arr
    npy_path = output_dir / "pred_npy" / f"pred_{proj_idx:02d}.npy"
    if npy_path.exists():
        arr = np.load(npy_path).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    png_path = output_dir / "pred_png" / f"pred_{proj_idx:02d}.png"
    if png_path.exists():
        arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
        return arr

    return None

class MELADataset_Temporal(Dataset):
    def __init__(self, args, degraded_path, gt_path): #, warped_noise_path):
        self.args = args
        self.degraded_path = degraded_path
        self.gt_path = gt_path
        # self.warped_noise_path = warped_noise_path

        with open(degraded_path, 'rb') as handle:
            degraded_data = pickle.load(handle)
        with open(gt_path, 'rb') as handle:
            gt_data = pickle.load(handle)

        # self.warped_noise_data = np.load(warped_noise_path, allow_pickle=True) # .permute(0, 3, 1, 2)
        # self.warped_noise_data = np.load(warped_noise_path, allow_pickle=True).transpose(0, 3, 1, 2) # shape: (100, 3, 512, 512)

        self.degraded_data_list = degraded_data['train']['projections'] # length: 1000
        self.gt_data_list = gt_data['train']['projections'] # length: 100
        self.transform = transforms.ToTensor()

        # self.N_prev = self.args.N_proj_per_iter // 2
        # self.N_next = self.args.N_proj_per_iter // 2

        if "param_search" in self.args.image_folder:
            self.degraded_data_list = self.degraded_data_list[::20]
            self.gt_data_list = self.gt_data_list[::20]

    def __len__(self):
        return len(self.gt_data_list)

    # 기존 MELADataset_Temporal 클래스 내부의 __getitem__ 메서드를 수정합니다.
    def __getitem__(self, idx):
        deg_idx = int(idx*10) if "proj1000" in self.degraded_path else idx

        # if "noNorm" in self.args.image_folder:
        #     temp_deg = self.degraded_data_list[deg_idx]
        #     temp_gt = self.gt_data_list[idx]
        #     print(np.min(temp_deg), np.max(temp_deg))
        #     print(np.min(temp_gt), np.max(temp_gt))
        #     temp_deg = normalize_array(temp_deg)
        #     temp_gt = normalize_array(temp_gt)
        #     print(np.min(temp_deg), np.max(temp_deg))
        #     print(np.min(temp_gt), np.max(temp_gt))
        #     exit()
        
        degraded_image = normalize_array(self.degraded_data_list[deg_idx])
        gt_image = normalize_array(self.gt_data_list[idx])
        # warped_noise = self.warped_noise_data[idx]

        degraded_image = np.array(Image.fromarray(degraded_image*255).convert('RGB'))
        gt_image = np.array(Image.fromarray(gt_image*255).convert('RGB'))

        degraded_image = self.transform(degraded_image)
        gt_image = self.transform(gt_image)

        data = {
            "deg_img": degraded_image,
            "gt_img": gt_image,
            # "warped_noise": warped_noise,
        }

        # 이전 N_prev 개의 데이터 로드
        '''
        if idx > 0:
            for i in range(1, self.N_prev + 1):
                prev_idx = deg_idx - int(i*3)
                prev_degraded = normalize_array(self.degraded_data_list[prev_idx])
                prev_degraded = self.transform(np.array(Image.fromarray(prev_degraded*255).convert('RGB')))
                # prev_gt = normalize_array(self.gt_data_list[prev_idx])
                # prev_gt = self.transform(np.array(Image.fromarray(prev_gt*255).convert('RGB')))
                # prev_warped_noise = self.warped_noise_data[prev_idx]
                
                data[f"deg_img_prev_{i}"] = prev_degraded
                # data[f"gt_img_prev_{i}"] = prev_gt
                # data[f"warped_noise_prev_{i}"] = prev_warped_noise

        # 다음 N_next 개의 데이터 로드
        if idx < len(self.gt_data_list) -1:
            for i in range(1, self.N_next + 1):
                next_idx = deg_idx + int(i*3)
                next_degraded = normalize_array(self.degraded_data_list[next_idx])
                next_degraded = self.transform(np.array(Image.fromarray(next_degraded*255).convert('RGB')))
                # next_gt = normalize_array(self.gt_data_list[next_idx])
                # next_gt = self.transform(np.array(Image.fromarray(next_gt*255).convert('RGB')))
                # next_warped_noise = self.warped_noise_data[next_idx]
                
                data[f"deg_img_next_{i}"] = next_degraded
                # data[f"gt_img_next_{i}"] = next_gt
                # data[f"warped_noise_next_{i}"] = next_warped_noise
        '''

        return data


class MELADataset_Temporal_KVswap(Dataset):
    def __init__(self, args, degraded_path, gt_path, ref_output_dir=""):
        self.args = args
        self.degraded_path = degraded_path
        self.gt_path = gt_path
        self.ref_output_dir = ref_output_dir
        self.kvswap_mode = getattr(args, "kvswap_mode", "nabgs_target")

        with open(degraded_path, "rb") as handle:
            degraded_data = pickle.load(handle)
        with open(gt_path, "rb") as handle:
            gt_data = pickle.load(handle)

        self.degraded_data_list = degraded_data["train"]["projections"]
        self.gt_data_list = gt_data["train"]["projections"]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.gt_data_list)

    def __getitem__(self, idx):
        deg_idx = int(idx * 10) if "proj1000" in self.degraded_path else idx

        degraded_image = normalize_array(self.degraded_data_list[deg_idx])
        gt_image = normalize_array(self.gt_data_list[idx])
        gt_image = np.array(Image.fromarray(gt_image * 255).convert("RGB"))
        gt_tensor = self.transform(gt_image)

        ddnm_arr = _load_projection_from_dir(self.ref_output_dir, idx) if self.ref_output_dir else None
        if ddnm_arr is None:
            ddnm_arr = degraded_image

        if self.kvswap_mode == "ddnm_target":
            target_arr = np.clip(ddnm_arr, 0.0, 1.0)
            ref_arr = np.clip(degraded_image, 0.0, 1.0)
        else:
            target_arr = np.clip(degraded_image, 0.0, 1.0)
            ref_arr = np.clip(ddnm_arr, 0.0, 1.0)

        target_img = np.array(Image.fromarray((target_arr * 255).astype(np.uint8)).convert("RGB"))
        ref_img = np.array(Image.fromarray((ref_arr * 255).astype(np.uint8)).convert("RGB"))

        before_refine_img = np.array(Image.fromarray((np.clip(degraded_image, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB"))
        ddnm_first_img = np.array(Image.fromarray((np.clip(ddnm_arr, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB"))

        data = {
            "deg_img": self.transform(target_img),
            "gt_img": gt_tensor,
            "ref_img": self.transform(ref_img),
            "before_refine_img": self.transform(before_refine_img),
            "ddnm_first_img": self.transform(ddnm_first_img),
        }

        return data
