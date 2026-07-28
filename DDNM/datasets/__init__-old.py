# import os
import torch
# import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
# from datasets.celeba import CelebA
# from datasets.lsun import LSUN
# from datasets.MELA_temporal import MELADataset
from torch.utils.data import Subset
import numpy as np
# import torchvision
from PIL import Image
# from functools import partial

MY_DATA_PATH = '/workspace/miccai2024/data/CheX-ray14'

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_dataset(args, config):
    '''
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    '''

    # test_dataset = MELADataset(f"/workspace/Diffusion/DDNM/exp/datasets/{args.path_y}")
    # test_dataset = MELADataset(f"/workspace/naf_modified/data/{args.path_y}.pickle")
    # test_dataset = MELADataset(f"/workspace/DDNM/exp/datasets/{args.path_y}.pickle")

    # target path
    if "rmbed" in args.image_folder:
        gt_path = f"/workspace/data/pickle_data/mela_0050_rmbed.pickle"
        degraded_path = f"/workspace/data/pickle_data/mela_0050_rmbed_128.pickle"
        if "R2GS_DDNM_cycle" in args.image_folder:
            degraded_path = f"/workspace/data/pickle_data/R2GS_DDNM_cycle_consistency.pickle"
        from datasets.MELA_temporal import MELADataset_Temporal
        test_dataset = MELADataset_Temporal(args, degraded_path, gt_path) #, warped_noise_path)
        
    elif "MELA" in args.config:
        if ("temporal" in args.image_folder) or \
            ("FreeU" in args.image_folder) or \
            ("perProj" in args.image_folder) or \
            ("srFromUpsampled" in args.image_folder) or \
            ("gaussian_sinc_interpolation" in args.image_folder):
            from datasets.MELA_temporal import MELADataset_Temporal
        # elif "warp" in args.image_folder:
        #     from datasets.MELA import MELADatasetWarped
        else:
            from datasets.MELA import MELADataset
        # if "temporal_v2" in args.image_folder:
            # from datasets.MELA_temporal_v2 import MELADataset_Temporal_v2

        # GT path
        gt_path = f"/workspace/data/pickle_data/test_512_clamp_pickle/{args.path_y}.pickle"

        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/test_64_clamp_pickle/{args.path_y}.pickle"
            warped_noise_path = f"/workspace/DDNM/noise_warp/test_64_clamp_pickle/{args.path_y}/warped_noise_512x512.npy"
            # optical_flow_path = f"/workspace/DDNM/noise_warp/test_64_clamp_pickle/{args.path_y}/optical_flow_64x64.npy"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/test_128_clamp_pickle/{args.path_y}.pickle"
            warped_noise_path = f"/workspace/DDNM/noise_warp/test_128_clamp_pickle/{args.path_y}/warped_noise_512x512.npy"
            # optical_flow_path = f"/workspace/DDNM/noise_warp/test_128_clamp_pickle/{args.path_y}/optical_flow.npy"
            if "gaussian_sinc_interpolate_tri" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/{args.path_y}_gaussian_sinc_interpolation_128.pickle"
                # warped_noise_path = f"/workspace/data/pickle_data/mela_{args.path_y}_gaussian_sinc_interpolation_128/warped_noise_512x512.npy"

        # if "warp" in args.image_folder:
        #     test_dataset = MELADatasetWarped(degraded_path, gt_path, warped_noise_path) # , optical_flow_path)
        if ("temporal" in args.image_folder) or \
                ("FreeU" in args.image_folder) or \
                ("perProj" in args.image_folder) or \
                ("srFromUpsampled" in args.image_folder) or \
                ("gaussian_sinc_interpolation" in args.image_folder):
            # if "v2" in args.image_folder:
            #     data_path = f"/workspace/data/MELA_selected/test512_clamp/{args.path_y}.nii.gz"
            #     config_path = "/workspace/data_config/pickle/up512.yml"
            #     test_dataset = MELADataset_Temporal_v2(args, data_path, config_path, gt_path)
            # else:
            '''
            if "norm_noise" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/downsample_{int(args.deg_scale)}x_proj1000_axis_fixed/{args.path_y}.pickle"
            elif "noise_norm" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/downsample_{int(args.deg_scale)}x_noise_norm_proj1000_fixed/{args.path_y}.pickle"
            elif "downNoiseNorm" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/down_addNoise_{int(args.deg_scale)}x_proj1000/{args.path_y}.pickle"
            elif "srFromUpsampled" in args.image_folder:
                # test_512_tri_x4
                # test_512_tri_x8
                degraded_path = f"/workspace/data/pickle_data/test_512_tri_x{int(args.deg_scale)}/{args.path_y}.pickle"
            '''

            test_dataset = MELADataset_Temporal(args, degraded_path, gt_path) #, warped_noise_path)
            print("Temporal dataset")
        else:
            test_dataset = MELADataset(degraded_path, gt_path)

        # for debug
        if args.path_y == "GT512_MELA0522":
            degraded_path = f"/workspace/DDNM/exp/datasets/{args.path_y}.pickle"
            gt_path = f"/workspace/DDNM/exp/datasets/{args.path_y}.pickle"
            test_dataset = MELADataset(degraded_path, gt_path)
    elif "UHRCT" in args.config:
        from datasets.UHRCT import UHRCTDataset
        degraded_path = f"/workspace/data/pickle_data/UHRCT_256_64/{args.path_y}.pickle"
        gt_path = f"/workspace/data/pickle_data/UHRCT_256/{args.path_y}.pickle"
        if "trilinear" in args.image_folder:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_up256_tri/{args.path_y}.pickle"

        if "UHRCT512" in args.image_folder:
            gt_path = f"/workspace/data/pickle_data/UHRCT_gt512/{args.path_y}.pickle"
            degraded_path = f"/workspace/data/pickle_data/UHRCT_512_128/{args.path_y}.pickle"

        if "rmBed" in args.image_folder:
            gt_path = f"/workspace/data/pickle_data/UHRCT_512_rmbed/{args.path_y}.pickle"
            degraded_path = f"/workspace/data/pickle_data/UHRCT_512_128_rmbed/{args.path_y}.pickle"
            
        test_dataset = UHRCTDataset(args, degraded_path, gt_path)

    dataset = None
    
    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled: # True
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

# python main.py --ni --simplified --config CheX-ray14.yml --path_y NAF --eta 0.85 --deg sr_averagepooling --deg_scale 2 --sigma_y 0