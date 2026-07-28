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
    # target path
    degraded_override = getattr(args, "degraded_path", "")
    gt_override = getattr(args, "gt_path", "")

    if "MELA" in args.config:
        if "cycle1" in args.image_folder:
            gt_path = f"/workspace/data/pickle_data/MELA_GT_512_rmbed/mela_0050_rmbed.pickle"
            degraded_path = f"/workspace/data/pickle_data/NegAlpha_R2GS_DDNM_4x_rmbed.pickle"
            # if "3455" in args.image_folder:
            #     degraded_path = f"/workspace/data/pickle_data/R2GS_output_3455_switch.pickle"
            if args.deg_scale == 8.:
                degraded_path = f"/workspace/data/pickle_data/NegAlpha_R2GS_DDNM_8x_rmbed.pickle"
            from datasets.MELA_temporal import MELADataset_Temporal
            test_dataset = MELADataset_Temporal(args, degraded_path, gt_path)
            dataset = None
            return dataset, test_dataset
        
        if ("temporal" in args.image_folder) or \
            ("FreeU" in args.image_folder) or \
            ("perProj" in args.image_folder) or \
            ("srFromUpsampled" in args.image_folder) or \
            ("gaussian_sinc_interpolation" in args.image_folder):
            from datasets.MELA_temporal import MELADataset_Temporal
        else:
            from datasets.MELA_temporal import MELADataset_Temporal

        # GT path
        gt_path = f"/workspace/data/pickle_data/MELA_GT_512_rmbed/{args.path_y}_rmbed.pickle"
        if gt_override:
            gt_path = gt_override

        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_64_rmbed/{args.path_y}_rmbed.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/MELA_UP_64_512_rmbed/{args.path_y}_rmbed.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_128_rmbed/{args.path_y}_rmbed.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/MELA_UP_128_512_rmbed/{args.path_y}_rmbed.pickle"

        if degraded_override:
            degraded_path = degraded_override

        if  ("temporal" in args.image_folder) or \
            ("FreeU" in args.image_folder) or \
            ("perProj" in args.image_folder) or \
            ("srFromUpsampled" in args.image_folder) or \
            ("gaussian_sinc_interpolation" in args.image_folder):
            test_dataset = MELADataset_Temporal(args, degraded_path, gt_path) #, warped_noise_path)
            print("Temporal dataset")
        else:
            test_dataset = MELADataset_Temporal(args, degraded_path, gt_path)

    elif "UHRCT" in args.config:
        from datasets.UHRCT import UHRCTDataset
        gt_path = f"/workspace/data/pickle_data/UHRCT_GT_512_rmbed/{args.path_y}.pickle"
        if gt_override:
            gt_path = gt_override
        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_64_rmbed/{args.path_y}.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/UHRCT_UP_64_512_rmbed/{args.path_y}.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_128_rmbed/{args.path_y}.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/UHRCT_UP_128_512_rmbed/{args.path_y}.pickle"

        if degraded_override:
            degraded_path = degraded_override
        test_dataset = UHRCTDataset(args, degraded_path, gt_path)

    dataset = None
    
    return dataset, test_dataset

def get_dataset_cascade(args, config):
    if "MELA" in args.config:
        from datasets.MELA_canDenorm import MELADataset_canDenorm
        # GT path
        gt_path = f"/workspace/data/pickle_data/MELA_GT_512_rmbed/{args.path_y}_rmbed.pickle"

        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_64_rmbed/{args.path_y}_rmbed.pickle"
            # if "bicubic" in args.image_folder:
            #     degraded_path = f"/workspace/data/pickle_data/MELA_UP_64_512_rmbed/{args.path_y}_rmbed.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_128_rmbed/{args.path_y}_rmbed.pickle"
            # if "bicubic" in args.image_folder:
            #     degraded_path = f"/workspace/data/pickle_data/MELA_UP_128_512_rmbed/{args.path_y}_rmbed.pickle"
        test_dataset = MELADataset_canDenorm(args, degraded_path, gt_path) #, warped_noise_path)
        print("Load temporal dataset (datasets.MELA_normSelect.MELADatasetNormSelect)")

    elif "UHRCT" in args.config:
        from datasets.UHRCT import UHRCTDataset
        gt_path = f"/workspace/data/pickle_data/UHRCT_GT_512_rmbed/{args.path_y}.pickle"
        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_64_rmbed/{args.path_y}.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/UHRCT_UP_64_512_rmbed/{args.path_y}.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_128_rmbed/{args.path_y}.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/UHRCT_UP_128_512_rmbed/{args.path_y}.pickle"
        test_dataset = UHRCTDataset(args, degraded_path, gt_path)

    dataset = None
    
    return dataset, test_dataset

def get_dataset_kvswap(args, config):
    if "MELA" in args.config:
        from datasets.MELA_temporal import MELADataset_Temporal, MELADataset_Temporal_KVswap
        # GT path
        gt_path = f"/workspace/data/pickle_data/MELA_GT_512_rmbed/{args.path_y}_rmbed.pickle"

        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_64_rmbed/{args.path_y}_rmbed.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/MELA_UP_64_512_rmbed/{args.path_y}_rmbed.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_128_rmbed/{args.path_y}_rmbed.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/MELA_UP_128_512_rmbed/{args.path_y}_rmbed.pickle"
        degraded_override = getattr(args, "degraded_path", "")
        if degraded_override:
            degraded_path = degraded_override

        ref_output_dir = getattr(args, "ddnm_first_output_path", "")
        if ref_output_dir:
            test_dataset = MELADataset_Temporal_KVswap(args, degraded_path, gt_path, ref_output_dir)
        else:
            test_dataset = MELADataset_Temporal(args, degraded_path, gt_path) #, warped_noise_path)
        print("Temporal dataset")

    elif "UHRCT" in args.config:
        from datasets.UHRCT import UHRCTDataset
        gt_path = f"/workspace/data/pickle_data/UHRCT_GT_512_rmbed/{args.path_y}.pickle"
        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_64_rmbed/{args.path_y}.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/UHRCT_UP_64_512_rmbed/{args.path_y}.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_128_rmbed/{args.path_y}.pickle"
            if "bicubic" in args.image_folder:
                degraded_path = f"/workspace/data/pickle_data/UHRCT_UP_128_512_rmbed/{args.path_y}.pickle"
        test_dataset = UHRCTDataset(args, degraded_path, gt_path)

    dataset = None
    
    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X, clip_max=1.0):
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


def inverse_data_transform(config, X, clip_max=1.0):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, clip_max)

def get_dataset_miccai26(args, config):
    """
    MICCAI26 dataset loading without image_folder based conditions.
    Uses explicit arguments (--ddnm_orig, --startStep) instead.
    """
    degraded_override = getattr(args, "degraded_path", "")
    gt_override = getattr(args, "gt_path", "")

    if "MELA" in args.config:
        from datasets.MELA_temporal import MELADataset_Temporal
        # GT path
        gt_path = f"/workspace/data/pickle_data/MELA_GT_512_rmbed/{args.path_y}_rmbed.pickle"
        if gt_override:
            gt_path = gt_override

        # 8x
        if args.deg_scale == 8. or (args.cgls_path is not None and args.deg_scale == 4.):
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_64_rmbed/{args.path_y}_rmbed.pickle"
        # 4x
        elif args.deg_scale == 4. or (args.cgls_path is not None and args.deg_scale == 2.):
            degraded_path = f"/workspace/data/pickle_data/MELA_GT_128_rmbed/{args.path_y}_rmbed.pickle"
        else:
            raise ValueError(f"Unsupported deg_scale: {args.deg_scale}")

        if degraded_override:
            degraded_path = degraded_override
        
        test_dataset = MELADataset_Temporal(args, degraded_path, gt_path)
        print(f"MICCAI26 dataset (MELA, deg_scale={args.deg_scale}x, degraded_path={degraded_path}, gt_path={gt_path})")

    elif "UHRCT" in args.config:
        from datasets.UHRCT import UHRCTDataset
        gt_path = f"/workspace/data/pickle_data/UHRCT_GT_512_rmbed/{args.path_y}.pickle"
        if gt_override:
            gt_path = gt_override
        # 8x
        if args.deg_scale == 8.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_64_rmbed/{args.path_y}.pickle"
        # 4x
        elif args.deg_scale == 4.:
            degraded_path = f"/workspace/data/pickle_data/UHRCT_GT_128_rmbed/{args.path_y}.pickle"
        else:
            raise ValueError(f"Unsupported deg_scale: {args.deg_scale}")

        if degraded_override:
            degraded_path = degraded_override
        
        test_dataset = UHRCTDataset(args, degraded_path, gt_path)
        print(f"MICCAI26 dataset (UHRCT, deg_scale={args.deg_scale}x, degraded_path={degraded_path}, gt_path={gt_path})")
    else:
        raise ValueError(f"Unsupported config: {args.config}")

    dataset = None
    return dataset, test_dataset

# python main.py --ni --simplified --config CheX-ray14.yml --path_y NAF --eta 0.85 --deg sr_averagepooling --deg_scale 2 --sigma_y 0
