import os
import math
import csv
import torch
import torch.utils.data as data
import numpy as np
import random
from scipy.linalg import orth
import tqdm
from PIL import Image
import cv2
from pathlib import Path
import pickle

from datasets import data_transform, inverse_data_transform
from datasets.tmi26 import get_dataset_tmi26
from functions.ckpt_util import get_ckpt_path, download

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
import matplotlib.pyplot as plt

import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

def get_linear_adaptive_threshold(proj_idx, total_projs=100, thr_min=5.0, thr_max=6.5):
    """
    proj_idx: 현재 투영 인덱스 (0~99)
    thr_min: Frontal 구간에서의 최소 임계값
    thr_max: Lateral 구간에서의 최대 임계값
    """
    mid_idx = total_projs // 2  # 보통 50번째 인덱스
    
    if proj_idx <= mid_idx:
        # 0 -> 50: 선형 증가
        slope = (thr_max - thr_min) / mid_idx
        adaptive_thr = thr_min + slope * proj_idx
    else:
        # 50 -> 100: 선형 감소
        slope = (thr_max - thr_min) / (total_projs - mid_idx)
        adaptive_thr = thr_max - slope * (proj_idx - mid_idx)
        
    return adaptive_thr

def get_adaptive_threshold(proj_idx, l2thr_min, l2thr_max, total_projs=100):
    """
    Compute adaptive l2 threshold based on projection angle.
    Lateral (90 degrees) has highest threshold, Frontal (0/180 degrees) has lowest.
    
    Args:
        proj_idx: projection index (0 to total_projs-1)
        l2thr_min: minimum threshold (at frontal views, 0/180 degrees)
        l2thr_max: maximum threshold (at lateral view, 90 degrees)
        total_projs: total number of projections (default 100)
    
    Returns:
        adaptive_threshold: float value between l2thr_min and l2thr_max
    """
    # Convert projection index to angle in radians (0 to pi)
    angle_rad = (proj_idx / total_projs) * np.pi
    
    # Sin function peaks at 90 degrees (pi/2), creating smooth variation
    # At 0 and 180 degrees: sin(0) = 0, sin(pi) = 0 -> use min threshold
    # At 90 degrees: sin(pi/2) = 1 -> use max threshold
    base_thr = l2thr_min
    offset = l2thr_max - l2thr_min
    adaptive_thr = base_thr + offset * np.sin(angle_rad)
    
    return adaptive_thr

def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def load_first_output_projection(first_output_dir, proj_idx):
    if not first_output_dir:
        return None

    first_output_dir = Path(first_output_dir)
    if first_output_dir.is_file() and first_output_dir.suffix in [".pickle", ".pkl"]:
        with open(first_output_dir, "rb") as f:
            obj = pickle.load(f)
        train = obj.get("train", obj)
        for key in ["up_projections", "projections"]:
            if key in train:
                arr = np.asarray(train[key][proj_idx]).astype(np.float32)
                arr = arr - float(arr.min())
                arr = arr / max(float(arr.max()), 1e-8)
                return np.clip(arr, 0.0, 1.0)

    npy_path = first_output_dir / "pred_npy" / f"pred_{proj_idx:02d}.npy"
    if npy_path.exists():
        arr = np.load(npy_path).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    png_path = first_output_dir / "pred_png" / f"pred_{proj_idx:02d}.png"
    if png_path.exists():
        arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
        return arr

    return None


def build_first_output_batch(first_output_dir, start_idx, batch_size, channels, device, fallback_tensor=None):
    first_refs = []
    missing_first = []
    for batch_i in range(batch_size):
        proj_idx = start_idx + batch_i
        first_output_arr = load_first_output_projection(first_output_dir, proj_idx)
        if first_output_arr is None:
            missing_first.append(proj_idx)
            if fallback_tensor is None:
                continue
            first_output_arr = fallback_tensor[batch_i, 0].detach().cpu().numpy()
        first_tensor = torch.from_numpy(first_output_arr).float().to(device)
        first_tensor = first_tensor.unsqueeze(0).repeat(channels, 1, 1)
        first_refs.append(first_tensor)
    return first_refs, missing_first


def build_neighbor_output_batch(output_dir, start_idx, batch_size, channels, device, total_count=None, fallback_tensor=None):
    refs = []
    missing = []
    for batch_i in range(batch_size):
        proj_idx = start_idx + batch_i
        neighbor_idx = proj_idx + 1 if proj_idx == 0 else proj_idx - 1
        if total_count is not None:
            neighbor_idx = max(0, min(total_count - 1, neighbor_idx))
        arr = load_first_output_projection(output_dir, neighbor_idx)
        if arr is None:
            missing.append(neighbor_idx)
            if fallback_tensor is None:
                continue
            arr = fallback_tensor[batch_i, 0].detach().cpu().numpy()
        tensor = torch.from_numpy(arr).float().to(device)
        tensor = tensor.unsqueeze(0).repeat(channels, 1, 1)
        refs.append(tensor)
    return refs, missing


def save_step_reference_batch(base_dir, name, batch_tensor, idx_start):
    os.makedirs(base_dir, exist_ok=True)
    if batch_tensor is None:
        return

    name_dir = os.path.join(base_dir, name)
    os.makedirs(name_dir, exist_ok=True)

    batch = batch_tensor.detach().cpu()
    for batch_i in range(batch.shape[0]):
        proj_idx = idx_start + batch_i
        proj_dir = os.path.join(name_dir, f"proj_{proj_idx:02d}")
        os.makedirs(proj_dir, exist_ok=True)
        img_1ch = batch[batch_i, 0]
        np.save(os.path.join(proj_dir, "reference.npy"), img_1ch.numpy().astype(np.float32))
        tvu.save_image(img_1ch, os.path.join(proj_dir, "reference.png"))

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def warp_noise_with_flow(prev_img, curr_img, prev_noise, alpha=0.1):
    """
    Warp previous noise to current frame using optical flow.
    Motion-based selective replacement: only replace regions with significant motion.
    
    Args:
        prev_img: [H, W] numpy array (upsampled LR projection, [0,1] range)
        curr_img: [H, W] numpy array (current upsampled LR projection, [0,1] range)
        prev_noise: [1, C, H, W] torch tensor (initial fixed noise)
        alpha: float, motion threshold percentile (default 0.1 = top 10% motion regions)
    
    Returns:
        combined_noise: [1, C, H, W] torch tensor
    """
    # Convert to uint8 for optical flow computation
    prev_gray = (prev_img * 255).astype(np.uint8)
    curr_gray = (curr_img * 255).astype(np.uint8)
    
    # Compute optical flow using Farneback algorithm
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Compute flow magnitude (motion intensity)
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)  # [H, W]
    
    # Identify regions with significant motion
    # Use percentile threshold: top alpha% of motion
    threshold = np.percentile(flow_magnitude, (1.0 - alpha) * 100)
    motion_mask = flow_magnitude > threshold  # [H, W] boolean mask
    
    # Convert prev_noise to numpy for processing
    noise_np = prev_noise.squeeze().cpu().numpy()  # [C, H, W] or [H, W]
    
    # Create fresh Gaussian noise for motion regions
    fresh_noise = np.random.randn(*noise_np.shape)
    
    # Apply motion mask: replace only high-motion regions with fresh noise
    if len(noise_np.shape) == 2:  # Single channel
        # Ensure Gaussian properties: mean=0, std=1
        fresh_noise = (fresh_noise - fresh_noise.mean()) / (fresh_noise.std() + 1e-8)
        combined_noise = noise_np.copy()
        combined_noise[motion_mask] = fresh_noise[motion_mask]
        combined_noise = combined_noise[np.newaxis, ...]  # Add channel dimension
    else:  # Multi-channel
        # Process each channel independently
        combined_noise = []
        for c_idx in range(noise_np.shape[0]):
            fresh_c = fresh_noise[c_idx]
            fresh_c = (fresh_c - fresh_c.mean()) / (fresh_c.std() + 1e-8)
            combined_c = noise_np[c_idx].copy()
            combined_c[motion_mask] = fresh_c[motion_mask]
            combined_noise.append(combined_c)
        combined_noise = np.stack(combined_noise)
    
    # Convert back to torch tensor
    combined_noise = torch.from_numpy(combined_noise).to(prev_noise.device)
    combined_noise = combined_noise.view_as(prev_noise)
    
    # check std and mean of combined noise
    print(f"Combined noise mean: {combined_noise.mean().item():.4f}, std: {combined_noise.std().item():.4f}")

    return combined_noise

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        betas = []
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            beta = min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999)
            betas.append(beta * (beta_end - beta_start) + beta_start)
        return np.array(betas)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # Use MICCAI26 version
        from functions.svd_ddnm_miccai26 import ddnm_diffusion, ddnm_plus_diffusion
        self.ddnm_diffusion = ddnm_diffusion
        self.ddnm_plus_diffusion = ddnm_plus_diffusion

        if self.args.cgls_path is not None:
            self.cgls_data = np.load(self.args.cgls_path)  # [N, 512, 512], range [0, 1]
        
        # Pre-generate shared noise if flag is set
        self.shared_noise = None
        if getattr(self.args, 'shared_noise', False):
            # Noise will be initialized when we know the shape in the main loop
            # For now, just mark that we need to use shared noise
            self.use_shared_noise = True
        else:
            self.use_shared_noise = False

    def _save_noise_visualization(self, x_noise, proj_idx, config):
        """
        Save noise visualization as PNG for analysis of noise warping.
        
        Args:
            x_noise: [batch_size, C, H, W] torch tensor with noise
            proj_idx: projection index (for numbering)
            config: config object
        """
        try:
            # Process first element in batch
            noise = x_noise[0]  # [C, H, W]
            
            # Average across channels to get single channel
            if noise.shape[0] > 1:
                noise_viz = noise.mean(dim=0)  # [H, W]
            else:
                noise_viz = noise[0]  # [H, W]
            
            # Normalize to [0, 1] range
            noise_np = noise_viz.detach().cpu().numpy()
            noise_min = noise_np.min()
            noise_max = noise_np.max()
            if noise_max > noise_min:
                noise_norm = (noise_np - noise_min) / (noise_max - noise_min)
            else:
                noise_norm = np.zeros_like(noise_np)
            
            # Convert to uint8 [0, 255]
            noise_uint8 = (noise_norm * 255).astype(np.uint8)
            
            # Save as grayscale PNG
            output_path = os.path.join(self.args.image_folder, f"warped_noise/noise_proj_{proj_idx:02d}.png")
            cv2.imwrite(output_path, noise_uint8)
        except Exception as e:
            print(f"Warning: Failed to save noise visualization for proj_idx {proj_idx}: {e}")

    def sample(self, simplified):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)
            ckpt = "/workspace/DDNM/exp/logs/totensor/ckpt.pth"
            print("Loading checkpoint {}".format(ckpt))
            
            if 'totensor' in ckpt or 'CheXpert_8' in ckpt:
                model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(ckpt, map_location=self.device)[0])
            
        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)

            if self.config.model.use_fp16:
                model.convert_to_fp16()
            
            if self.args.ckpt == "MICCAI25":
                ckpt = "/workspace/DDNM/exp/logs/chestx14/ema_0.9999_300000.pt"
            elif self.args.ckpt == "SIDE":
                ckpt = "/workspace/DDNM/exp/logs/CheX-ray14_CheXpert_512x512-2gb/ema_0.9999_620000.pt"
            elif self.args.ckpt == "680k":
                ckpt = "/tmp/openai-2026-01-06-10-36-16-194901/ema_0.9999_680000.pt"
            elif self.args.ckpt == "700k":
                ckpt = "/tmp/openai-2026-01-06-10-36-16-194901/ema_0.9999_700000.pt"
            elif self.args.ckpt == "750k":
                ckpt = "/workspace/improved-diffusion/checkpoint/gpu4_2gb_model/ema_0.9999_750000.pt"

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn
            
        if simplified:
            print('Run Simplified DDNM, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            result = self.simplified_ddnm_plus(model, cls_fn)
        else: # here
            print('Run SVD-based DDNM.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            result = self.svd_based_ddnm_plus(model, cls_fn)
            
        np.save(os.path.join(self.args.image_folder, "whole.npy"), np.array(result))
        return result

            
    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset_tmi26(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        print("args.deg:",args.deg)
        if args.deg =='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)
        elif args.deg =='denoising':
            A = lambda z: z
            Ap = A
        elif args.deg == 'fourier':
            A = lambda z: z
            Ap = A
        elif args.deg =='sr_averagepooling':
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((512//scale,512//scale))
            Ap = lambda z: MeanUpsample(z,scale)
        elif args.deg =='inpainting':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A = lambda z: z*mask
            Ap = A
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_before_psnr = 0.0
        avg_first_psnr = 0.0
        avg_mse = 0.0
        avg_before_mse = 0.0
        avg_first_mse = 0.0
        first_metric_count = 0
        pbar = tqdm.tqdm(val_loader)
        whole_result = []
        metric_rows = []
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if config.sampling.batch_size!=1:
                y = y + torch.randn_like(y) * sigma_y if sigma_y > 0. else y
            
            # Generate or reuse shared noise
            if self.use_shared_noise:
                if self.shared_noise is None:
                    # Initialize shared noise on first iteration
                    self.shared_noise = torch.randn(
                        1,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                x = self.shared_noise.expand(x_orig.shape[0], -1, -1, -1).clone()
            else:
                x = torch.randn(
                    x_orig.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

            # xt ~ q(xt|x0)
            x = x * torch.sqrt(self.alphas_cumprod_prev).view(-1, 1, 1, 1) + x_orig * torch.sqrt(1. - self.alphas_cumprod_prev).view(-1, 1, 1, 1)

            # Simplified (without SVD)
            result = self.ddnm_diffusion(
                x, model, self.betas, args.eta, None, y, 
                temp_y=y, cls_fn=cls_fn, classes=classes, 
                config=config, args=args
            )
            
            whole_result.append(result)
            
        return whole_result    

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset_tmi26(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'fourier':
            from functions.fourier_deg import FourierGuidedOperator
            from functions.svd_operators import SuperResolution
            inter_radius = float(getattr(args, "fourier_inter_radius", 30.0))
            intra_radius = float(getattr(args, "fourier_intra_radius", 45.0))
            transition_width = float(getattr(args, "fourier_transition_width", 8.0))
            base_operator = SuperResolution(config.data.channels, config.data.image_size, int(args.deg_scale), self.device)
            A_funcs = FourierGuidedOperator(
                base_operator,
                config.data.channels,
                self.config.data.image_size,
                self.device,
                inter_radius,
                intra_radius,
                transition_width=transition_width,
            )
        elif deg == 'fourier_deblur_aniso':
            from functions.fourier_deg import FourierGuidedOperator, GrayChannelOperator
            from functions.svd_operators import Deblurring2D
            inter_radius = float(getattr(args, "fourier_inter_radius", 30.0))
            intra_radius = float(getattr(args, "fourier_intra_radius", 45.0))
            transition_width = float(getattr(args, "fourier_transition_width", 8.0))
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            base_operator = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 1, self.config.data.image_size, self.device)
            base_operator = GrayChannelOperator(base_operator, config.data.channels, self.config.data.image_size, self.device)
            A_funcs = FourierGuidedOperator(
                base_operator,
                config.data.channels,
                self.config.data.image_size,
                self.device,
                inter_radius,
                intra_radius,
                transition_width=transition_width,
            )
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling': # here
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.tmi26_operators import DeblurringColor
            sigma = float(getattr(args, "blur_sigma", 1.0))
            kernel_size = int(getattr(args, "blur_kernel_size", 5))
            if kernel_size % 2 == 0:
                raise ValueError(f"blur_kernel_size must be odd, got {kernel_size}")
            radius = kernel_size // 2
            offsets = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
            A_funcs = DeblurringColor(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.fourier_deg import GrayChannelOperator
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(
                kernel1 / kernel1.sum(),
                kernel2 / kernel2.sum(),
                1,
                self.config.data.image_size,
                self.device,
            )
            A_funcs = GrayChannelOperator(A_funcs, config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'highfreq':
            from functions.highFrequency_deg import HighFreqDegradation
            cutoff_radius = float(getattr(args, "cutoff_radius", 32.0))
            hf_boost = float(getattr(args, "hf_boost", 1.0))
            A_funcs = HighFreqDegradation(
                config.data.channels,
                config.data.image_size,
                cutoff_radius,
                self.device,
                hf_boost=hf_boost,
            )
        elif deg == 'detail_enhance_lowpass':
            from functions.detail_enhance_deg import DetailEnhanceLowpass

            cutoff_radius = float(getattr(args, "cutoff_radius", 32.0))
            hf_boost = float(getattr(args, "hf_boost", 1.0))
            A_funcs = DetailEnhanceLowpass(
                config.data.channels,
                config.data.image_size,
                cutoff_radius,
                self.device,
                highfreq_gain=hf_boost,
            )
        elif deg == 'detail_enhance_gaussian':
            from functions.detail_enhance_deg import DetailEnhanceGaussian

            sigma = float(getattr(args, "blur_sigma", 1.2))
            hf_boost = float(getattr(args, "hf_boost", 1.0))
            A_funcs = DetailEnhanceGaussian(
                config.data.channels,
                config.data.image_size,
                sigma,
                self.device,
                highfreq_gain=hf_boost,
            )
        elif deg == 'detail_enhance_lowref':
            from functions.detail_enhance_deg import DetailEnhanceLowRef

            sigma = float(getattr(args, "band_sigma_low", getattr(args, "blur_sigma", 2.0)))
            low_anchor_strength = float(getattr(args, "low_anchor_strength", 1.0))
            A_funcs = DetailEnhanceLowRef(
                config.data.channels,
                config.data.image_size,
                sigma,
                self.device,
                low_anchor_strength=low_anchor_strength,
            )
        elif deg == 'detail_enhance_bandpass' or deg == 'detail_enhance_bandpass_ref':
            from functions.detail_enhance_deg import DetailEnhanceBandpass

            sigma_low = float(getattr(args, "band_sigma_low", 2.0))
            sigma_mid = float(getattr(args, "band_sigma_mid", 1.0))
            hf_boost = float(getattr(args, "hf_boost", 1.0))
            low_anchor_strength = float(getattr(args, "low_anchor_strength", 1.0))
            A_funcs = DetailEnhanceBandpass(
                config.data.channels,
                config.data.image_size,
                sigma_low,
                sigma_mid,
                self.device,
                mid_gain=hf_boost,
                low_anchor_strength=low_anchor_strength,
            )
        else:
            raise ValueError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        if getattr(args, "sigma_y_min", -1.0) >= 0:
            args.sigma_y_min = 2 * args.sigma_y_min
        if getattr(args, "sigma_y_max", -1.0) >= 0:
            args.sigma_y_max = 2 * args.sigma_y_max
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_before_psnr = 0.0
        avg_first_psnr = 0.0
        avg_mse = 0.0
        avg_before_mse = 0.0
        avg_first_mse = 0.0
        first_metric_count = 0
        pbar = tqdm.tqdm(val_loader)
        whole_result = []
        metric_rows = []

        start_steps = []
        
        # For warped noise: track previous noise and upsampled LR
        prev_noise = None
        prev_upsampled_lr = None
        prev_projection_output = None
        use_warped_noise = self.use_shared_noise and getattr(self.args, 'warped_noise', False)

        for idx, ITER_DATA in enumerate(pbar):
            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 512, 512]

            x_deg = x_deg.to(self.device)
            x_orig = x_orig.to(self.device)

            before_refine_imgs = x_deg.detach().clone()
            y = data_transform(self.config, x_deg)
            x_orig = data_transform(self.config, x_orig)
            fourier_ref_batch = None
            fourier_intra_batch = None

            if deg == 'detail_enhance_lowref':
                first_refs = []
                missing_first = []
                low_ref_alpha = float(getattr(self.args, "low_ref_alpha", 0.5))
                for batch_i in range(x_deg.shape[0]):
                    proj_idx = idx_so_far + batch_i
                    first_output_arr = load_first_output_projection(getattr(self.args, "ddnm_first_output_path", ""), proj_idx)
                    if first_output_arr is None:
                        missing_first.append(proj_idx)
                        first_output_arr = before_refine_imgs[batch_i, 0].detach().cpu().numpy()
                    first_tensor = torch.from_numpy(first_output_arr).float().to(self.device)
                    first_tensor = first_tensor.unsqueeze(0).repeat(x_deg.shape[1], 1, 1)
                    first_refs.append(first_tensor)
                if missing_first:
                    print(f"[detail_enhance_lowref] Missing first-stage projections for indices {missing_first}; falling back to before_refine.")
                first_ref_batch = torch.stack(first_refs, dim=0)
                first_ref_batch = data_transform(self.config, first_ref_batch)
                y = A_funcs.compose_low_target(y, first_ref_batch, low_ref_alpha=low_ref_alpha)

            if deg == 'detail_enhance_bandpass_ref':
                first_refs = []
                missing_first = []
                for batch_i in range(x_deg.shape[0]):
                    proj_idx = idx_so_far + batch_i
                    first_output_arr = load_first_output_projection(getattr(self.args, "ddnm_first_output_path", ""), proj_idx)
                    if first_output_arr is None:
                        missing_first.append(proj_idx)
                        first_output_arr = before_refine_imgs[batch_i, 0].detach().cpu().numpy()
                    first_tensor = torch.from_numpy(first_output_arr).float().to(self.device)
                    first_tensor = first_tensor.unsqueeze(0).repeat(x_deg.shape[1], 1, 1)
                    first_refs.append(first_tensor)
                if missing_first:
                    print(f"[detail_enhance_bandpass_ref] Missing first-stage projections for indices {missing_first}; falling back to before_refine.")
                first_ref_batch = torch.stack(first_refs, dim=0)
                first_ref_batch = data_transform(self.config, first_ref_batch)
                y = A_funcs.compose_from_refs(y, first_ref_batch)

            if deg in ['fourier', 'fourier_deblur_aniso']:
                inter_refs, missing_inter = build_neighbor_output_batch(
                    getattr(self.args, "fourier_inter_ref_path", ""),
                    idx_so_far,
                    x_deg.shape[0],
                    x_deg.shape[1],
                    self.device,
                    total_count=args.subset_end if args.subset_end > 0 else None,
                    fallback_tensor=before_refine_imgs,
                )
                intra_refs, missing_intra = build_first_output_batch(
                    getattr(self.args, "ddnm_first_output_path", ""),
                    idx_so_far,
                    x_deg.shape[0],
                    x_deg.shape[1],
                    self.device,
                    fallback_tensor=before_refine_imgs,
                )
                if inter_refs:
                    fourier_ref_batch = torch.stack(inter_refs, dim=0)
                    fourier_ref_batch = F.interpolate(
                        fourier_ref_batch,
                        size=(self.config.data.image_size, self.config.data.image_size),
                        mode='bilinear',
                        align_corners=False,
                    )
                    fourier_ref_batch = data_transform(self.config, fourier_ref_batch)
                if intra_refs:
                    fourier_intra_batch = torch.stack(intra_refs, dim=0)
                    fourier_intra_batch = F.interpolate(
                        fourier_intra_batch,
                        size=(self.config.data.image_size, self.config.data.image_size),
                        mode='bilinear',
                        align_corners=False,
                    )
                    fourier_intra_batch = data_transform(self.config, fourier_intra_batch)
                if missing_inter:
                    print(f"[{deg}] Missing inter-slice references for indices {missing_inter}; falling back to degraded input.")
                if missing_intra:
                    print(f"[{deg}] Missing intra-slice DDNM-first references for indices {missing_intra}; falling back to degraded input.")

            save_formula = getattr(args, "save_ddnm_formula", False) and idx_so_far == getattr(args, "formula_proj_idx", 0)
            formula_dir = os.path.join(self.args.image_folder, "ddnm_formula", f"proj_{idx_so_far:02d}")
            if save_formula:
                os.makedirs(formula_dir, exist_ok=True)
                y_1ch = x_deg[0, 0].detach().cpu()
                tvu.save_image(y_1ch, os.path.join(formula_dir, "y.png"))
                np.save(os.path.join(formula_dir, "y.npy"), y_1ch.numpy())

            b, c, h, w = y.size()
            hwc = c * h * w
            
            if self.args.add_noise: # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y)
            
            # temp_y used inside DDNM sampling. By default this is the upsampled degraded input,
            # but we can optionally swap it to the first-stage DDNM output while keeping
            # measurement updates on the degraded observation y.
            temp_y_original = torch.clone(y).to(self.device)
            if getattr(self.args, "always_noised_ddnm_first", False):
                first_refs, missing_first = build_first_output_batch(
                    getattr(self.args, "ddnm_first_output_path", ""),
                    idx_so_far,
                    x_deg.shape[0],
                    x_deg.shape[1],
                    self.device,
                    fallback_tensor=before_refine_imgs,
                )
                if first_refs:
                    first_ref_batch = torch.stack(first_refs, dim=0)
                    temp_y_original = data_transform(self.config, first_ref_batch)
                if missing_first:
                    print(
                        f"[always_noised_ddnm_first] Missing first-stage projections for indices {missing_first}; falling back to degraded input for those projections."
                    )
            if getattr(args, "save_step_outputs", False):
                step_dir = os.path.join(self.args.image_folder, "step_analysis")
                sr_ref_batch = F.interpolate(
                    before_refine_imgs,
                    size=(self.config.data.image_size, self.config.data.image_size),
                    mode='bilinear',
                    align_corners=False,
                )
                save_step_reference_batch(step_dir, "sr_upsampled_ref", sr_ref_batch, idx_so_far)
            # Keep an upsampled LR copy for noise warping before flattening
            lr_for_warp = None
            if use_warped_noise:
                lr_for_warp = F.interpolate(
                    temp_y_original,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False,
                )
            
            # Load CGLS if path is provided for diffusion initialization
            if self.args.cgls_path is not None:
                cgls_slice = torch.from_numpy(self.cgls_data[idx_so_far:idx_so_far+b]).float().to(self.device)
                cgls_slice = cgls_slice.unsqueeze(1)  # [B, 1, 512, 512]
                temp_y_cgls = cgls_slice * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            else:
                temp_y_cgls = None

            measurement_operator = None
            if hasattr(A_funcs, "prepare_measurement"):
                measurement_operator = A_funcs
            elif hasattr(A_funcs, "base") and hasattr(A_funcs.base, "prepare_measurement"):
                measurement_operator = A_funcs.base

            if measurement_operator is not None:
                y = measurement_operator.prepare_measurement(y)
            else:
                y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size) # [B, 1, 512, 512]
            if deg in ['fourier', 'fourier_deblur_aniso'] and fourier_ref_batch is not None and fourier_intra_batch is not None:
                Apy = A_funcs.compose_guidance(Apy, fourier_ref_batch, fourier_intra_batch)

            apy_imgs = [inverse_data_transform(config, Apy[i]) for i in range(len(Apy))]
            if deg in ["sr_averagepooling", "sr_bicubic"]:
                before_refine_vis = [apy_imgs[i].detach().cpu() for i in range(len(apy_imgs))]
            elif deg in ['fourier', 'fourier_deblur_aniso']:
                if fourier_ref_batch is not None:
                    fourier_ref_vis = inverse_data_transform(config, fourier_ref_batch)
                    before_refine_vis = [fourier_ref_vis[i].detach().cpu() for i in range(len(fourier_ref_vis))]
                else:
                    before_refine_vis = [apy_imgs[i].detach().cpu() for i in range(len(apy_imgs))]
            else:
                before_refine_vis = [before_refine_imgs[i].detach().cpu() for i in range(len(before_refine_imgs))]

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "Orig"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "before_refine_png"), exist_ok=True)
            if getattr(self.args, "ddnm_first_output_path", ""):
                os.makedirs(os.path.join(self.args.image_folder, "first_ddnm_png"), exist_ok=True)
            for i in range(len(Apy)): ## what is Apy and orig?
                apy_i = apy_imgs[i]
                before_refine_i = before_refine_vis[i]
                if "UHRCT" in args.config: # resize from [3, 512, 512] to [256, 256]
                    apy_i = F.interpolate(apy_i.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                    before_refine_i = F.interpolate(before_refine_i.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                x_orig_i = inverse_data_transform(config, x_orig[i])
                apy_i_1ch = apy_i[0]
                before_refine_i_1ch = before_refine_i[0]
                x_orig_i_1ch = x_orig_i[0]
                tvu.save_image(
                    apy_i_1ch,
                    os.path.join(self.args.image_folder, f"Apy/Apy_{(idx_so_far + i):02d}.png")
                )
                tvu.save_image(
                    before_refine_i_1ch,
                    os.path.join(self.args.image_folder, f"before_refine_png/before_refine_{(idx_so_far + i):02d}.png")
                )
                first_output_arr = load_first_output_projection(getattr(self.args, "ddnm_first_output_path", ""), idx_so_far + i)
                if first_output_arr is not None:
                    first_output_tensor = torch.from_numpy(first_output_arr).float()
                    tvu.save_image(
                        first_output_tensor,
                        os.path.join(self.args.image_folder, f"first_ddnm_png/first_ddnm_{(idx_so_far + i):02d}.png")
                    )
                if save_formula and i == 0:
                    tvu.save_image(
                        apy_i_1ch,
                        os.path.join(formula_dir, "A_dagger_y.png")
                    )
                    np.save(os.path.join(formula_dir, "A_dagger_y.npy"), apy_i_1ch.detach().cpu().numpy())
                tvu.save_image(
                    x_orig_i_1ch,
                    os.path.join(self.args.image_folder, f"Orig/orig_{(idx_so_far + i):02d}.png")
                )

            #Start DDIM
            # Generate or reuse shared noise with optional warping
            if self.use_shared_noise:
                if self.shared_noise is None:
                    # Initialize shared noise on first iteration
                    self.shared_noise = torch.randn(
                        1,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                    )
                    x_noise = self.shared_noise.expand(y.shape[0], -1, -1, -1).clone()
                    
                    # Prepare for warped noise: get upsampled LR for next iteration
                    if use_warped_noise:
                        # Inverse transform to get [0,1] range upsampled LR (already upsampled to 512)
                        curr_lr_img = inverse_data_transform(config, lr_for_warp)  # [B, C, 512, 512]
                        prev_upsampled_lr = curr_lr_img[0, 0].cpu().numpy()  # [H, W]
                        prev_noise = self.shared_noise.clone()
                        
                        # Create warped_noise directory and save initial noise
                        os.makedirs(os.path.join(self.args.image_folder, "warped_noise"), exist_ok=True)
                        self._save_noise_visualization(x_noise, idx_so_far, config)
                else:
                    # Use shared noise for subsequent iterations
                    if use_warped_noise and prev_noise is not None:
                        # Warp previous noise to current frame using optical flow
                        curr_lr_img = inverse_data_transform(config, lr_for_warp)  # [B, C, 512, 512]
                        curr_upsampled_lr = curr_lr_img[0, 0].cpu().numpy()  # [H, W]
                        
                        x_noise = warp_noise_with_flow(
                            prev_upsampled_lr, curr_upsampled_lr, prev_noise, alpha=0.1
                        )
                        x_noise = x_noise.expand(y.shape[0], -1, -1, -1).clone()
                        
                        # Update for next iteration
                        prev_upsampled_lr = curr_upsampled_lr
                        prev_noise = x_noise.clone()
                        
                        # Save warped noise visualization
                        self._save_noise_visualization(x_noise, idx_so_far, config)
                    else:
                        # Without warping, just reuse shared noise
                        x_noise = self.shared_noise.expand(y.shape[0], -1, -1, -1).clone()
            else:
                x_noise = torch.randn(
                    y.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                )
            x = torch.clone(x_noise).to(self.device)
            temp_y_original = F.interpolate(temp_y_original, size=(512, 512), mode='bilinear', align_corners=False)
            if temp_y_cgls is not None:
                temp_y_cgls = F.interpolate(temp_y_cgls, size=(512, 512), mode='bilinear', align_corners=False)
            if deg in ['fourier', 'fourier_deblur_aniso'] and fourier_ref_batch is not None and fourier_intra_batch is not None:
                temp_y_original = A_funcs.compose_guidance(temp_y_original, fourier_ref_batch, fourier_intra_batch)
            
            if args.setup == "ddnm_orig" and not getattr(self.args, "always_noised_ddnm_first", False):
                if deg not in ['fourier', 'fourier_deblur_aniso']:
                    temp_y_original = None
                    temp_y_cgls = None

            use_ddnm_plus = (sigma_y != 0.)
            if deg == 'fourier_deblur_aniso':
                use_ddnm_plus = False

            with torch.no_grad():
                if not use_ddnm_plus: # noise-free case, or operators without DDNM+ Lambda support
                    x, x0_preds, x0t_preds, start_step = self.ddnm_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, temp_y_original=temp_y_original, temp_y_cgls=temp_y_cgls, cls_fn=cls_fn, classes=None, config=config, args=self.args, idx_so_far=idx_so_far, prev_projection_output=prev_projection_output)
                else: # noisy case, turn to ddnm+
                    x, x0_preds, x0t_preds, start_step = self.ddnm_plus_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, temp_y_original=temp_y_original, temp_y_cgls=temp_y_cgls, cls_fn=cls_fn, classes=None, config=config, args=self.args, idx_so_far=idx_so_far, prev_projection_output=prev_projection_output)

            start_steps.append(start_step)
            prev_projection_output = x[0].clone()
            x = [inverse_data_transform(config, xi, clip_max=args.clip_max) for xi in x]

            if deg == 'detail_enhance_lowref':
                before_refine_batch = torch.stack([img.to(self.device) for img in before_refine_vis], dim=0)
                refined_batch = x[0].to(self.device)
                x[0] = A_funcs.compose_final(refined_batch, before_refine_batch).detach().cpu()
            elif deg in ['fourier', 'fourier_deblur_aniso']:
                if fourier_ref_batch is not None and fourier_intra_batch is not None:
                    inter_batch = inverse_data_transform(config, fourier_ref_batch).to(self.device)
                    intra_batch = inverse_data_transform(config, fourier_intra_batch).to(self.device)
                else:
                    inter_batch = torch.stack([img.to(self.device) for img in before_refine_vis], dim=0)
                    intra_batch = inter_batch
                refined_batch = x[0].to(self.device)
                x[0] = A_funcs.compose_final(refined_batch, inter_batch, intra_batch).detach().cpu()

            # print(x[0].min(), x[0].max())
            # exit()

            os.makedirs(os.path.join(self.args.image_folder, "pred_png"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "pred_npy"), exist_ok=True)
            for j in range(x[0].size(0)):
                # x[0].shape: [3, 512, 512]
                # x[0] to [512, 512]
                x_1ch = x[0][j][0]
                tvu.save_image(
                    x_1ch, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far + j):02d}.png")
                )
                if save_formula and j == 0:
                    tvu.save_image(
                        x_1ch, os.path.join(formula_dir, "x0_hat_final.png")
                    )
                    np.save(os.path.join(formula_dir, "x0_hat_final.npy"), x_1ch.detach().cpu().numpy())
                whole_result.append((x[0][j]).detach().cpu().numpy())
                np.save(os.path.join(self.args.image_folder, f"pred_npy/pred_{(idx_so_far + j):02d}.npy"), x_1ch.detach().cpu().numpy())
                orig = inverse_data_transform(config, x_orig[j])
                orig_1ch = orig[0]
                before_refine_1ch = before_refine_vis[j][0].to(orig_1ch.device)
                first_output_arr = load_first_output_projection(getattr(self.args, "ddnm_first_output_path", ""), idx_so_far + j)
                first_output_1ch = None
                if first_output_arr is not None:
                    first_output_1ch = torch.from_numpy(first_output_arr).float().to(orig_1ch.device)

                mse = torch.mean((x_1ch.to(self.device) - orig_1ch) ** 2)
                before_mse = torch.mean((before_refine_1ch - orig_1ch) ** 2)
                psnr = 10 * torch.log10(1 / torch.clamp(mse, min=1e-12))
                before_psnr = 10 * torch.log10(1 / torch.clamp(before_mse, min=1e-12))
                if first_output_1ch is not None:
                    first_mse = torch.mean((first_output_1ch - orig_1ch) ** 2)
                    first_psnr = 10 * torch.log10(1 / torch.clamp(first_mse, min=1e-12))
                else:
                    first_mse = None
                    first_psnr = None

                avg_psnr += float(psnr.detach().cpu())
                avg_before_psnr += float(before_psnr.detach().cpu())
                avg_mse += float(mse.detach().cpu())
                avg_before_mse += float(before_mse.detach().cpu())
                if first_psnr is not None:
                    avg_first_psnr += float(first_psnr.detach().cpu())
                    avg_first_mse += float(first_mse.detach().cpu())
                    first_metric_count += 1

                row = {
                    "proj_idx": idx_so_far + j,
                    "before_mse": float(before_mse.detach().cpu()),
                    "pred_mse": float(mse.detach().cpu()),
                    "mse_improvement": float((before_mse - mse).detach().cpu()),
                    "before_psnr": float(before_psnr.detach().cpu()),
                    "pred_psnr": float(psnr.detach().cpu()),
                    "psnr_improvement": float((psnr - before_psnr).detach().cpu()),
                }
                if first_psnr is not None:
                    row.update({
                        "first_mse": float(first_mse.detach().cpu()),
                        "pred_minus_first_mse": float((first_mse - mse).detach().cpu()),
                        "first_psnr": float(first_psnr.detach().cpu()),
                        "pred_minus_first_psnr": float((psnr - first_psnr).detach().cpu()),
                    })
                metric_rows.append(row)

            idx_so_far += y.shape[0]

            n_seen = idx_so_far - idx_init
            pbar.set_description(
                "PSNR pred/before%s: %.2f / %.2f%s"
                % (
                    "/first" if getattr(self.args, "ddnm_first_output_path", "") else "",
                    avg_psnr / n_seen,
                    avg_before_psnr / n_seen,
                    (" / %.2f" % (avg_first_psnr / n_seen)) if getattr(self.args, "ddnm_first_output_path", "") else "",
                )
            )

        num_samples = idx_so_far - idx_init
        avg_psnr = avg_psnr / num_samples
        avg_before_psnr = avg_before_psnr / num_samples
        avg_mse = avg_mse / num_samples
        avg_before_mse = avg_before_mse / num_samples
        if first_metric_count > 0:
            avg_first_psnr = avg_first_psnr / first_metric_count
            avg_first_mse = avg_first_mse / first_metric_count

        print("Total Average pred PSNR: %.2f" % avg_psnr)
        print("Total Average before PSNR: %.2f" % avg_before_psnr)
        print("Average PSNR improvement: %.4f" % (avg_psnr - avg_before_psnr))
        print("Total Average pred MSE: %.6f" % avg_mse)
        print("Total Average before MSE: %.6f" % avg_before_mse)
        print("Average MSE improvement: %.6f" % (avg_before_mse - avg_mse))
        if first_metric_count > 0:
            print("Total Average first DDNM PSNR: %.2f" % avg_first_psnr)
            print("Pred minus first PSNR: %.4f" % (avg_psnr - avg_first_psnr))
            print("Total Average first DDNM MSE: %.6f" % avg_first_mse)
            print("Pred minus first MSE: %.6f" % (avg_first_mse - avg_mse))
        print("Number of samples: %d" % num_samples)

        metrics_csv = os.path.join(self.args.image_folder, "per_projection_metrics.csv")
        with open(metrics_csv, "w", newline="") as f:
            fieldnames = [
                "proj_idx",
                "before_mse",
                "pred_mse",
                "mse_improvement",
                "before_psnr",
                "pred_psnr",
                "psnr_improvement",
            ]
            if first_metric_count > 0:
                fieldnames.extend([
                    "first_mse",
                    "pred_minus_first_mse",
                    "first_psnr",
                    "pred_minus_first_psnr",
                ])
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(metric_rows)

        with open(os.path.join(self.args.image_folder, "metric_summary.txt"), "w") as f:
            f.write(f"num_samples: {num_samples}\n")
            f.write(f"avg_before_psnr: {float(avg_before_psnr):.6f}\n")
            f.write(f"avg_pred_psnr: {float(avg_psnr):.6f}\n")
            f.write(f"avg_psnr_improvement: {float(avg_psnr - avg_before_psnr):.6f}\n")
            f.write(f"avg_before_mse: {float(avg_before_mse):.8f}\n")
            f.write(f"avg_pred_mse: {float(avg_mse):.8f}\n")
            f.write(f"avg_mse_improvement: {float(avg_before_mse - avg_mse):.8f}\n")
            if first_metric_count > 0:
                f.write(f"avg_first_psnr: {float(avg_first_psnr):.6f}\n")
                f.write(f"avg_pred_minus_first_psnr: {float(avg_psnr - avg_first_psnr):.6f}\n")
                f.write(f"avg_first_mse: {float(avg_first_mse):.8f}\n")
                f.write(f"avg_pred_minus_first_mse: {float(avg_first_mse - avg_mse):.8f}\n")

        # save start_steps as a txt and plot
        # make dir_start_steps
        if self.args.setup == "ddnm_pas":
            dir_start_steps = os.path.join(self.args.image_folder, "start_steps")
            os.makedirs(os.path.join(dir_start_steps), exist_ok=True)

            with open(os.path.join(dir_start_steps, "start_steps.txt"), "w") as f:
                for step in start_steps:
                    f.write(f"{step}\n")
            plt.figure(figsize=(10, 5))
            plt.plot(start_steps)
            plt.title("Start Steps")
            plt.xlabel("Projection Index")
            plt.ylabel("Start Step")
            plt.savefig(os.path.join(dir_start_steps, f"start_steps_plot.png"))
            print("Avg start step:", sum(start_steps) / len(start_steps))
            print("Std start step:", np.std(np.array(start_steps)))

        return whole_result

# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
