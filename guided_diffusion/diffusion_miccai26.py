import os
import math
import torch
import torch.utils.data as data
import numpy as np
import random
from scipy.linalg import orth
import tqdm
from PIL import Image
import cv2

from datasets import get_dataset_miccai26, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
import matplotlib.pyplot as plt

import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

try:
    import sys
    if "/workspace/ISCS" not in sys.path:
        sys.path.append("/workspace/ISCS")
    from algorithms.utils import slerp_path as iscs_slerp_path
except Exception:
    iscs_slerp_path = None

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
        self.slerp_noise_path = None
        self.slerp_noise_proj_start = 0
        noise_control = str(getattr(self.args, "noise_control", "none")).lower()
        if getattr(self.args, "shared_noise", False) and noise_control == "none":
            noise_control = "shared"
        if getattr(self.args, "warped_noise", False) and noise_control in ["none", "shared"]:
            noise_control = "warped"
        self.noise_control = noise_control
        self.use_shared_noise = self.noise_control in ["shared", "warped", "slerp"]

    def _randn_with_seed(self, shape, seed):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        return torch.randn(*shape, generator=gen, device=self.device)

    def _build_slerp_noise_path(self, num_projs, channels, image_size):
        if num_projs <= 0:
            return None
        if iscs_slerp_path is None:
            raise ImportError("ISCS slerp_path import failed; cannot use noise_control=slerp")
        z0 = self._randn_with_seed((1, channels, image_size, image_size), getattr(self.args, "slerp_endpoint_seed_0", 1234))
        z1 = self._randn_with_seed((1, channels, image_size, image_size), getattr(self.args, "slerp_endpoint_seed_1", 5678))
        if num_projs == 1:
            return z0
        path = iscs_slerp_path(z0, z1, n_mid=max(num_projs - 2, 0), include_endpoints=True)
        if path.dim() == 3:
            path = path.unsqueeze(0)
        return path.to(self.device)

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
            
        if getattr(self.args, "save_whole", True):
            np.save(os.path.join(self.args.image_folder, "whole.npy"), np.array(result))
        return result

            
    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset_miccai26(args, config)

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
        pbar = tqdm.tqdm(val_loader)
        whole_result = []
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

        dataset, test_dataset = get_dataset_miccai26(args, config)

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
            from functions.svd_operators import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        elif deg == 'highfreq':
            from functions.highFrequency_deg import HighFreqDegradation
            # 원하는 cutoff_radius(고주파 반경) 값 지정
            cutoff_radius = 32  # 예시값, 이미지 크기에 맞게 조정
            A_funcs = HighFreqDegradation(
                config.data.channels,
                config.data.image_size,
                cutoff_radius,
                self.device
            )
        else:
            raise ValueError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        whole_result = []

        start_steps = []
        
        # For warped noise: track previous noise and upsampled LR
        prev_noise = None
        prev_upsampled_lr = None
        use_warped_noise = self.noise_control == "warped"
        use_slerp_noise = self.noise_control == "slerp"
        if use_slerp_noise:
            self.slerp_noise_proj_start = args.subset_start
            self.slerp_noise_path = self._build_slerp_noise_path(
                len(test_dataset),
                config.data.channels,
                config.data.image_size,
            )

        for idx, ITER_DATA in enumerate(pbar):
            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 512, 512]

            x_deg = x_deg.to(self.device)
            x_orig = x_orig.to(self.device)

            y = data_transform(self.config, x_deg)
            x_orig = data_transform(self.config, x_orig)

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
            
            # Always create temp_y_original from upsampled LR for select_start_step
            temp_y_original = torch.clone(y).to(self.device)
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
            
            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size) # [B, 1, 512, 512]

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "Orig"), exist_ok=True)
            for i in range(len(Apy)): ## what is Apy and orig?
                apy_i = inverse_data_transform(config, Apy[i])
                if "UHRCT" in args.config: # resize from [3, 512, 512] to [256, 256]
                    apy_i = F.interpolate(apy_i.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                x_orig_i = inverse_data_transform(config, x_orig[i])
                apy_i_1ch = apy_i[0]
                x_orig_i_1ch = x_orig_i[0]
                tvu.save_image(
                    apy_i_1ch,
                    os.path.join(self.args.image_folder, f"Apy/Apy_{(idx_so_far + i):02d}.png")
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
            if use_slerp_noise:
                local_idx = idx_so_far - self.slerp_noise_proj_start
                base_noise = self.slerp_noise_path[local_idx:local_idx + 1]
                x_noise = base_noise.expand(y.shape[0], -1, -1, -1).clone()
            elif self.use_shared_noise:
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
            
            if args.setup == "ddnm_orig":
                temp_y_original = None
                temp_y_cgls = None

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm -> HERE ##
                    x, x0_preds, x0t_preds, start_step = self.ddnm_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, temp_y_original=temp_y_original, temp_y_cgls=temp_y_cgls, cls_fn=cls_fn, classes=None, config=config, args=self.args, idx_so_far=idx_so_far)
                else: # noisy case, turn to ddnm+
                    x, x0_preds, x0t_preds, start_step = self.ddnm_plus_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, temp_y_original=temp_y_original, temp_y_cgls=temp_y_cgls, cls_fn=cls_fn, classes=None, config=config, args=self.args, idx_so_far=idx_so_far)

            start_steps.append(start_step)
            x = [inverse_data_transform(config, xi, clip_max=args.clip_max) for xi in x]

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
                mse = torch.mean((x_1ch.to(self.device) - orig_1ch) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

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
