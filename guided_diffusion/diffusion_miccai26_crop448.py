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


def crop_lr_measurement(y, deg_scale):
    """
    Crop LR measurement based on degradation scale to handle edge artifacts.
    Args:
        y: [B, C, H, W] LR measurement tensor
        deg_scale: degradation scale (4.0 or 8.0)
    Returns:
        cropped_y: cropped LR measurement
        crop_pixels: number of pixels cropped from each side
    """
    if deg_scale == 4.0:
        crop_pixels = 8
    elif deg_scale == 8.0:
        crop_pixels = 4
    else:
        return y, 0
    
    if crop_pixels > 0 and y.shape[2] > 2 * crop_pixels:
        cropped_y = y[:, :, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
        return cropped_y, crop_pixels
    return y, 0

def pad_apy_to_target(apy, crop_pixels, deg_scale, target_size=512):
    """
    Pad upsampled LR (Apy) to target size by adding zeros on the sides that were cropped.
    Args:
        apy: [B, C, H, W] upsampled LR tensor
        crop_pixels: number of pixels that were cropped from LR
        deg_scale: degradation scale (determines upsampling factor)
        target_size: target HR size (default 448)
    Returns:
        padded_apy: [B, C, target_size, target_size] tensor
    """
    if crop_pixels == 0:
        return apy
    
    # Calculate padding needed (crop_pixels were removed at LR level, so at HR level it's crop_pixels * deg_scale)
    scale = int(deg_scale)
    hr_crop_pixels = crop_pixels * scale

    # Pad with min apy
    padded_apy = F.pad(apy, (hr_crop_pixels, hr_crop_pixels, hr_crop_pixels, hr_crop_pixels), 
                       mode='constant', value=apy.min())
    
    # Ensure it's exactly target_size (handle any rounding issues)
    if padded_apy.shape[2] != target_size or padded_apy.shape[3] != target_size:
        padded_apy = F.interpolate(padded_apy, size=(target_size, target_size), mode='bilinear', align_corners=False)

    return padded_apy

def pad_to_512(img):
    """
    Pad 448x448 image to 512x512 with 32 pixels of zero padding on all sides.
    Args:
        img: [C, 448, 448] or [B, C, 448, 448] tensor
    Returns:
        padded_img: [C, 512, 512] or [B, C, 512, 512] tensor
    """
    return F.pad(img, (32, 32, 32, 32), mode='constant', value=0)
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
    """
    prev_gray = (prev_img * 255).astype(np.uint8)
    curr_gray = (curr_img * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    threshold = np.percentile(flow_magnitude, (1.0 - alpha) * 100)
    motion_mask = flow_magnitude > threshold
    noise_np = prev_noise.squeeze().cpu().numpy()
    fresh_noise = np.random.randn(*noise_np.shape)
    if len(noise_np.shape) == 2:
        fresh_noise = (fresh_noise - fresh_noise.mean()) / (fresh_noise.std() + 1e-8)
        combined_noise = noise_np.copy()
        combined_noise[motion_mask] = fresh_noise[motion_mask]
        combined_noise = combined_noise[np.newaxis, ...]
    else:
        combined_noise = []
        for c_idx in range(noise_np.shape[0]):
            fresh_c = fresh_noise[c_idx]
            fresh_c = (fresh_c - fresh_c.mean()) / (fresh_c.std() + 1e-8)
            combined_c = noise_np[c_idx].copy()
            combined_c[motion_mask] = fresh_c[motion_mask]
            combined_noise.append(combined_c)
        combined_noise = np.stack(combined_noise)
    combined_noise = torch.from_numpy(combined_noise).to(prev_noise.device)
    combined_noise = combined_noise.view_as(prev_noise)
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
        self.config.data.image_size = 448 # crop448
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
        from functions.svd_ddnm_miccai26_crop448 import ddnm_diffusion, ddnm_plus_diffusion
        self.ddnm_diffusion = ddnm_diffusion
        self.ddnm_plus_diffusion = ddnm_plus_diffusion

        if self.args.cgls_path is not None:
            self.cgls_data = np.load(self.args.cgls_path)  # [N, 448, 448], range [0, 1]
        # Shared noise support (mirrors diffusion_miccai26.py behavior)
        self.shared_noise = None
        if getattr(self.args, 'shared_noise', False):
            # noise will be initialized when shape is known in sampling loop
            self.use_shared_noise = True
        else:
            self.use_shared_noise = False


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
            # Adjust model config for rsz448 (1-channel model)
            if "rsz448" in self.args.ckpt:
                self.config.model.in_channels = 1
                self.config.model.out_channels = 1
                self.config.data.channels = 1
            
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
                ckpt = "/tmp/openai-2026-01-06-10-36-16-194901/ema_0.9999_750000.pt"
            elif self.args.ckpt == "rsz448":
                ckpt = "/workspace/improved-diffusion/checkpoint/gpu4_2gb_model_resize448x448/ema_0.9999_060000.pt"
            elif self.args.ckpt == "rsz448_100k":
                ckpt = "/workspace/improved-diffusion/checkpoint/gpu4_2gb_model_resize448x448/ema_0.9999_100000.pt"
                # batch size: 9, GPU: 1 (96GB)
            else:
                ckpt = self.args.ckpt

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

    def _save_noise_visualization(self, x_noise, proj_idx, config):
        try:
            noise = x_noise[0]
            if noise.shape[0] > 1:
                noise_viz = noise.mean(dim=0)
            else:
                noise_viz = noise[0]
            noise_np = noise_viz.detach().cpu().numpy()
            noise_min = noise_np.min()
            noise_max = noise_np.max()
            if noise_max > noise_min:
                noise_norm = (noise_np - noise_min) / (noise_max - noise_min)
            else:
                noise_norm = np.zeros_like(noise_np)
            noise_uint8 = (noise_norm * 255).astype(np.uint8)
            output_path = os.path.join(self.args.image_folder, f"warped_noise/noise_proj_{proj_idx:02d}.png")
            cv2.imwrite(output_path, noise_uint8)
        except Exception as e:
            print(f"Warning: Failed to save noise visualization for proj_idx {proj_idx}: {e}")

            
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
            A = torch.nn.AdaptiveAvgPool2d((448//scale,448//scale))
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
            
            x = torch.randn(
                x_orig.shape[0],
                config.data.channels,
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
            # Adjust channels for rsz448 (1-channel model)
            num_channels = 1 if self.args.ckpt == "rsz448" else config.data.channels
            A_funcs = SuperResolution(num_channels, config.data.image_size, blur_by, self.device)
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

        # For warped/shared noise handling
        prev_noise = None
        prev_upsampled_lr = None
        use_warped_noise = self.use_shared_noise and getattr(self.args, 'warped_noise', False)

        for idx, ITER_DATA in enumerate(pbar):
            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 448, 448]

            x_deg = x_deg.to(self.device)
            x_orig = x_orig.to(self.device)

            # For rsz448 model (1-channel), extract channel 0 only
            if "rsz448" in self.args.ckpt:
                x_deg = x_deg[:, 0:1, :, :]  # [B, 1, H, W]
                x_orig = x_orig[:, 0:1, :, :]  # [B, 1, H, W]

            y = data_transform(self.config, x_deg)
            
            # Crop LR measurement for crop448 model
            y, crop_pixels = crop_lr_measurement(y, args.deg_scale)
            x_orig = data_transform(self.config, x_orig)

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
                cgls_slice = cgls_slice.unsqueeze(1)  # [B, 1, 448, 448]
                temp_y_cgls = cgls_slice * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            else:
                temp_y_cgls = None
            
            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size) # [B, 1, 448, 448]
            
            # Pad Apy to target size
            Apy = pad_apy_to_target(Apy, crop_pixels, args.deg_scale, target_size=512)

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "Orig"), exist_ok=True)
            for i in range(len(Apy)): ## what is Apy and orig?
                apy_i = inverse_data_transform(config, Apy[i])
                if "UHRCT" in args.config: # resize from [3, 448, 448] to [256, 256]
                    apy_i = F.interpolate(apy_i.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                x_orig_i = inverse_data_transform(config, x_orig[i])
                apy_i_1ch = apy_i[0]
                x_orig_i_1ch = x_orig_i[0]
                tvu.save_image(
                    apy_i_1ch,
                    os.path.join(self.args.image_folder, f"Apy/Apy_{(idx_so_far + i):02d}.png")
                )
                # Pad orig to 512x512 before saving
                # x_orig_i_1ch_padded = pad_to_512(x_orig_i_1ch)
                tvu.save_image(
                    x_orig_i_1ch,
                    os.path.join(self.args.image_folder, f"Orig/orig_{(idx_so_far + i):02d}.png")
                )

            #Start DDIM: generate or reuse shared noise with optional warping
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
                    if use_warped_noise:
                        curr_lr_img = inverse_data_transform(config, lr_for_warp)
                        prev_upsampled_lr = curr_lr_img[0, 0].cpu().numpy()
                        prev_noise = self.shared_noise.clone()
                        os.makedirs(os.path.join(self.args.image_folder, "warped_noise"), exist_ok=True)
                        self._save_noise_visualization(x_noise, idx_so_far, config)
                else:
                    if use_warped_noise and prev_noise is not None:
                        curr_lr_img = inverse_data_transform(config, lr_for_warp)
                        curr_upsampled_lr = curr_lr_img[0, 0].cpu().numpy()

                        x_noise = warp_noise_with_flow(
                            prev_upsampled_lr, curr_upsampled_lr, prev_noise, alpha=0.1
                        )
                        x_noise = x_noise.expand(y.shape[0], -1, -1, -1).clone()

                        prev_upsampled_lr = curr_upsampled_lr
                        prev_noise = x_noise.clone()

                        self._save_noise_visualization(x_noise, idx_so_far, config)
                    else:
                        x_noise = self.shared_noise.expand(y.shape[0], -1, -1, -1).clone()
            else:
                x_noise = torch.randn(
                    y.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                )
            x = torch.clone(x_noise).to(self.device)
            temp_y_original = F.interpolate(temp_y_original, size=(448, 448), mode='bilinear', align_corners=False)
            if temp_y_cgls is not None:
                temp_y_cgls = F.interpolate(temp_y_cgls, size=(448, 448), mode='bilinear', align_corners=False)
            
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
            x = [inverse_data_transform(config, xi) for xi in x]
            
            # For rsz448 model (1-channel output), convert to 3-channel for consistency
            if self.args.ckpt == "rsz448":
                x = [xi.repeat(1, 3, 1, 1) for xi in x]  # [B, 1, H, W] -> [B, 3, H, W]

            os.makedirs(os.path.join(self.args.image_folder, "pred_png"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "pred_npy"), exist_ok=True)
            for j in range(x[0].size(0)):
                # x[0].shape: [3, 448, 448]
                # x[0] to [448, 448]
                x_1ch = x[0][j][0]
                # Pad pred to 512x512 before saving
                x_1ch_padded = pad_to_512(x_1ch)
                tvu.save_image(
                    x_1ch_padded, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far + j):02d}.png")
                )
                # Pad whole to 512x512 before appending
                whole_padded = pad_to_512(x[0][j])
                whole_result.append(whole_padded.detach().cpu().numpy())
                np.save(os.path.join(self.args.image_folder, f"pred_npy/pred_{(idx_so_far + j):02d}.npy"), x_1ch.detach().cpu().numpy())
                orig = inverse_data_transform(config, x_orig[j])
                orig_1ch = orig[0]
                mse = torch.mean((x_1ch_padded.to(self.device) - orig_1ch) ** 2)
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
