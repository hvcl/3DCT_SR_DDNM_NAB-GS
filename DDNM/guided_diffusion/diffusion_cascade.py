import os
import logging
import time
import glob
import math

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import tifffile

from datasets import get_dataset_cascade, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
# from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
import matplotlib.pyplot as plt

import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
import csv, json
from datasets import get_dataset_cascade  # already imported earlier but ensure available here
from guided_diffusion.utils.metrics import compute_and_save_metrics_cascade
import os
import json
import csv
import glob
import pickle

def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

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

        from functions.svd_ddnm_cascade import ddnm_diffusion, ddnm_plus_diffusion

        self.ddnm_diffusion = ddnm_diffusion
        self.ddnm_plus_diffusion = ddnm_plus_diffusion

    def sample(self, simplified):
        # simplified: True if it is in the argument
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)
            ckpt = "/workspace/DDNM/exp/logs/totensor/ckpt.pth"
            print("Loading checkpoint {}".format(ckpt))
            
            if 'totensor' in ckpt or 'CheXpert_8' in ckpt:
                model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(ckpt, map_location=self.device)[0])
            
        elif self.config.model.type == 'openai': # here 

            config_dict = vars(self.config.model)
            model = create_model(**config_dict)

            if self.config.model.use_fp16:
                model.convert_to_fp16()
            
            # Ver: MICCAI25
            if self.args.ckpt == "MICCAI25":
                ckpt = "/workspace/DDNM/exp/logs/chestx14/ema_0.9999_300000.pt" # epoch > 50

            # Ver: 500MB, ddpm baseline
            # ckpt = "/workspace/DDNM/exp/logs/CheX-ray14_CheXpert_512x512/ema_0.9999_900000.pt"

            # Ver: same model architecture as used in MICCAI25 submission
            # ckpt = "/workspace/DDNM/exp/logs/CheX-ray14_CheXpert_512x512-2gb/ema_0.9999_300000.pt" # epoch: 17.8

            elif self.args.ckpt == "SIDE":
                ckpt = "/workspace/DDNM/exp/logs/CheX-ray14_CheXpert_512x512-2gb/ema_0.9999_620000.pt"

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

                # import torch.nn.functional as F
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
            # --- Stage2-only handling: load stage1 temporal output and run ddnm with deg=denoising ---
            if getattr(self.args, "stage2_only", False) and "GS0asTempy" not in self.args.image_folder:
                # stage1_dir = getattr(self.args, "stage1_dir", "")
                # if not stage1_dir or not os.path.exists(stage1_dir):
                #     raise FileNotFoundError(f"stage1_dir not found: {stage1_dir}")

                # Prefer temporally-smoothed file
                # sigma_t = getattr(self.args, "temporal_sigma_t", None)
                # candidate = None
                # if sigma_t is not None:
                #     fname = f"whole_gaussian_filter1d_sigma_t{sigma_t}.npy"
                #     p = os.path.join(stage1_dir, fname)
                #     if os.path.exists(p):
                #         candidate = p

                # if candidate is None:
                #     for fn in ["whole_gaussian_filter1d_sigma_t1.0.npy", "whole.npy", "original_stack.npy"]:
                #         p = os.path.join(stage1_dir, fn)
                #         if os.path.exists(p):
                #             candidate = p
                #             break

                # # load stack (N,H,W) normalized [0,1]
                # if candidate is not None:
                #     stack = np.load(candidate)
                #     if stack.ndim == 4:
                #         stack = stack[:, 0, :, :]
                # else:
                #     # fallback to pred_png folder
                #     png_dir = os.path.join(stage1_dir, "pred_png")
                #     if os.path.isdir(png_dir):
                #         files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
                #         arrs = []
                #         for fpath in files:
                #             im = plt.imread(os.path.join(png_dir, fpath))
                #             if im.ndim == 3:
                #                 im = im[..., 0]
                #             arrs.append(im.astype(np.float32))
                #         stack = np.stack(arrs, axis=0)
                #     else:
                #         raise FileNotFoundError("No stage1 output (npy/png) found in stage1_dir")

                if "cycle" in self.args.image_folder:
                    # candidate = "/Alexandrite/jhnoh/r2_gaussian/NAB_GS_mela0050.pickle"
                    candidate = "/Alexandrite/jhnoh/r2_gaussian/cycle_1_NAB_GS_mela0050.pickle"
                    # candidate = "/Alexandrite/jhnoh/r2_gaussian/cycle2_t15_NAB_GS_mela0050.pickle"
                    with open(candidate, 'rb') as f:
                        round0_data = pickle.load(f)
                    stack = round0_data["train"]["projections"] # [100, 512, 512]
                    # min-max norm (100장 projections에 대해 per-projection으로 시행)
                    stack = np.array(stack)
                    for i in range(stack.shape[0]):
                        p = stack[i]
                        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
                        stack[i] = p
                    # stack = (stack - stack.min()) / (stack.max() - stack.min() + 1e-8)

                stack = stack.astype(np.float32)
                N, H, W = stack.shape
                # print(f"Loaded stage1 stack from {stage1_dir}, shape={stack.shape}")

                # deg_mode = getattr(self.args, "deg", "sr_averagepooling")

                # Stage2 control params (from CLI)
                eta2 = float(getattr(self.args, "eta", 0.9))      # DDIM / DDNM eta (deterministic <-> stochastic)
                sigma_y = float(getattr(self.args, "sigma_y", 0.0))      # measurement noise used by ddnm_plus (if >0)
                # SR multi-stage params
                # sr_gauss_sigma = float(getattr(self.args, "sr_stage_gauss_sigma", 1.0))

                    # exit(
                # --- SR multi-stage processing ---
                # define per-mode stage levels:
                # deg_mode = getattr(self.args, "deg", "sr_averagepooling")
                # if deg_mode == "sr_averagepooling":
                #     # interpret deg_scale: if 4 -> single-level (512->256). if 8 -> two levels [4,2] as requested.
                #     ds = int(round(getattr(self.args, "deg_scale", 4)))
                #     if ds == 8:
                #         levels = [4, 2]
                #     elif ds == 4:
                #         levels = [2]
                #     else:
                #         # fallback: single downsample by ds//? treat ds as overall factor -> produce factor ds/ (ds//2)
                #         levels = [max(2, ds // 2)]
                # else:
                #     levels = []

                # y_source holds current input images (0..1) for the next stage; initialize from stage1 stack
                y_source_np = stack.copy()  # shape (N,H,W) in [0,1]

                # for level_idx, level in enumerate(levels):
                #     print(f"SR stage {level_idx+1}/{len(levels)}: downsample factor {level} -> LR size {512//level}")
                stage_out = []
                #     # configure A_funcs for this level
                from functions.svd_operators import SuperResolution
                A_funcs_lvl = SuperResolution(self.config.data.channels, 
                                              self.config.data.image_size, 
                                              int(self.args.deg_scale),
                                              self.device)

                for i in tqdm.tqdm(range(N), desc=f"DDNM cycle per-frame"):
                    frame = y_source_np[i:i+1]  # (1,H,W) in [0,1]
                    c = self.config.data.channels if hasattr(self.config.data, "channels") else 1
                    x_deg = torch.from_numpy(frame).unsqueeze(1).float().to(self.device) # [1,1,H,W]
                    # if "cycle" in self.args.image_folder:
                    #     # min-max normalize
                    #     x_min = x_deg.min()
                    #     x_max = x_deg.max()
                    #     x_deg = (x_deg - x_min) / (x_max - x_min + 1e-8)

                    if c > 1:
                        x_deg = x_deg.repeat(1, c, 1, 1)
                    # model domain
                    try:
                        y_tensor = data_transform(self.config, x_deg)
                    except Exception:
                        y_tensor = x_deg * 2.0 - 1.0

                    # downsample to LR for measurement
                    # lr_h = max(1, self.config.data.image_size // level)
                    # y_lr = F.interpolate(y_tensor, size=(lr_h, lr_h), mode='bilinear', align_corners=False)
                    # y_lr by average pooling to match A_funcs behavior
                    y_lr = F.avg_pool2d(y_tensor, 
                                        kernel_size=int(self.args.deg_scale),
                                        stride=int(self.args.deg_scale))
                    
                    b, ch, h, w = y_lr.size()
                    y_vec = y_lr.reshape((b, -1))

                    # init noise image for ddnm (full-res)
                    x_noise = torch.randn(
                        1,
                        self.config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                        device=self.device,
                    )
                    # baseline temp_y: use current full-res source (model-domain)
                    temp_y = y_tensor.clone()
                    if "cycle" in self.args.image_folder:
                        temp_y = F.interpolate(y_lr, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)

                    with torch.no_grad():
                        if float(sigma_y) == 0.0:
                            out_list, _, _, _ = self.ddnm_diffusion(
                                x_noise, model, self.betas, eta2, A_funcs_lvl, y_vec, temp_y=temp_y, cls_fn=cls_fn,
                                classes=None, config=self.config, args=self.args
                            )
                        else:
                            out_list, _, _, _ = self.ddnm_plus_diffusion(
                                x_noise, model, self.betas, eta2, A_funcs_lvl, y_vec, float(sigma_y), temp_y=temp_y,
                                cls_fn=cls_fn, classes=None, config=self.config, args=self.args
                            )

                    out_t = out_list[0].to(self.device)
                    try:
                        out_img = inverse_data_transform(self.config, out_t)
                    except Exception:
                        out_img = torch.clamp((out_t + 1.0) / 2.0, 0.0, 1.0)

                    out_gray = out_img[0, 0].detach().cpu().numpy()  # 0..1
                    stage_out.append(out_gray)
                    # makedirs
                    os.makedirs(os.path.join(self.args.image_folder, f"cycle_pngs"), exist_ok=True)
                    # save png
                    tvu.save_image(out_img[0, 0:1], os.path.join(self.args.image_folder, f"cycle_pngs", f"cycle_pred_{i:02d}.png"))

                stage_out_arr = np.stack(stage_out, axis=0).astype(np.float32)  # (N,H,W) in [0,1]
                # save pre-GF result
                pre_name = os.path.join(self.args.image_folder, f"cycle_whole.npy")
                np.save(pre_name, stage_out_arr)
                print(f"Saved cycle -> {pre_name}")

                # apply spatial gaussian filter per-frame (post-processing) and save
                # from scipy.ndimage import gaussian_filter
                # post_arr = np.stack([gaussian_filter(img, sigma=sr_gauss_sigma) for img in stage_out_arr], axis=0)
                # post_arr_not_gau_filtered = np.stack([gaussian_filter(img, sigma=sr_gauss_sigma) for img in stage_out_arr], axis=0)
                # try:
                # except Exception:
                #     post_arr = stage_out_arr.copy()
                # post_name = os.path.join(self.args.image_folder, f"stage2_lvl{level_idx}_postGF.npy")
                # np.save(post_name, post_arr)
                # print(f"Saved stage {level_idx+1} post-GF -> {post_name}")

                # # --- remove this per-stage metric call (delete the block below) ---
                # # compute metrics vs GT for this stage (uses compute_and_save_metrics)
                # try:
                #     metrics = self.compute_and_save_metrics(stage1_dir, post_arr, sigma_t=float(getattr(self.args, "temporal_sigma_t", 1.0)))
                #     # also save stage-specific summary
                #     with open(os.path.join(self.args.image_folder, f"summary_stage{level_idx+1}.json"), "w") as f:
                #         json.dump(metrics, f, indent=2)
                # except Exception as e:
                #     print("compute_and_save_metrics for stage failed:", e)
                # (metrics collection deferred until all SR stages complete)

                # prepare y_source_np for next level: downsample the post-GF (full-res) to next LR size if another stage follows,
                # otherwise keep final SR result as final output
                # if level_idx + 1 < len(levels):
                #     # downsample post_arr (current SR results) to next LR resolution (next_level)
                #     next_level = levels[level_idx + 1]
                #     next_lr_h = max(1, self.config.data.image_size // next_level)
                #     # post_arr currently full-res (512). downsample to next_lr_h -> then upsample to full-res during next stage temp_y creation
                #     post_t = torch.from_numpy(post_arr).unsqueeze(1).to(self.device)  # (N,1,H,W)
                #     post_down = F.interpolate(post_t, size=(next_lr_h, next_lr_h), mode='bilinear', align_corners=False)
                #     # upsample back to full-res for y_source_np content (so next iteration's temp_y uses this baseline)
                #     post_up = F.interpolate(post_down, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)
                #     y_source_np = post_up.squeeze(1).detach().cpu().numpy()
                # else:
                #     # final output (4x: 256->512, 8x: 128->512->256->512)
                #     final_name = os.path.join(self.args.image_folder, f"stage2_final_noGF.npy")
                #     np.save(final_name, post_arr_not_gau_filtered)
                #     print(f"Saved final stage output -> {final_name}")
                #     # also make stage2_out equal to final for backward compatibility
                #     # stage2_out = post_arr
                # # end multi-stage
    
                # after all stages finished, run single unified metrics calculation:
                try:
                    # sigma_t_val = float(getattr(self.args, "temporal_sigma_t", 1.0))
                    # self.compute_and_save_metrics(stage1_dir, self.args.image_folder, sigma_t=sigma_t_val)
                    compute_and_save_metrics_cascade(self.args, self.config, self.args.image_folder)
                except Exception as e:
                    print("compute_and_save_metrics failed:", e)                
                return # stage2_arr
            
            else: # stage 1
                print('Run SVD-based DDNM.',
                    f'{self.config.time_travel.T_sampling} sampling steps.',
                    f'travel_length = {self.config.time_travel.travel_length},',
                    f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                    f'Task: {self.args.deg}.'
                    )
                result_denorm, result_denorm_noAddMin, result_norm, result = self.svd_based_ddnm_plus(model, cls_fn)
            # np.save(os.path.join(self.args.image_folder, "stage1_denorm.npy"), result_denorm)
            # np.save(os.path.join(self.args.image_folder, "stage1_denorm_noAddMin.npy"), result_denorm_noAddMin)
            # np.save(os.path.join(self.args.image_folder, "stage1_norm.npy"), result_norm)
            np.save(os.path.join(self.args.image_folder, "whole.npy"), result)
            
            # save whole_gf.npy with temporal gaussian filter applied
            # from scipy.ndimage import gaussian_filter1d
            # sigma_t = float(getattr(self.args, "temporal_sigma_t", 1.0))
            # whole_gf = gaussian_filter1d(result, sigma=sigma_t, axis=0)
            # np.save(os.path.join(self.args.image_folder, "whole_gf.npy"), whole_gf)
            # # post_arr = np.stack([gaussian_filter(img, sigma=sr_gauss_sigma) for img in stage_out_arr], axis=0)
            # #
            
    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset_cascade(args, config)

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
        elif args.deg =='sr_averagepooling': # here
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((512//scale,512//scale))
            Ap = lambda z: MeanUpsample(z,scale)
        elif args.deg == 'z_averagepooling':
            def ZMeanUpsample(x, scale):
                n, c, h, w = x.shape
                out = torch.zeros(n, c, h, scale, w).to(x.device) + x.view(n, c, h, 1, w)
                out = out.view(n, c, h*scale, w)
                return out
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((512//scale,512))
            Ap = lambda z: ZMeanUpsample(z,scale)
        elif args.deg =='inpainting':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A = lambda z: z*mask
            Ap = A
        elif args.deg =='mask_color_sr':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=round(args.deg_scale)
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        elif args.deg =='diy':
            # design your own degradation
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=args.deg_scale
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
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
                raise ValueError("please change the config file to set batch size as 1")

            Apy = Ap(y)  # Batch Size

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{(idx_so_far + i):02d}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{(idx_so_far + i):02d}.png")
                )
                
            # init x_T
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next**2).sqrt()
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        # Eq. 19
                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.

                        # Eq. 17
                        x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                        eta = self.args.eta

                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                        # different from the paper, we use DDIM here instead of DDPM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))    
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"pred_{(idx_so_far + j):02d + 1}.png")
            )
            whole_result.append(x[0].detach().cpu().numpy())
            np.save(os.path.join(self.args.image_folder, f"pred_{(idx_so_far + j):02d + 1}.npy"), x[0].detach().cpu().numpy())
            orig = inverse_data_transform(config, x_orig[0])
            mse = torch.mean((x[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        return whole_result    

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset_cascade(args, config)

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
        elif deg == 'sr_averagepooling': ###### here
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
            blur_by_2 = int(args.deg_scale) // 2
            A_funcs_2 = SuperResolution(config.data.channels, config.data.image_size, blur_by_2, self.device)
            blur_by_3 = int(args.deg_scale) // 4
            A_funcs_3 = SuperResolution(config.data.channels, config.data.image_size, blur_by_3, self.device)
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
        # elif deg == 'highfreq':
        #     from functions.highFrequency_deg import HighFreqDegradation
        #     cutoff_radius = 32
        #     A_funcs = HighFreqDegradation(
        #         config.data.channels,
        #         config.data.image_size,
        #         cutoff_radius,
        #         self.device
        #     )
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_psnr_stage2 = 0.0
        avg_psnr_stage3 = 0.0
        count_stage2 = 0
        count_stage3 = 0

        pbar = tqdm.tqdm(val_loader)
        whole_result = []
        whole_result_denorm = []
        whole_result_denorm_noAddMin = []
        whole_result_norm   = []
        whole_result_stage2 = []
        whole_result_stage3 = []

        start_steps = []

        if "GS0asTempy" in self.args.image_folder:
            # load "/Alexandrite/jhnoh/r2_gaussian/NAB_GS_mela0050.pickle"
            # candidate = "/Alexandrite/jhnoh/r2_gaussian/NAB_GS_mela0050.pickle"
            candidate = "/Alexandrite/jhnoh/r2_gaussian/cycle_1_NAB_GS_mela0050.pickle"
            with open(candidate, 'rb') as f:
                round0_data = pickle.load(f)
            deg_round0 = round0_data["train"]["projections"].astype(np.float32) # [100, 512, 512]
            deg_round0_tensor = torch.from_numpy(deg_round0) # [100, 512, 512]

        for idx, ITER_DATA in enumerate(pbar):
            if "GS0asTempy" in self.args.image_folder:
                x_deg_round0 = deg_round0_tensor[idx:idx+1, :, :].to(self.device) # [1, 512, 512]
                # min-max norm
                x_deg_round0 = (x_deg_round0 - x_deg_round0.min()) / (x_deg_round0.max() - x_deg_round0.min() + 1e-8)
                x_deg_round0 = x_deg_round0.unsqueeze(1) # [1, 1, 512, 512]
                # data_transform
                y_round0 = data_transform(self.config, x_deg_round0)
                temp_y_round0 = torch.clone(y_round0).to(self.device)

            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 512, 512]
            if "perVolNorm" in self.args.image_folder:
                x_deg = ITER_DATA["deg_norm_vol"]
                x_orig = ITER_DATA["gt_norm_vol"]

            x_deg_min = ITER_DATA["deg_min_max"][0].numpy() # (np.min(self.degraded_data[idx]), np.max(self.degraded_data[idx]))
            x_deg_max = ITER_DATA["deg_min_max"][1].numpy() # (np.min(self.degraded_data[idx]), np.max(self.degraded_data[idx]))
            # x_gt_min = ITER_DATA["gt_min_max"][0].numpy() # (np.min(self.gt_data[idx]), np.max(self.gt_data[idx]))
            # x_gt_max = ITER_DATA["gt_min_max"][1].numpy() # (np.min(self.gt_data[idx]), np.max(self.gt_data[idx]))

            x_deg = x_deg.to(self.device)
            x_orig = x_orig.to(self.device)

            y = data_transform(self.config, x_deg)
            x_orig = data_transform(self.config, x_orig)

            b, c, h, w = y.size()
            hwc = c * h * w
            
            if self.args.add_noise: # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y)
            
            temp_y = torch.clone(y).to(self.device)
            y = y.reshape((b, hwc))

            # if self.args.deg == "sr_averagepooling":
            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size) # [B, 1, 512, 512]
            # elif self.args.deg == "sr_bicubic":
            #     Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size // self.args.deg_scale,
            #     )
            # exit()

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
                tvu.save_image(
                    x_orig_i_1ch,
                    os.path.join(self.args.image_folder, f"Orig/orig_{(idx_so_far + i):02d}.png")
                )
            
            #Start DDIM
            x_noise = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                # device=self.device,
            )
            x = torch.clone(x_noise).to(self.device)
            temp_y = F.interpolate(temp_y, size=(512, 512), mode='bilinear', align_corners=False)
            if "GS0asTempy" in self.args.image_folder:
                temp_y = temp_y_round0
            elif "prevProj" in self.args.image_folder and idx > 0:
                # use previous prediction as temp_y
                temp_y = x_prev.to(self.device)

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm -> HERE ##
                    x, x0_preds, x0t_preds, start_step = self.ddnm_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)
                else: # noisy case, turn to ddnm+
                    x, x0_preds, x0t_preds, start_step = self.ddnm_plus_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)

            start_steps.append(start_step)
            x = [inverse_data_transform(config, xi) for xi in x]
            # x = [inverse_data_transform_noClamp(config, xi) for xi in x] # expected range: [0.0, 1.0]
            # x_prev = x[-1] # shape: [3, 512, 512]
            # x_clamp = [torch.clamp(xi, 0.0, 1.0) for xi in x]

            os.makedirs(os.path.join(self.args.image_folder, "pred_png"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "pred_npy"), exist_ok=True)

            # x[0].shape: [3, 512, 512]
            # x[0] to [512, 512]
            x_1ch = x[0][0][0]
            tvu.save_image(
                x_1ch, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far):02d}.png")
            )
            exit()
            # 
            xout_temp = x[0][0].detach().cpu().numpy()
            xout_denorm = xout_temp * (x_deg_max - x_deg_min + 1e-8) + x_deg_min
            xout_denorm_noAddMin = xout_temp * (x_deg_max - x_deg_min + 1e-8)
            xout_denorm = np.clip(xout_denorm, 0.0, 1.0)
            xout_norm = np.clip(xout_temp, 0.0, 1.0)
            whole_result.append(xout_norm)
            whole_result_denorm.append(xout_denorm)
            whole_result_denorm_noAddMin.append(xout_denorm_noAddMin)
            whole_result_norm.append(xout_norm)
            #
            np.save(os.path.join(self.args.image_folder, f"pred_npy/pred_{(idx_so_far):02d}.npy"), x_1ch.detach().cpu().numpy())
            orig = inverse_data_transform(config, x_orig[0])
            orig_1ch = orig[0]
            mse = torch.mean((x_1ch.to(self.device) - orig_1ch) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            # --- SECOND ROUND: create aligned 256x256 proxy and run ddnm (256->512) without GT ---
            if "2rounds" in self.args.image_folder:
                # Only run if A_funcs_2 (scale 2) exists (defined earlier for sr_averagepooling)
                pred_full = x_1ch.detach().to(self.device).unsqueeze(0).unsqueeze(0)  # [1,1,512,512]
                _, _, h_round1, w_round1 = pred_full.shape
                h_round1_down2x = h_round1 // int(self.args.deg_scale // 2)
                w_round1_down2x = w_round1 // int(self.args.deg_scale // 2)

                # downsample prediction to 256 (avg pooling)
                # pred_down = F.avg_pool2d(pred_full, kernel_size=2, stride=2)  # [1,1,256,256]
                pred_down = F.interpolate(pred_full, size=(h_round1_down2x, w_round1_down2x), mode='bilinear', align_corners=False)

                # prepare deg_img upsample to 256 (use ITER_DATA['deg_img'], which is LR e.g.128)
                deg_img = ITER_DATA["deg_img"][:,0:1]  # [1,1,128,128]
                # ensure on device and single-channel
                deg_img = deg_img.to(self.device)
                deg_up = F.interpolate(deg_img, size=(h_round1_down2x, w_round1_down2x), mode='bilinear', align_corners=False)  # [1,3,256,256]

                # compute linear align between pred_down and deg_up (solve a,b per-frame)
                p = pred_down.view(-1)
                d = deg_up.view(-1)
                p_mean = torch.mean(p)
                d_mean = torch.mean(d)
                p_var = torch.mean((p - p_mean) ** 2) + 1e-12
                cov = torch.mean((p - p_mean) * (d - d_mean))
                a = (cov / p_var).detach()
                b = (d_mean - a * p_mean).detach()

                # apply affine (a,b) to full-res prediction
                if "noAlign" in self.args.image_folder:
                    pred_down_al = pred_down
                else:
                    pred_down_al = (a * pred_down + b).clamp(0.0, 1.0)  # [1,1,h_round1_down2x,w_round1_down2x]

                # save aligned full-res as intermediate (optional)
                os.makedirs(os.path.join(self.args.image_folder, "stage2_input_aligned"), exist_ok=True)
                tvu.save_image(pred_down_al[0, 0:1], os.path.join(self.args.image_folder, f"stage2_input_aligned/aligned_{(idx_so_far):02d}.png"))
                np.save(os.path.join(self.args.image_folder, f"stage2_input_aligned/aligned_{(idx_so_far):02d}.npy"), pred_down_al[0,0].detach().cpu().numpy())

                # Prepare measurement y for second ddnm: treat aligned full-res as if it were stage1 output,
                # convert to model domain then downsample to LR (256) as measurement
                try:
                    y2_lr = data_transform(self.config, pred_down_al)  # [1,1,256,256] in model domain
                except Exception:
                    y2_lr = pred_down_al * 2.0 - 1.0
                y2_lr = y2_lr.repeat(1, self.config.data.channels, 1, 1)  # [1,3,256,256]
                y2_vec = y2_lr.view(y2_lr.size(0), -1)
                temp_y2 = F.interpolate(y2_lr, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False) # [1,3,512,512]

                # run ddnm for second stage (use same eta and sigma_y)
                x_noise2 = torch.randn(
                    1,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                    device=self.device,
                )

                with torch.no_grad():
                    if float(sigma_y) == 0.0:
                        out2_list, _, _, _ = self.ddnm_diffusion(
                            x_noise2, model, self.betas, self.args.eta, A_funcs_2, y2_vec, temp_y=temp_y2, cls_fn=cls_fn, classes=None, config=self.config, args=self.args
                        )
                    else:
                        out2_list, _, _, _ = self.ddnm_plus_diffusion(
                            x_noise2, model, self.betas, self.args.eta, A_funcs_2, y2_vec, float(sigma_y), temp_y=temp_y2, cls_fn=cls_fn, classes=None, config=self.config, args=self.args
                        )

                out2_t = out2_list[0].to(self.device)
                try:
                    out2_img = inverse_data_transform(self.config, out2_t)
                except Exception:
                    out2_img = torch.clamp((out2_t + 1.0) / 2.0, 0.0, 1.0)

                # save second-stage outputs
                os.makedirs(os.path.join(self.args.image_folder, "stage2_png_secondround"), exist_ok=True)
                tvu.save_image(out2_img[0, 0:1], os.path.join(self.args.image_folder, f"stage2_png_secondround/stage2_refined_{(idx_so_far):02d}.png"))
                np.save(os.path.join(self.args.image_folder, f"stage2_png_secondround/stage2_refined_{(idx_so_far):02d}.npy"), out2_img[0,0].detach().cpu().numpy())
                # append second-stage to dedicated list
                whole_result_stage2.append(out2_img[0,0].detach().cpu().numpy())
                # compute PSNR vs original (orig_1ch available above) for this second-round output
                pred2 = out2_img[0,0].to(self.device)
                mse2 = torch.mean((pred2 - orig_1ch.to(self.device)) ** 2)
                psnr2 = 10 * torch.log10(1 / (mse2 + 1e-12))
                avg_psnr_stage2 += float(psnr2)
                count_stage2 += 1
            # --- END SECOND ROUND ---

            # --- THIRD ROUND (8x only) ---
            if "2rounds" in self.args.image_folder and int(self.args.deg_scale) == 8:
                # prepare deg_img upsample to 256 (use ITER_DATA['deg_img'], which is LR e.g.64)
                deg_img = ITER_DATA["deg_img"][:,0:1]  # [1,1,64,64]
                # ensure on device and single-channel
                deg_img = deg_img.to(self.device)
                h_round2_down4x = self.config.data.image_size // int(self.args.deg_scale // 4)
                w_round2_down4x = self.config.data.image_size // int(self.args.deg_scale // 4)
                deg_up2 = F.interpolate(deg_img, size=(h_round2_down4x, w_round2_down4x), mode='bilinear', align_corners=False)  # [1,1,64,64] -> [1,1,64,64]

                out2_1ch = out2_img[:,0:1]
                pred_down2 = F.interpolate(out2_1ch, size=(h_round2_down4x, w_round2_down4x), mode='bilinear', align_corners=False) # [1,1,512,512]

                # compute linear align between pred_down and deg_up (solve a,b per-frame)
                p = pred_down2.view(-1)
                d = deg_up2.view(-1)
                p_mean = torch.mean(p)
                d_mean = torch.mean(d)
                p_var = torch.mean((p - p_mean) ** 2) + 1e-12
                cov = torch.mean((p - p_mean) * (d - d_mean))
                a = (cov / p_var).detach()
                b = (d_mean - a * p_mean).detach()

                # apply affine (a,b) to full-res prediction
                pred_down_al2 = (a * pred_down2 + b).clamp(0.0, 1.0)  # [1,1,h_round2_down4x,w_round2_down4x]

                # print(deg_up2.shape, pred_down2.shape, pred_down_al2.shape)

                # save aligned full-res as intermediate (optional)
                os.makedirs(os.path.join(self.args.image_folder, "round2_stage2_input_aligned"), exist_ok=True)
                tvu.save_image(pred_down_al2[0, 0:1], os.path.join(self.args.image_folder, f"round2_stage2_input_aligned/aligned_{(idx_so_far):02d}.png"))
                np.save(os.path.join(self.args.image_folder, f"round2_stage2_input_aligned/aligned_{(idx_so_far):02d}.npy"), pred_down_al2[0,0].detach().cpu().numpy())

                # Prepare measurement y for second ddnm: treat aligned full-res as if it were stage1 output,
                # convert to model domain then downsample to LR (256) as measurement
                try:
                    y2_lr = data_transform(self.config, pred_down_al2)  # [1,1,256,256] in model domain
                except Exception:
                    y2_lr = pred_down_al2 * 2.0 - 1.0
                y2_lr = y2_lr.repeat(1, self.config.data.channels, 1, 1)  # [1,3,256,256]
                y2_vec = y2_lr.view(y2_lr.size(0), -1)
                temp_y2 = F.interpolate(y2_lr, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False) # [1,3,512,512]

                # print(y2_lr.shape, temp_y2.shape)

                # run ddnm for second stage (use same eta and sigma_y)
                x_noise3 = torch.randn(
                    1,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                    device=self.device,
                )

                with torch.no_grad():
                    if float(sigma_y) == 0.0:
                        out3_list, _, _, _ = self.ddnm_diffusion(
                            x_noise3, model, self.betas, self.args.eta, A_funcs_3, y2_vec, temp_y=temp_y2, cls_fn=cls_fn, classes=None, config=self.config, args=self.args
                        )
                    else:
                        out3_list, _, _, _ = self.ddnm_plus_diffusion(
                            x_noise3, model, self.betas, self.args.eta, A_funcs_3, y2_vec, float(sigma_y), temp_y=temp_y2, cls_fn=cls_fn, classes=None, config=self.config, args=self.args
                        )

                out3_t = out3_list[0].to(self.device)
                try:
                    out3_img = inverse_data_transform(self.config, out3_t)
                except Exception:
                    out3_img = torch.clamp((out3_t + 1.0) / 2.0, 0.0, 1.0)
                # save third-stage outputs
                os.makedirs(os.path.join(self.args.image_folder, "stage3_png_thirdround"), exist_ok=True)
                tvu.save_image(out3_img[0, 0:1], os.path.join(self.args.image_folder, f"stage3_png_thirdround/stage3_refined_{(idx_so_far):02d}.png"))
                np.save(os.path.join(self.args.image_folder, f"stage3_png_thirdround/stage3_refined_{(idx_so_far):02d}.npy"), out3_img[0,0].detach().cpu().numpy())
                # append third-stage to dedicated list
                whole_result_stage3.append(out3_img[0,0].detach().cpu().numpy())
                # compute PSNR vs original (orig_1ch available above) for this third-round output
                pred3 = out3_img[0,0].to(self.device)
                mse3 = torch.mean((pred3 - orig_1ch.to(self.device)) ** 2)
                psnr3 = 10 * torch.log10(1 / (mse3 + 1e-12))
                avg_psnr_stage3 += float(psnr3)
                count_stage3 += 1

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        # print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average PSNR (stage1): %.2f" % avg_psnr)
        if count_stage2 > 0:
            avg_psnr_stage2_val = avg_psnr_stage2 / count_stage2
            print("Total Average PSNR (stage2 refined): %.2f (n=%d)" % (avg_psnr_stage2_val, count_stage2))
            # save whole_result_stage2 as numpy stack
            stage2_arr = np.stack(whole_result_stage2, axis=0).astype(np.float32)
            np.save(os.path.join(self.args.image_folder, "stage2_refined.npy"), stage2_arr)
            print("Saved stage2_refined.npy with shape", stage2_arr.shape)
        if count_stage3 > 0:
            avg_psnr_stage3_val = avg_psnr_stage3 / count_stage3
            print("Total Average PSNR (stage3 refined): %.2f (n=%d)" % (avg_psnr_stage3_val, count_stage3))
            # save whole_result_stage3 as numpy stack
            stage3_arr = np.stack(whole_result_stage3, axis=0).astype(np.float32)
            np.save(os.path.join(self.args.image_folder, "stage3_refined.npy"), stage3_arr)
            print("Saved stage3_refined.npy with shape", stage3_arr.shape)
        print("Number of samples: %d" % (idx_so_far - idx_init))

        dir_start_steps = os.path.join(self.args.image_folder, "start_steps")
        os.makedirs(os.path.join(dir_start_steps), exist_ok=True)

        return whole_result_denorm, whole_result_denorm_noAddMin, whole_result_norm, whole_result

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

def inverse_data_transform_noClamp(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return X # expected range: [0.0, 1.0]
    # torch.clamp(X, 0.0, 1.0)
