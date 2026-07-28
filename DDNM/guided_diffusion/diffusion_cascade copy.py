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

        # from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion
        # from functions.svd_ddnm_perProj import ddnm_diffusion, ddnm_plus_diffusion
        # if getattr(self.args, "use_pas", False):
        #     from functions.svd_ddnm_perProj import ddnm_diffusion, ddnm_plus_diffusion
        # if getattr(self.args, "use_cascade", False):
        from functions.svd_ddnm_cascade import ddnm_diffusion, ddnm_plus_diffusion

        self.ddnm_diffusion = ddnm_diffusion
        self.ddnm_plus_diffusion = ddnm_plus_diffusion

    def _stage2_refine_stack(self, model, stack_np, out_folder, steps=10, eta=0.9, noise_level=0.01):
        """
        stack_np: (N,H,W) float32 in [0,1]
        Performs a short DDIM-like denoising refinement per-frame (or batch all frames) starting
        from the temporally-smoothed images. Saves refined stack npy and PNGs to out_folder.
        """
        os.makedirs(out_folder, exist_ok=True)
        model.eval()
        # prepare tensor: (N, C, H, W)
        N, H, W = stack_np.shape
        channels = getattr(self.config.data, "channels", 1)
        t_in = torch.from_numpy(stack_np).float().to(self.device)  # (N,H,W)
        if channels == 1:
            x_t = t_in.unsqueeze(1)  # (N,1,H,W)
        else:
            x_t = t_in.unsqueeze(1).repeat(1, channels, 1, 1)

        # map [0,1] -> [-1,1] if model expects
        try:
            x_t = x_t * 2.0 - 1.0
        except Exception:
            pass

        # add small noise
        x = x_t + torch.randn_like(x_t) * float(noise_level)
        x = x.to(self.device)

        # choose timesteps: last `steps` timesteps of diffusion schedule
        max_t = int(self.num_timesteps - 1)
        ts = list(range(max_t, max(0, max_t - steps), -1))
        for t_val in ts:
            t = torch.ones(x.size(0), device=self.device, dtype=torch.long) * int(t_val)
            with torch.no_grad():
                et = model(x, t)
            if et.size(1) == 6:
                et = et[:, :3]
            at = compute_alpha(self.betas, t)
            at_sqrt = at.sqrt()
            one_minus_at_sqrt = (1 - at).sqrt()
            # estimate x0_t
            x0_t = (x - et * one_minus_at_sqrt) / at_sqrt
            # DDIM-like step to next (t-1)
            next_t = max(0, t_val - 1)
            at_next = compute_alpha(self.betas, torch.ones_like(t) * next_t)
            c1 = (1 - at_next).sqrt() * eta
            # conservative update (no extra noise)
            x = at_next.sqrt() * x0_t + c1 * et
            # clamp occasionally to keep numeric stable
            x = torch.clamp(x, -1.0, 1.0)

        # map back to [0,1]
        x_img = (x.detach().cpu() + 1.0) / 2.0
        if x_img.ndim == 4:
            if x_img.size(1) > 1:
                out_gray = x_img[:, 0, :, :].numpy()
            else:
                out_gray = x_img[:, 0, :, :].numpy()
        else:
            out_gray = x_img.squeeze().numpy()

        # save results
        np.save(os.path.join(out_folder, f"whole_stage2_refined_steps{steps}.npy"), out_gray)
        # save pngs
        from torchvision.utils import save_image
        os.makedirs(os.path.join(out_folder, "stage2_pngs"), exist_ok=True)
        for i in range(out_gray.shape[0]):
            img = torch.from_numpy(out_gray[i:i+1]).unsqueeze(0)  # (1,1,H,W)
            save_image(img, os.path.join(out_folder, "stage2_pngs", f"pred_{i:02d}.png"))
        return out_gray

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

            ## Check model keys
            # with open("model_keys.txt", "w") as f:
            #     for key in model.state_dict().keys():
            #         f.write(f"{key}\n")
            # exit()

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
            if getattr(self.args, "stage2_only", False):
                stage1_dir = getattr(self.args, "stage1_dir", "")
                if not stage1_dir or not os.path.exists(stage1_dir):
                    raise FileNotFoundError(f"stage1_dir not found: {stage1_dir}")

                # Prefer temporally-smoothed file
                sigma_t = getattr(self.args, "temporal_sigma_t", None)
                candidate = None
                if sigma_t is not None:
                    fname = f"whole_gaussian_filter1d_sigma_t{sigma_t}.npy"
                    p = os.path.join(stage1_dir, fname)
                    if os.path.exists(p):
                        candidate = p

                if candidate is None:
                    for fn in ["whole_gaussian_filter1d_sigma_t1.0.npy", "whole.npy", "original_stack.npy"]:
                        p = os.path.join(stage1_dir, fn)
                        if os.path.exists(p):
                            candidate = p
                            break


                # load stack (N,H,W) normalized [0,1]
                if candidate is not None:
                    stack = np.load(candidate)
                    if stack.ndim == 4:
                        stack = stack[:, 0, :, :]
                else:
                    # fallback to pred_png folder
                    png_dir = os.path.join(stage1_dir, "pred_png")
                    if os.path.isdir(png_dir):
                        files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
                        arrs = []
                        for fpath in files:
                            im = plt.imread(os.path.join(png_dir, fpath))
                            if im.ndim == 3:
                                im = im[..., 0]
                            arrs.append(im.astype(np.float32))
                        stack = np.stack(arrs, axis=0)
                    else:
                        raise FileNotFoundError("No stage1 output (npy/png) found in stage1_dir")

                if "cycle" in self.args.image_folder:
                    candidate = "/Alexandrite/jhnoh/r2_gaussian/NAB_GS_mela0050.pickle"
                    with open(candidate, 'rb') as f:
                        data = pickle.load(f)
                    # print(data["train"]["projections"][0].shape)
                    # print(data["train"]["projections"].shape)
                    # # min max of print(data["train"]["projections"][0])
                    # print(data["train"]["projections"][0].min(), data["train"]["projections"][0].max())
                    # exit()
                    stack = data["train"]["projections"] # [100, 512, 512]
                    # min-max norm (100장 projections에 대해 1번만 시행)
                    stack = np.array(stack)
                    # print("stack min/max before norm:", np.min(stack), np.max(stack))
                    stack = (stack - stack.min()) / (stack.max() - stack.min() + 1e-8)

                stack = stack.astype(np.float32)
                N, H, W = stack.shape
                print(f"Loaded stage1 stack from {stage1_dir}, shape={stack.shape}")

                # prepare Denoising operator (A_funcs) and other params
                # choose A_funcs based on requested deg for stage2
                deg_mode = getattr(self.args, "deg", "denoising")
                if deg_mode == "highfreq":
                    from functions.highFrequency_deg import HighFreqDegradation
                    cutoff = getattr(self.args, "hf_cutoff", 192.0)
                    A_funcs = HighFreqDegradation(self.config.data.channels,
                                                  self.config.data.image_size,
                                                  cutoff,
                                                  self.device)
                elif deg_mode == "sr_averagepooling":
                    # use SuperResolution operator: stage1 (512) -> downsample to LR and use as measurement
                    from functions.svd_operators import SuperResolution
                    blur_by = int(getattr(self.args, "deg_scale", 4)) // 2  # deg_scale as integer downsampling factor div 2
                    A_funcs = SuperResolution(self.config.data.channels, self.config.data.image_size, blur_by, self.device)
                else:
                    from functions.svd_operators import Denoising
                    A_funcs = Denoising(self.config.data.channels, self.config.data.image_size, self.device)

                # Stage2 control params (from CLI)
                eta2 = float(getattr(self.args, "eta", 0.9))      # DDIM / DDNM eta (deterministic <-> stochastic)
                sigma_y = float(getattr(self.args, "sigma_y", 0.0))      # measurement noise used by ddnm_plus (if >0)
                # SR multi-stage params
                sr_gauss_sigma = float(getattr(self.args, "sr_stage_gauss_sigma", 1.0))

                # os.makedirs(os.path.join(self.args.image_folder, "stage2_pngs"), exist_ok=True)
                # stage2_out = []

                # Process per-frame using existing ddnm functions (svd_ddnm_cascade.ddnm_diffusion / ddnm_plus_diffusion)
                # to tqdm to check progress
                # for i in tqdm.tqdm(range(N), desc="Stage2 DDNM per-frame"):
                #     frame = stack[i:i+1]  # (1,H,W)
                #     # build x_deg tensor: (1, C, H, W)
                #     c = self.config.data.channels if hasattr(self.config.data, "channels") else 1
                #     x_deg = torch.from_numpy(frame).unsqueeze(1).float().to(self.device)  # (1,1,H,W)
                #     if c > 1:
                #         x_deg = x_deg.repeat(1, c, 1, 1)
                    
                #     # map to model domain using data_transform if available
                #     try:
                #         y_tensor = data_transform(self.config, x_deg)  # expects [0,1] -> model domain
                #     except Exception:
                #         y_tensor = x_deg * 2.0 - 1.0

                #     # define temp_y early so debug prints can use it safely (baseline image in model domain)
                #     temp_y = y_tensor.clone()

                #     b, ch, h, w = y_tensor.size()
                #     # build measurement vector y depending on requested degration for stage2
                #     deg_mode = getattr(self.args, "deg", "denoising")
                #     if deg_mode == "highfreq":
                #         # A_funcs.A expects image-shaped [B,C,H,W] and returns flattened HF vector [B, M]
                #         y_vec = A_funcs.A(y_tensor)  # full-res -> HF measurement
                #     elif deg_mode == "sr_averagepooling":
                #         # downsample stage1 output to LR measurement size before forming y_vec
                #         scale = int(round(getattr(self.args, "deg_scale", 4))) // 2
                #         lr_h = max(1, self.config.data.image_size // scale)
                #         # y_tensor is in model domain (e.g. [-1,1]); downsample in that domain
                #         y_lr = F.interpolate(y_tensor, size=(lr_h, lr_h), mode='bilinear', align_corners=False)
                #         # upsample y_lr into [512,512] again
                #         # y_lr_upsampled = F.interpolate(y_lr, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)
                #         # use upsampled y_lr as temp_y (baseline image in model domain)
                #         temp_y = y_lr.clone()
                #         # If A_funcs expects an image input, pass downsampled image; else flatten
                #         # if hasattr(A_funcs, "A"):
                #         #     y_vec = A_funcs.A(y_lr)
                #         # else:
                #         y_vec = y_lr.reshape((b, -1))
                #         # ensure flattened vector (ddnm expects flattened y)
                #         if isinstance(y_vec, torch.Tensor) and y_vec.dim() == 4:
                #             y_vec = y_vec.view(b, -1)
                #     else:
                #         hwc = ch * h * w
                #         y_vec = y_tensor.reshape((b, hwc))
                    
                #     # save y for debugging
                #     # y_img = (y_tensor.detach().cpu() + 1.0) / 2.0
                #     # tvu.save_image(y_img[0, 0:1], "temp_y_img.png")

                #     # save y_vec as image for debugging
                #     # try to invert A to image domain for visualization
                #     # try:
                #     #     if getattr(self.args, "deg", "denoising") == "highfreq":
                #     #         # Prefer At if available, else fall back to A_pinv
                #     #         if hasattr(A_funcs, "At"):
                #     #             y_inv = A_funcs.At(y_vec)
                #     #         elif hasattr(A_funcs, "A_pinv"):
                #     #             y_inv = A_funcs.A_pinv(y_vec)
                #     #         else:
                #     #             raise AttributeError("A_funcs has no At or A_pinv")
                #     #         # ensure image-shaped
                #     #         if y_inv.dim() == 2:
                #     #             y_inv = y_inv.view(b, ch, h, w)
                #     #     else:
                #     #         y_inv = y_vec.reshape((b, ch, h, w))

                #         # -- DEBUG: print ranges and stats to logfile/stdout --
                #         # print("DEBUG y_tensor range:", float(y_tensor.min()), float(y_tensor.max()), "mean", float(y_tensor.mean()), "std", float(y_tensor.std()))
                #         # print("DEBUG y_vec shape/range:", getattr(y_vec, "shape", None), np.nanmin(y_vec.detach().cpu().numpy()), np.nanmax(y_vec.detach().cpu().numpy()))
                #         # print("DEBUG y_inv shape/range:", getattr(y_inv, "shape", None), float(y_inv.min()), float(y_inv.max()), "mean", float(y_inv.mean()), "std", float(y_inv.std()))
                #         # print("DEBUG temp_y shape/range:", getattr(temp_y, "shape", None), float(temp_y.min()), float(temp_y.max()), "mean", float(temp_y.mean()), "std", float(temp_y.std()))

                #         # print("DEBUG y_tensor range:", float(y_tensor.min()), float(y_tensor.max()), "mean", float(y_tensor.mean()), "std", float(y_tensor.std()))
                #         # try:
                #         #     yvec_np = y_vec.detach().cpu().numpy()
                #         #     print("DEBUG y_vec shape:", getattr(y_vec, "shape", None), "min/max:", yvec_np.min(), yvec_np.max())
                #         # except Exception:
                #         #     print("DEBUG y_vec shape:", getattr(y_vec, "shape", None))
                #         # print("DEBUG y_inv shape/range:", getattr(y_inv, "shape", None), float(y_inv.min()), float(y_inv.max()), "mean", float(y_inv.mean()), "std", float(y_inv.std()))
                #         # print("DEBUG temp_y shape/range:", getattr(temp_y, "shape", None), float(temp_y.min()), float(temp_y.max()), "mean", float(temp_y.mean()), "std", float(temp_y.std()))

                #         # for visualization convert model-domain (-1..1) -> [0,1] if applicable
                #         # but ensure NOT to change y_inv numerically for ddnm use (this is only for saving)
                #     #     try:
                #     #         vis = (y_inv.detach().cpu() + 1.0) / 2.0
                #     #         vis = torch.clamp(vis, 0.0, 1.0)
                #     #     except Exception:
                #     #         vis = y_inv.detach().cpu()
                #     #     tvu.save_image(vis[0, 0:1], "temp_y_vec_inverted.png")
                #     # except Exception as e:
                #     #     print("Inversion of y_vec failed:", e)
                #     #     try:
                #     #         print("y_vec type:", type(y_vec), "shape:", getattr(y_vec, "shape", None))
                #     #     except Exception:
                #     #         pass
                #     #     # fallback: save numeric summary to help debugging
                #     #     try:
                #     #         if isinstance(y_vec, torch.Tensor):
                #     #             ym = y_vec.detach().cpu().numpy()
                #     #             np.save("debug_y_vec.npy", ym)
                #     #             print("Saved debug_y_vec.npy")
                #     #     except Exception:
                #     #         pass
                    
                #     # init x noise for ddnm (match image_size)
                #     x_noise = torch.randn(
                #         b,
                #         self.config.data.channels,
                #         self.config.data.image_size,
                #         self.config.data.image_size,
                #         device=self.device,
                #     )

                #     # temp_y used inside ddnm functions (baseline image in image-domain)
                #     # for highfreq we still use the full image baseline (stage1 smoothed) as temp_y
                #     temp_y = y_tensor.clone()

                #     with torch.no_grad():
                #         # call appropriate ddnm routine
                #         if float(sigma_y) == 0.0:
                #             out_list, _, _, _ = self.ddnm_diffusion(
                #                 x_noise, model, self.betas, eta2, A_funcs, y_vec, temp_y=temp_y, cls_fn=cls_fn,
                #                 classes=None, config=self.config, args=self.args
                #             )
                #         else:
                #             out_list, _, _, _ = self.ddnm_plus_diffusion(
                #                 x_noise, model, self.betas, eta2, A_funcs, y_vec, float(sigma_y), temp_y=temp_y,
                #                 cls_fn=cls_fn, classes=None, config=self.config, args=self.args
                #             )
                #     # out_list is list of tensors (x0 predictions). Take first and convert to image domain
                #     out_t = out_list[0].to(self.device)

                #     try:
                #         out_img = inverse_data_transform(self.config, out_t)
                #     except Exception:
                #         out_img = torch.clamp((out_t + 1.0) / 2.0, 0.0, 1.0)

                #     # # value range check (y_tensor, temp_y)
                #     # print(f"Apy min/max: {y_tensor.min().item():.4f} / {y_tensor.max().item():.4f}")
                #     # print(f"temp_y min/max: {temp_y[i][0].min().item():.4f} / {temp_y[i][0].max().item():.4f}")

                #     # # value range check out_t
                #     # print(f"out_t[0] min/max: {out_t[0].min().item():.4f} / {out_t[0].max().item():.4f}")
                #     # # value range check out_img
                #     # print(f"out_img[0] min/max: {out_img[0].min().item():.4f} / {out_img[0].max().item():.4f}")

                #     out_gray = out_img[0, 0].detach().cpu().numpy()
                #     stage2_out.append(out_gray)

                #     # save png
                #     tvu.save_image(out_img[0, 0:1], os.path.join(self.args.image_folder, "stage2_pngs", f"pred_{i:02d}.png"))

                    # exit(
                # --- SR multi-stage processing ---
                # define per-mode stage levels:
                deg_mode = getattr(self.args, "deg", "sr_averagepooling")
                if deg_mode == "sr_averagepooling":
                    # interpret deg_scale: if 4 -> single-level (512->256). if 8 -> two levels [4,2] as requested.
                    ds = int(round(getattr(self.args, "deg_scale", 4)))
                    if ds == 8:
                        levels = [4, 2]
                    elif ds == 4:
                        levels = [2]
                    else:
                        # fallback: single downsample by ds//? treat ds as overall factor -> produce factor ds/ (ds//2)
                        levels = [max(2, ds // 2)]
                else:
                    levels = []

                # y_source holds current input images (0..1) for the next stage; initialize from stage1 stack
                y_source_np = stack.copy()  # shape (N,H,W) in [0,1]

                for level_idx, level in enumerate(levels):
                    print(f"SR stage {level_idx+1}/{len(levels)}: downsample factor {level} -> LR size {512//level}")
                    stage_out = []
                    # configure A_funcs for this level
                    from functions.svd_operators import SuperResolution
                    A_funcs_lvl = SuperResolution(self.config.data.channels, self.config.data.image_size, level, self.device)

                    for i in tqdm.tqdm(range(N), desc=f"Stage{level_idx+1} DDNM per-frame"):
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
                        lr_h = max(1, self.config.data.image_size // level)
                        # y_lr = F.interpolate(y_tensor, size=(lr_h, lr_h), mode='bilinear', align_corners=False)
                        # y_lr by average pooling to match A_funcs behavior
                        y_lr = F.avg_pool2d(y_tensor, kernel_size=level, stride=level)
                        
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
                        # if "cycle" in self.args.image_folder:
                        #     temp_y = F.interpolate(temp_y, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)

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
                        os.makedirs(os.path.join(self.args.image_folder, f"stage2_pngs_lvl{level_idx}"), exist_ok=True)
                        # save png
                        tvu.save_image(out_img[0, 0:1], os.path.join(self.args.image_folder, f"stage2_pngs_lvl{level_idx}", f"stage2_lvl_{level_idx}_pred_{i:02d}.png"))

                    stage_out_arr = np.stack(stage_out, axis=0).astype(np.float32)  # (N,H,W) in [0,1]
                    # save pre-GF result
                    pre_name = os.path.join(self.args.image_folder, f"stage2_lvl{level_idx}_preGF.npy")
                    np.save(pre_name, stage_out_arr)
                    print(f"Saved stage {level_idx+1} pre-GF -> {pre_name}")

                    # apply spatial gaussian filter per-frame (post-processing) and save
                    try:
                        from scipy.ndimage import gaussian_filter
                        post_arr = np.stack([gaussian_filter(img, sigma=sr_gauss_sigma) for img in stage_out_arr], axis=0)
                    except Exception:
                        post_arr = stage_out_arr.copy()
                    post_name = os.path.join(self.args.image_folder, f"stage2_lvl{level_idx}_postGF.npy")
                    np.save(post_name, post_arr)
                    print(f"Saved stage {level_idx+1} post-GF -> {post_name}")

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
                    if level_idx + 1 < len(levels):
                        # downsample post_arr (current SR results) to next LR resolution (next_level)
                        next_level = levels[level_idx + 1]
                        next_lr_h = max(1, self.config.data.image_size // next_level)
                        # post_arr currently full-res (512). downsample to next_lr_h -> then upsample to full-res during next stage temp_y creation
                        post_t = torch.from_numpy(post_arr).unsqueeze(1).to(self.device)  # (N,1,H,W)
                        post_down = F.interpolate(post_t, size=(next_lr_h, next_lr_h), mode='bilinear', align_corners=False)
                        # upsample back to full-res for y_source_np content (so next iteration's temp_y uses this baseline)
                        post_up = F.interpolate(post_down, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)
                        y_source_np = post_up.squeeze(1).detach().cpu().numpy()
                    else:
                        # final output
                        final_name = os.path.join(self.args.image_folder, f"stage2_final.npy")
                        np.save(final_name, post_arr)
                        print(f"Saved final stage output -> {final_name}")
                        # also make stage2_out equal to final for backward compatibility
                        # stage2_out = post_arr
                # end multi-stage
    
                # after all stages finished, run single unified metrics calculation:
                try:
                    # sigma_t_val = float(getattr(self.args, "temporal_sigma_t", 1.0))
                    # self.compute_and_save_metrics(stage1_dir, self.args.image_folder, sigma_t=sigma_t_val)
                    self.compute_and_save_metrics_cascade(stage1_dir, self.args.image_folder)
                except Exception as e:
                    print("compute_and_save_metrics failed:", e)                
                return # stage2_arr
            
            print('Run SVD-based DDNM.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            result = self.svd_based_ddnm_plus(model, cls_fn)
            
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

        for idx, ITER_DATA in enumerate(pbar):
            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 512, 512]

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

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm -> HERE ##
                    x, x0_preds, x0t_preds, start_step = self.ddnm_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)
                else: # noisy case, turn to ddnm+
                    x, x0_preds, x0t_preds, start_step = self.ddnm_plus_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)

            start_steps.append(start_step)
            x = [inverse_data_transform(config, xi) for xi in x]

            os.makedirs(os.path.join(self.args.image_folder, "pred_png"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "pred_npy"), exist_ok=True)
            # for j in range(x[0].size(0)):
            #     # x[0].shape: [3, 512, 512]
            #     # x[0] to [512, 512]
            #     x_1ch = x[0][j][0]
            #     tvu.save_image(
            #         x_1ch, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far + j):02d}.png")
            #     )
            #     whole_result.append(x[0][j].detach().cpu().numpy())
            #     plt.title("x0_pred Changes over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("L2 Norm Change")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0_changes.png"))
            #     plt.close()

            #     # x0t_pred 변화량 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(x0t_changes)
            #     plt.title("x0t_pred Changes over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("L2 Norm Change")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0t_changes.png"))
            #     plt.close()

            #     # x0t_pred PSNR 변화 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(psnr_values)
            #     plt.title("PSNR between x0t_pred and Original Image over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("PSNR (dB)")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/psnr_x0t_orig.png"))
            #     plt.close()

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

        for idx, ITER_DATA in enumerate(pbar):
            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 512, 512]

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

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm -> HERE ##
                    x, x0_preds, x0t_preds, start_step = self.ddnm_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)
                else: # noisy case, turn to ddnm+
                    x, x0_preds, x0t_preds, start_step = self.ddnm_plus_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)

            start_steps.append(start_step)
            x = [inverse_data_transform(config, xi) for xi in x]

            os.makedirs(os.path.join(self.args.image_folder, "pred_png"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "pred_npy"), exist_ok=True)
            # for j in range(x[0].size(0)):
            #     # x[0].shape: [3, 512, 512]
            #     # x[0] to [512, 512]
            #     x_1ch = x[0][j][0]
            #     tvu.save_image(
            #         x_1ch, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far + j):02d}.png")
            #     )
            #     whole_result.append(x[0][j].detach().cpu().numpy())
            #     plt.title("x0_pred Changes over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("L2 Norm Change")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0_changes.png"))
            #     plt.close()

            #     # x0t_pred 변화량 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(x0t_changes)
            #     plt.title("x0t_pred Changes over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("L2 Norm Change")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0t_changes.png"))
            #     plt.close()

            #     # x0t_pred PSNR 변화 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(psnr_values)
            #     plt.title("PSNR between x0t_pred and Original Image over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("PSNR (dB)")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/psnr_x0t_orig.png"))
            #     plt.close()

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

        for idx, ITER_DATA in enumerate(pbar):
            x_deg = ITER_DATA["deg_img"] # [1, 3, 128, 128] or [1, 3, 64, 64]
            x_orig = ITER_DATA["gt_img"] # [1, 3, 512, 512]

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

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm -> HERE ##
                    x, x0_preds, x0t_preds, start_step = self.ddnm_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)
                else: # noisy case, turn to ddnm+
                    x, x0_preds, x0t_preds, start_step = self.ddnm_plus_diffusion(
                        x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, temp_y=temp_y, cls_fn=cls_fn, classes=None, config=config, args=self.args)

            start_steps.append(start_step)
            x = [inverse_data_transform(config, xi) for xi in x]

            os.makedirs(os.path.join(self.args.image_folder, "pred_png"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "pred_npy"), exist_ok=True)
            # for j in range(x[0].size(0)):
            #     # x[0].shape: [3, 512, 512]
            #     # x[0] to [512, 512]
            #     x_1ch = x[0][j][0]
            #     tvu.save_image(
            #         x_1ch, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far + j):02d}.png")
            #     )
            #     whole_result.append(x[0][j].detach().cpu().numpy())
            #     plt.title("x0_pred Changes over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("L2 Norm Change")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0_changes.png"))
            #     plt.close()

            #     # x0t_pred 변화량 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(x0t_changes)
            #     plt.title("x0t_pred Changes over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("L2 Norm Change")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0t_changes.png"))
            #     plt.close()

            #     # x0t_pred PSNR 변화 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(psnr_values)
            #     plt.title("PSNR between x0t_pred and Original Image over Timesteps")
            #     plt.xlabel("Timestep")
            #     plt.ylabel("PSNR (dB)")
            #     plt.savefig(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/psnr_x0t_orig.png"))
            #     plt.close()

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

        # save start_steps as a txt and plot
        # make dir_start_steps
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

        return whole_result

    def compute_and_save_metrics_cascade(self, stage1_dir, stage2_dir=None):
        """
        Compute PSNR/SSIM/LPIPS for:
          - stage1 (loaded from a pickled GS file here)
          - any stage2 files saved as stage2_lvl{idx}_preGF.npy and stage2_lvl{idx}_postGF.npy
        Saves per-stage CSV and JSON summary and a combined metrics JSON into self.args.image_folder.
        """
        combined_path = os.path.join(self.args.image_folder, "metrics_combined_summary.json")
        if os.path.exists(combined_path):
            try:
                with open(combined_path, "r") as f:
                    metrics = json.load(f)
                print("Loaded existing metrics from", combined_path)
                return metrics
            except Exception:
                pass

        def load_npy_if_exists(path):
            if path and os.path.exists(path):
                a = np.load(path)
                # collapse channel dim if present
                if a.ndim == 4 and a.shape[1] == 1:
                    a = a[:, 0, :, :]
                return a.astype(np.float32)
            return None

        # --- load stage1 from hardcoded candidate (existing behavior) ---
        candidate = "/Alexandrite/jhnoh/r2_gaussian/NAB_GS_mela0050.pickle"
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Expected stage1 pickle not found: {candidate}")
        with open(candidate, 'rb') as f:
            data = pickle.load(f)
        out_stage1 = data["train"]["projections"].astype(np.float32)  # (N, 512, 512)
        # per-projection min-max normalize -> 0..255 (preserve current pipeline behavior)
        for i in range(out_stage1.shape[0]):
            mn = float(np.min(out_stage1[i])); mx = float(np.max(out_stage1[i]))
            out_stage1[i] = (out_stage1[i] - mn) / (mx - mn) * 255.0
            # if mx - mn > 1e-8:
            #     out_stage1[i] = (out_stage1[i] - mn) / (mx - mn) * 255.0
            # else:
            #     out_stage1[i] = out_stage1[i] * 255.0

        # create a temporally-smoothed stage1 (same name convention as elsewhere)
        sigma_t = float(getattr(self.args, "temporal_sigma_t", 1.0))
        try:
            out_stage1_sm = gaussian_filter1d(out_stage1, sigma=sigma_t, axis=0)
        except Exception:
            out_stage1_sm = out_stage1.copy()

        # --- discover stage2 files ---
        stage2_pre_dict = {}
        # stage2_post_dict = {}
        if stage2_dir is not None and os.path.isdir(stage2_dir):
            pre_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_preGF.npy")))
            # post_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_postGF.npy")))
            for p in pre_files:
                base = os.path.basename(p)
                try:
                    idx = int(base.split("stage2_lvl")[1].split("_")[0])
                except Exception:
                    idx = len(stage2_pre_dict)
                stage2_pre_dict[idx] = load_npy_if_exists(p)
            # for p in post_files:
            #     base = os.path.basename(p)
            #     try:
            #         idx = int(base.split("stage2_lvl")[1].split("_")[0])
            #     except Exception:
            #         idx = len(stage2_post_dict)
            #     stage2_post_dict[idx] = load_npy_if_exists(p)

        # --- build GT stack ---
        def build_gt_stack():
            try:
                _, test_dataset = get_dataset_cascade(self.args, self.config)
                gt_list = []
                for item in test_dataset:
                    gt = item["gt_img"] * 255. # range: [0,255] -> 그대로 stack하면 됨.
                    # gt = None
                    # if isinstance(item, dict) and "gt_img" in item:
                    #     gt = item["gt_img"]
                    # elif isinstance(item, (list, tuple)) and len(item) > 1:
                    #     gt = item[1] if isinstance(item[1], torch.Tensor) else None
                    # if gt is None:
                    #     continue
                    # if isinstance(gt, torch.Tensor):
                    #     g = inverse_data_transform(self.config, gt).cpu().numpy()
                    # else:
                    #     g = gt
                    # if g.ndim == 3:
                    #     g = g[0]
                    # g = g.astype(np.float32)
                    # if np.nanmax(g) <= 1.01:
                    #     g = g * 255.0
                    gt_list.append(gt)
                if len(gt_list) > 0:
                    return np.stack(gt_list, axis=0)
            except Exception as e:
                print("compute_and_save_metrics_cascade: failed to build GT stack:", e)
            return None

        gt_stack = build_gt_stack()  # expected in 0..255
        if gt_stack is None:
            print("GT stack not available - metrics will be skipped.")
        else:
            # ensure dtype float32
            gt_stack = gt_stack.astype(np.float32)

        # optional LPIPS model
        lpips_model = None
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='vgg').to(self.device).eval()
            print("LPIPS model loaded.")
        except Exception:
            lpips_model = None
            print("LPIPS model not available - LPIPS metric will be skipped.")

        # helper prepare
        def ensure_255(arr):
            a = arr.astype(np.float32)
            if np.nanmax(a) <= 1.01:
                a = a * 255.0
            return a

        def compute_and_save(name, pred_stack):
            out = {"n": 0}
            if gt_stack is None or pred_stack is None:
                print(f"SKIP metrics {name}: GT or pred not available")
                return None
            pred = ensure_255(pred_stack)
            N = min(pred.shape[0], gt_stack.shape[0])
            ps = np.full(N, np.nan, dtype=np.float32)
            ss = np.full(N, np.nan, dtype=np.float32)
            lp = np.full(N, np.nan, dtype=np.float32) if lpips_model is not None else None

            for i in tqdm.tqdm(range(N), desc=f"Computing metrics for {name}"):
                gt_i = gt_stack[i].astype(np.float32) # [3, 512, 512]
                gt_i = gt_i[0]
                pred_i = pred[i].astype(np.float32) # [512, 512]

                # GT, pred min max
                # print("gt_i min/max:", np.min(gt_i), np.max(gt_i))
                # print("pred_i min/max:", np.min(pred_i), np.max(pred_i))
                # exit()

                ps[i] = psnr_fn(gt_i, pred_i, data_range=255.0)
                ss[i] = ssim_fn(gt_i, pred_i, data_range=255.0)

                # try:
                #     ps[i] = psnr_fn(gt_i, pred_i, data_range=255.0)
                # except Exception:
                #     ps[i] = np.nan
                # try:
                #     ss[i] = ssim_fn(gt_i, pred_i, data_range=255.0)
                # except Exception:
                #     ss[i] = np.nan
                if lpips_model is not None:
                    # lpips expects torch tensor in [-1,1] and 3 channels
                    p_t = torch.from_numpy(pred_i / 255.0).float()
                    g_t = torch.from_numpy(gt_i / 255.0).float()
                    if p_t.ndim == 2:
                        p_t = p_t.unsqueeze(0)
                    if g_t.ndim == 2:
                        g_t = g_t.unsqueeze(0)
                    # make 3 channels by repeating
                    p_t3 = p_t.unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
                    g_t3 = g_t.unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
                    # to [-1,1]
                    p_t3 = (p_t3 * 2.0) - 1.0
                    g_t3 = (g_t3 * 2.0) - 1.0
                    with torch.no_grad():
                        lpv = lpips_model(p_t3, g_t3)
                    lp[i] = float(lpv.cpu().numpy().ravel()[0])
                    # try:
                    #     # lpips expects torch tensor in [-1,1] and 3 channels
                    #     p_t = torch.from_numpy(pred_i / 255.0).float()
                    #     g_t = torch.from_numpy(gt_i / 255.0).float()
                    #     if p_t.ndim == 2:
                    #         p_t = p_t.unsqueeze(0)
                    #     if g_t.ndim == 2:
                    #         g_t = g_t.unsqueeze(0)
                    #     # make 3 channels by repeating
                    #     p_t3 = p_t.unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
                    #     g_t3 = g_t.unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
                    #     # to [-1,1]
                    #     p_t3 = (p_t3 * 2.0) - 1.0
                    #     g_t3 = (g_t3 * 2.0) - 1.0
                    #     with torch.no_grad():
                    #         lpv = lpips_model(p_t3, g_t3)
                    #     lp[i] = float(lpv.cpu().numpy().ravel()[0])
                    # except Exception:
                    #     lp[i] = np.nan

            summary = {
                "n": int(N),
                "psnr_mean": float(np.nanmean(ps)),
                "psnr_std": float(np.nanstd(ps)),
                "ssim_mean": float(np.nanmean(ss)),
                "ssim_std": float(np.nanstd(ss))
            }
            if lpips_model is not None:
                summary["lpips_mean"] = float(np.nanmean(lp))
                summary["lpips_std"] = float(np.nanstd(lp))

            # save per-frame CSV
            csv_path = os.path.join(self.args.image_folder, f"metrics_{name}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                header = ["idx", "psnr", "ssim"]
                if lpips_model is not None:
                    header.append("lpips")
                writer.writerow(header)
                for j in range(N):
                    row = [j, float(ps[j]) if not np.isnan(ps[j]) else "", float(ss[j]) if not np.isnan(ss[j]) else ""]
                    if lpips_model is not None:
                        row.append(float(lp[j]) if not np.isnan(lp[j]) else "")
                    writer.writerow(row)

            # save json summary
            with open(os.path.join(self.args.image_folder, f"summary_{name}.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # print in requested format
            print(f"{name} -> PSNR {summary['psnr_mean']:.4f} _ {summary['psnr_std']:.4f}, SSIM {summary['ssim_mean']:.4f} _ {summary['ssim_std']:.4f} LPIPS {summary['lpips_mean']:.4f} _ {summary['lpips_std']:.4f}")

            return summary

        metrics = {}
        # stage1 whole & smoothed
        # metrics['stage1_whole'] = compute_and_save("stage1_whole", out_stage1)
        # metrics['stage1_smoothed'] = compute_and_save("stage1_smoothed", out_stage1_sm)

        metrics['out_gs'] = compute_and_save("out_gs", out_stage1)

        # stage2 levels (sorted by idx)
        # for idx in sorted(set(list(stage2_pre_dict.keys()) + list(stage2_post_dict.keys()))):
        for idx in sorted(stage2_pre_dict.keys()):
            pre = stage2_pre_dict.get(idx, None)
            # post = stage2_post_dict.get(idx, None)
            name_pre = f"stage2_sr{idx+1}_pre"
            # name_post = f"stage2_sr{idx+1}_post"
            if pre is not None:
                metrics[name_pre] = compute_and_save(name_pre, pre)
            # if post is not None:
            #     metrics[name_post] = compute_and_save(name_post, post)

        # write combined json
        with open(combined_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def compute_and_save_metrics(self, stage1_dir, stage2_dir=None, sigma_t=1.0):
        """
        Compute PSNR/SSIM using:
          - stage1_dir: folder with stage1 whole.npy and whole_gaussian_filter1d_sigma_t{sigma}.npy
          - stage2_dir: folder containing stage2_lvl{idx}_preGF.npy and stage2_lvl{idx}_postGF.npy
        Only stage2_lvl* files are considered (do NOT map stage2_final.npy into extra level).
        Results saved as metrics_{name}.csv and summary_{name}.json in self.args.image_folder and a combined json.
        """
        # avoid duplicate work: if combined summary already exists, load & return it
        combined_path = os.path.join(self.args.image_folder, "metrics_combined_summary.json")
        if os.path.exists(combined_path):
            try:
                with open(combined_path, "r") as f:
                    metrics = json.load(f)
                print("Loaded existing metrics from", combined_path)
                return metrics
            except Exception:
                # fall through to recompute if file corrupted
                pass

        def load_stack(path):
            if path and os.path.exists(path):
                a = np.load(path)
                # accept (N,H,W) or (N,1,H,W)
                if a.ndim == 4 and a.shape[1] == 1:
                    a = a[:, 0, :, :]
                return a.astype(np.float32)
            return None

        # load stage1 stacks
        stage1_whole = load_stack(os.path.join(stage1_dir, "whole.npy"))
        stage1_sm = load_stack(os.path.join(stage1_dir, f"whole_gaussian_filter1d_sigma_t{sigma_t}.npy"))

        # discover stage2 pre/post files under stage2_dir
        stage2_pre_dict = {}
        stage2_post_dict = {}
        if stage2_dir is not None and os.path.isdir(stage2_dir):
            # find files matching stage2_lvl{idx}_preGF.npy and _postGF.npy, and stage2_final.npy
            pre_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_preGF.npy")))
            post_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_postGF.npy")))
            # map by level index extracted
            for p in pre_files:
                base = os.path.basename(p)
                # expect stage2_lvl{idx}_preGF.npy
                try:
                    idx = int(base.split("stage2_lvl")[1].split("_")[0])
                except Exception:
                    idx = len(stage2_pre_dict)
                stage2_pre_dict[idx] = np.load(p).astype(np.float32)
            for p in post_files:
                base = os.path.basename(p)
                try:
                    idx = int(base.split("stage2_lvl")[1].split("_")[0])
                except Exception:
                    idx = len(stage2_post_dict)
                stage2_post_dict[idx] = np.load(p).astype(np.float32)
            # also allow single final file
            final_p = os.path.join(stage2_dir, "stage2_final.npy")
            if os.path.exists(final_p):
                stage2_post_dict[max(stage2_post_dict.keys())+1 if len(stage2_post_dict)>0 else 0] = np.load(final_p).astype(np.float32)

        # helper to build GT stack from dataset
        def build_gt_stack():
            try:
                _, test_dataset = get_dataset_cascade(self.args, self.config)
                gt_list = []
                for item in test_dataset:
                    gt = item["gt_img"] * 255. # range: [0,255] -> 그대로 stack하면 됨.
                    # gt = None
                    # if isinstance(item, dict) and "gt_img" in item:
                    #     gt = item["gt_img"]
                    # elif isinstance(item, (list, tuple)) and len(item) > 1:
                    #     gt = item[1] if isinstance(item[1], torch.Tensor) else None
                    # if gt is None:
                    #     continue
                    # if isinstance(gt, torch.Tensor):
                    #     g = inverse_data_transform(self.config, gt).cpu().numpy()
                    # else:
                    #     g = gt
                    # if g.ndim == 3:
                    #     g = g[0]
                    # g = g.astype(np.float32)
                    # if np.nanmax(g) <= 1.01:
                    #     g = g * 255.0
                    gt_list.append(gt)
                if len(gt_list) > 0:
                    return np.stack(gt_list, axis=0)
                if len(gt_list) > 0:
                    return np.stack(gt_list, axis=0)
            except Exception as e:
                print("compute_and_save_metrics: failed to build GT stack:", e)
            return None

        gt_stack = build_gt_stack()

        # normalize pred -> 0..255 for PSNR/SSIM computation if needed
        def prepare_pred(pred):
            p = pred.astype(np.float32)
            if np.nanmax(p) <= 1.01:
                p = p * 255.0
            return p

        def compute_metrics(name, pred_stack):
            if gt_stack is None:
                print(f"SKIP metrics {name}: GT not available")
                return None
            pred = prepare_pred(pred_stack)
            N = min(gt_stack.shape[0], pred.shape[0])
            ps = np.full(N, np.nan, dtype=np.float32)
            ss = np.full(N, np.nan, dtype=np.float32)
            for i in range(N):
                gt_i = gt_stack[i].astype(np.float32)
                pred_i = pred[i].astype(np.float32)
                try:
                    ps[i] = psnr_fn(gt_i, pred_i, data_range=255.0)
                except Exception:
                    ps[i] = np.nan
                try:
                    ss[i] = ssim_fn(gt_i, pred_i, data_range=255.0)
                except Exception:
                    ss[i] = np.nan
            summary = {
                "n": int(N),
                "psnr_mean": float(np.nanmean(ps)),
                "psnr_std": float(np.nanstd(ps)),
                "ssim_mean": float(np.nanmean(ss)),
                "ssim_std": float(np.nanstd(ss))
            }
            # save per-frame CSV
            csv_path = os.path.join(self.args.image_folder, f"metrics_{name}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["idx", "psnr", "ssim"])
                for j in range(N):
                    writer.writerow([j, float(ps[j]) if not np.isnan(ps[j]) else "", float(ss[j]) if not np.isnan(ss[j]) else ""])
            # save json summary
            with open(os.path.join(self.args.image_folder, f"summary_{name}.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"{name} -> PSNR {summary['psnr_mean']:.4f} _ {summary['psnr_std']:.4f}, SSIM {summary['ssim_mean']:.4f} _ {summary['ssim_std']:.4f}")
            return summary

        metrics = {}
        if stage1_whole is not None:
            metrics['stage1_whole'] = compute_metrics("stage1_whole", stage1_whole)
        if stage1_sm is not None:
            metrics['stage1_smoothed'] = compute_metrics("stage1_smoothed", stage1_sm)

        # handle stage2 levels in sorted order
        for idx in sorted(set(list(stage2_pre_dict.keys()) + list(stage2_post_dict.keys()))):
            pre = stage2_pre_dict.get(idx, None)
            post = stage2_post_dict.get(idx, None)
            name_pre = f"stage2_sr{idx+1}_pre"
            name_post = f"stage2_sr{idx+1}_post"
            # name as stage2_sr{level}_{pre/post} where level indexes are 1-based for readability
            if pre is not None:
                metrics[name_pre] = compute_metrics(name_pre, pre)
            if post is not None:
                metrics[name_post] = compute_metrics(name_post, post)

        # combined summary
        # with open(os.path.join(self.args.image_folder, "metrics_combined_summary.json"), "w") as f:
        with open(combined_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

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
