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

from datasets import get_dataset_kvswap, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
# from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model_KVswap, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
import matplotlib.pyplot as plt

import torch.nn.functional as F

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

        from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion
        if self.args.use_pas:
            from functions.svd_ddnm_perProj_modified import ddnm_diffusion, ddnm_plus_diffusion

        self.ddnm_diffusion = ddnm_diffusion
        self.ddnm_plus_diffusion = ddnm_plus_diffusion

        # stepInterval, l2thr
        # self.args.l2thr = float(self.args.image_folder.split("l2thr_")[-1].split("/")[0])

    def sample(self, simplified):
        # simplified: True if it is in the argument
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)
            ckpt = "/workspace/DDNM/exp/logs/totensor/ckpt.pth"

            # if self.config.data.dataset == "CIFAR10":
            #     name = "cifar10"
            # elif self.config.data.dataset == "LSUN":
            #     name = f"lsun_{self.config.data.category}"
            # elif self.config.data.dataset == 'CelebA_HQ':
            #     name = 'celeba_hq'
            # else:
            #     raise ValueError
            # if name != 'celeba_hq':
            #     ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
            
            # if self.config.data.dataset == "CIFAR10":
            #     name = "cifar10"
            # elif self.config.data.dataset == "LSUN":
            #     name = f"lsun_{self.config.data.category}"
            # elif self.config.data.dataset == 'CelebA_HQ':
            #     name = 'celeba_hq'
            # else:
            #     raise ValueError
            # if name != 'celeba_hq':
            #     ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
            print("Loading checkpoint {}".format(ckpt))
            
            if 'totensor' in ckpt or 'CheXpert_8' in ckpt: # here
                model.to(self.device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(ckpt, map_location=self.device)[0])
            # elif 'Code test2' in ckpt:
            #     model.to(self.device)
            #     model.load_state_dict(torch.load(ckpt, map_location=self.device)[0])
            #     model = torch.nn.DataParallel(model)
            
        elif self.config.model.type == 'openai': # here 

            config_dict = vars(self.config.model)
            model = create_model_KVswap(**config_dict)
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
            
        # print(result[0].shape, type(result[0]))
        np.save(os.path.join(self.args.image_folder, "whole.npy"), np.array(result))
        return result

            
    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        _, test_dataset = get_dataset_kvswap(args, config)

        # device_count = torch.cuda.device_count()

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

        _, test_dataset = get_dataset_kvswap(args, config)

        # device_count = torch.cuda.device_count()

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
            for j in range(x[0].size(0)):
                # x[0].shape: [3, 512, 512]
                # x[0] to [512, 512]
                x_1ch = x[0][j][0]
                tvu.save_image(
                    x_1ch, os.path.join(self.args.image_folder, f"pred_png/pred_{(idx_so_far + j):02d}.png")
                )
                whole_result.append(x[0][j].detach().cpu().numpy())
                np.save(os.path.join(self.args.image_folder, f"pred_npy/pred_{(idx_so_far + j):02d}.npy"), x_1ch.detach().cpu().numpy())
                orig = inverse_data_transform(config, x_orig[j])
                orig_1ch = orig[0]
                mse = torch.mean((x_1ch.to(self.device) - orig_1ch) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

            # 0TH and 30th image -> save x0_preds into os.path.join(args.image_folder, "x0_preds/0th or /30th")
            # if (idx % 101 == 0 or "param_search" in self.args.image_folder):
            #     os.makedirs(os.path.join(self.args.image_folder, F"x0_preds/{idx:02d}"), exist_ok=True)
            #     os.makedirs(os.path.join(self.args.image_folder, F"x0_preds/{idx:02d}_t"), exist_ok=True)
            #     x0_changes = []  # 변화량을 저장할 리스트
            #     x0t_changes = []
            #     psnr_values = []  # PSNR 값을 저장할 리스트
            #     for i in range(len(x0_preds)):
            #         x0_pred_current = x0_preds[i][0]
            #         x0_pred_current = (x0_pred_current + 1.0) / 2.0
            #         x0_pred_current = torch.clamp(x0_pred_current, 0.0, 1.0)

            #         x0t_pred_current = x0t_preds[i][0]
            #         x0t_pred_current = (x0t_pred_current + 1.0) / 2.0
            #         x0t_pred_current = torch.clamp(x0t_pred_current, 0.0, 1.0)

            #         x0_pred_previous = x0_preds[i-1][0]
            #         x0_pred_previous = (x0_pred_previous + 1.0) / 2.0
            #         x0_pred_previous = torch.clamp(x0_pred_previous, 0.0, 1.0)

            #         x0t_pred_previous = x0t_preds[i-1][0]
            #         x0t_pred_previous = (x0t_pred_previous + 1.0) / 2.0
            #         x0t_pred_previous = torch.clamp(x0t_pred_previous, 0.0, 1.0)

            #         # save x0_preds into os.path.join(args.image_folder, "x0_preds/00 or /20")
            #         tvu.save_image(
            #             x0_pred_current, os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}/{i:02d}.png")
            #         )
            #         # save x0t_preds into os.path.join(args.image_folder, "x0_preds/00_t or /20_t")
            #         tvu.save_image(
            #             x0t_pred_current, os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_t/{i:02d}.png")
            #         )

            #         # 변화량 계산 (L2 norm)
            #         x0_change = torch.norm(x0_pred_current - x0_pred_previous).item()
            #         x0t_change = torch.norm(x0t_pred_current - x0t_pred_previous).item()

            #         x0_changes.append(x0_change)  # 변화량을 리스트에 추가
            #         x0t_changes.append(x0t_change)  # 변화량을 리스트에 추가

            #         # x0t_pred와 orig 이미지 비교하여 PSNR 계산
            #         temp_orig = inverse_data_transform(config, x_orig[0].unsqueeze(0))[0][0]  # orig 이미지 불러오기 및 변환
            #         # device를 cpu로 변경하여 계산
            #         temp_orig = temp_orig.to('cpu')
            #         mse_x0t_orig = torch.mean((x0t_pred_current - temp_orig) ** 2)
            #         psnr_x0t_orig = 10 * torch.log10(1 / mse_x0t_orig) if mse_x0t_orig > 0 else torch.tensor(float('inf'))  # PSNR 계산, mse가 0인 경우 inf로 처리
            #         psnr_values.append(psnr_x0t_orig.item())  # PSNR 값을 리스트에 추가


            #     # 변화량들을 파일에 저장
            #     os.makedirs(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes"), exist_ok=True)
                
            #     with open(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0_changes.txt"), "w") as f:
            #         for change in x0_changes:
            #             f.write(f"{change}\n")
            #     with open(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/x0t_changes.txt"), "w") as f:
            #         for change in x0t_changes:
            #             f.write(f"{change}\n")

            #     # PSNR 값들을 파일에 저장
            #     with open(os.path.join(self.args.image_folder, f"x0_preds/{idx:02d}_changes/psnr_x0t_orig.txt"), "w") as f:
            #         for psnr in psnr_values:
            #             f.write(f"{psnr}\n")

            #     # x0_pred 변화량 그래프
            #     plt.figure(figsize=(10, 5))
            #     plt.plot(x0_changes)
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
