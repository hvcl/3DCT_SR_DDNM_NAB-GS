from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
import torchvision.utils as tvu

class_num = 951


def _build_radial_low_mask(h, w, cutoff, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (rr <= float(cutoff)).to(dtype)
    return mask[None, None, :, :]


def _fft_split_low_high(x, cutoff):
    h, w = x.shape[-2:]
    low_mask = _build_radial_low_mask(h, w, cutoff, x.device, x.dtype)
    fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    low_fft = fft * low_mask
    high_fft = fft * (1.0 - low_mask)
    low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
    high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return low, high


def apply_late_highfreq_controls(x0_t_hat, prev_projection_output, args, t_idx, total_steps):
    late_steps_k = int(getattr(args, "late_steps_k", 0))
    if late_steps_k <= 0 or total_steps <= 0:
        return x0_t_hat
    if t_idx < max(0, total_steps - late_steps_k):
        return x0_t_hat

    cutoff = float(getattr(args, "late_highpass_cutoff", 45.0))
    damp = float(getattr(args, "late_highpass_damp", 1.0))
    prev_blend = float(getattr(args, "late_prev_blend", 0.0))

    current_low, current_high = _fft_split_low_high(x0_t_hat, cutoff)
    new_high = current_high

    if damp < 1.0:
        new_high = new_high * max(damp, 0.0)

    if prev_blend > 0.0 and prev_projection_output is not None:
        prev = prev_projection_output.to(x0_t_hat.device)
        if prev.dim() == 3:
            prev = prev.unsqueeze(0)
        if prev.shape[0] != x0_t_hat.shape[0]:
            prev = prev[:1].expand(x0_t_hat.shape[0], -1, -1, -1)
        _, prev_high = _fft_split_low_high(prev, cutoff)
        blend = min(max(prev_blend, 0.0), 1.0)
        new_high = (1.0 - blend) * new_high + blend * prev_high

    return current_low + new_high

def downsample_tensor(tensor_list, scale_factor=0.5):
    """
    Downsample tensor list by scale_factor.
    """
    downsampled_list = []
    for tensor in tensor_list:
        if tensor is not None:
            downsampled_list.append(F.interpolate(tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False))
        else:
            downsampled_list.append(None)
    return downsampled_list

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def compute_sigma_y_t(args, beta, t, sigma_y_base):
    schedule = getattr(args, "sigma_y_schedule", "fixed")
    if schedule == "fixed":
        return float(sigma_y_base)
    if schedule == "two_phase":
        sigma_early = float(getattr(args, "sigma_y_early", -1.0))
        sigma_late = float(getattr(args, "sigma_y_late", -1.0))
        if sigma_early < 0:
            sigma_early = float(sigma_y_base)
        if sigma_late < 0:
            sigma_late = float(sigma_y_base)
        switch_frac = float(getattr(args, "sigma_y_switch_fraction", 0.5))
        step_idx = int(t[0].item()) if torch.is_tensor(t) else int(t)
        total_steps = int(beta.shape[0])
        switch_step = int(total_steps * switch_frac)
        return float(sigma_early if step_idx >= switch_step else sigma_late)

    sigma_min = getattr(args, "sigma_y_min", -1.0)
    sigma_max = getattr(args, "sigma_y_max", -1.0)
    gamma = float(getattr(args, "sigma_y_gamma", 1.0))

    if sigma_min < 0:
        sigma_min = float(sigma_y_base)
    if sigma_max < 0:
        sigma_max = float(sigma_y_base)

    if schedule == "alphabar":
        at = compute_alpha(beta, t.long())
        noise_level = torch.sqrt(torch.clamp(1.0 - at, min=0.0))
        weight = float(noise_level[0, 0, 0, 0].item()) ** gamma
        return float(sigma_min + (sigma_max - sigma_min) * weight)

    if schedule == "beta":
        beta_full = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
        beta_t = float(beta_full.index_select(0, t.long() + 1)[0].item())
        beta_norm = max(0.0, min(1.0, beta_t / float(beta.max().item())))
        weight = beta_norm ** gamma
        return float(sigma_min + (sigma_max - sigma_min) * weight)

    return float(sigma_y_base)

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def _should_use_temp_y_for_step(args, step_idx):
    if getattr(args, "deg", "") in ["fourier", "fourier_deblur_aniso"]:
        return True
    if bool(getattr(args, "always_noised_ddnm_first", False)):
        return True
    late_start = int(getattr(args, "noised_ddnm_first_start_step", -1))
    return late_start >= 0 and step_idx >= late_start


def _compose_xt_for_denoiser(xs_last, at, temp_y, use_temp_y_this_step):
    if use_temp_y_this_step and temp_y is not None:
        return xs_last * (1 - at).sqrt() + at.sqrt() * temp_y
    return xs_last


def compute_eta_t(args, step_idx, total_steps):
    schedule = getattr(args, "eta_schedule", "fixed")
    base_eta = float(getattr(args, "eta", 0.85))
    if schedule == "fixed":
        return base_eta
    if schedule == "two_phase":
        eta_early = float(getattr(args, "eta_early", -1.0))
        eta_late = float(getattr(args, "eta_late", -1.0))
        if eta_early < 0:
            eta_early = base_eta
        if eta_late < 0:
            eta_late = base_eta
        switch_frac = float(getattr(args, "eta_switch_fraction", 0.5))
        switch_step = int(total_steps * switch_frac)
        return eta_early if step_idx < switch_step else eta_late
    return base_eta

def save_formula_tensor(base_dir, name, tensor, step_idx=None):
    """
    Save tensor as both NPY and PNG for formula-visualization.
    tensor: [B,C,H,W] or [C,H,W]
    """
    os.makedirs(base_dir, exist_ok=True)

    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor_1ch = tensor[0]
    else:
        tensor_1ch = tensor

    if step_idx is None:
        stem = name
    else:
        stem = f"step_{step_idx:03d}_{name}"

    tensor_01 = inverse_data_transform(tensor_1ch.detach().cpu())
    np.save(os.path.join(base_dir, f"{stem}.npy"), tensor_01.numpy())
    tvu.save_image(tensor_01, os.path.join(base_dir, f"{stem}.png"))

    # For weak residual-like maps, also save contrast-enhanced visualization.
    if name == "I_minus_AdagA_x0_given_t":
        arr = tensor_1ch.detach().cpu().numpy().astype(np.float32)
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        if hi <= lo:
            vis = np.zeros_like(arr, dtype=np.float32)
        else:
            vis = (arr - lo) / (hi - lo)
            vis = np.clip(vis, 0.0, 1.0)

        vis_t = torch.from_numpy(vis)
        np.save(os.path.join(base_dir, f"{stem}_contrast.npy"), vis.astype(np.float32))
        tvu.save_image(vis_t, os.path.join(base_dir, f"{stem}_contrast.png"))


def save_step_prediction_batch(base_dir, name, tensor, idx_so_far, step_idx, t_cur, t_next):
    os.makedirs(base_dir, exist_ok=True)
    batch = tensor.detach().cpu()
    if batch.dim() != 4:
        return

    schedule_path = os.path.join(base_dir, "schedule.csv")
    if not os.path.exists(schedule_path):
        with open(schedule_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step_idx", "t_cur", "t_next"])

    with open(schedule_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([int(step_idx), int(t_cur), int(t_next)])

    name_dir = os.path.join(base_dir, name)
    os.makedirs(name_dir, exist_ok=True)

    for batch_i in range(batch.shape[0]):
        proj_idx = idx_so_far + batch_i
        proj_dir = os.path.join(name_dir, f"proj_{proj_idx:02d}")
        os.makedirs(proj_dir, exist_ok=True)
        tensor_01 = inverse_data_transform(batch[batch_i, 0])
        np.save(os.path.join(proj_dir, f"step_{step_idx:03d}.npy"), tensor_01.numpy().astype(np.float32))
        tvu.save_image(tensor_01, os.path.join(proj_dir, f"step_{step_idx:03d}.png"))

def ddnm_diffusion(x, model, b, eta, A_funcs, y, temp_y_original=None, temp_y_cgls=None, cls_fn=None, classes=None, config=None, args=None, idx_so_far=0, prev_projection_output=None):
    with torch.no_grad():
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        x0t_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                                config.time_travel.travel_length, 
                                config.time_travel.travel_repeat,
                                )

        time_pairs = list(zip(times[:-1], times[1:]))
        
        if args.setup == "ddnm_orig":
            start_step = 0
        elif args.setup == "ddnm_fixedSteps":
            start_step = args.startStep
        elif args.setup == "ddnm_pas":
            # Use temp_y_original (upsampled LR) for start step selection
            start_step = select_start_step(args, b, x, y, temp_y_original, model, time_pairs, skip, eta, A_funcs, sigma_y=0, idx_so_far=idx_so_far)
        
        if args.minimum_PAS_startStep >=0 and start_step > args.minimum_PAS_startStep:
            time_pairs = time_pairs[args.minimum_PAS_startStep:]
        else:
            time_pairs = time_pairs[start_step:]

        # Use temp_y_cgls for initialization if available, otherwise use temp_y_original
        temp_y_for_init = temp_y_cgls if temp_y_cgls is not None else temp_y_original

        save_formula = getattr(args, "save_ddnm_formula", False) and idx_so_far == getattr(args, "formula_proj_idx", 0)
        formula_dir = os.path.join(args.image_folder, "ddnm_formula", f"proj_{idx_so_far:02d}")

        total_steps = len(time_pairs)
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            i, j = i*skip, j*skip
            
            # Always initialize from temp_y if cgls_path is provided or for PAS/fixedSteps modes
            if t_idx == 0 and (args.setup in ["ddnm_fixedSteps", "ddnm_pas"] or args.cgls_path is not None):
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                # xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
                xt = xs[-1] * (1 - at).sqrt() + at.sqrt() * temp_y_for_init
                xs.append(xt.to('cpu'))

            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt_base = xs[-1].to(x.device)
                use_temp_y_this_step = _should_use_temp_y_for_step(args, t_idx)
                xt = _compose_xt_for_denoiser(xt_base, at, temp_y_for_init, use_temp_y_this_step)
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=x.device) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() # shape: (1, 3, 128, 128)
                eta_t = compute_eta_t(args, t_idx, total_steps)
                skip_measure = args.ddnm_step_before is not None and args.ddnm_step_before >= 0 and t_idx >= args.ddnm_step_before
                a_dag_a_x0_t = A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1))
                ).reshape(*x0_t.size())
                i_minus_a_dag_a_x0_t = x0_t - a_dag_a_x0_t
                if skip_measure:
                    x0_t_hat = x0_t
                else:
                    x0_t_hat = x0_t - A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(*x0_t.size())

                x0_t_hat = apply_late_highfreq_controls(x0_t_hat, prev_projection_output, args, t_idx, total_steps)

                c1 = (1 - at_next).sqrt() * eta_t
                c2 = (1 - at_next).sqrt() * ((1 - eta_t ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                if save_formula:
                    save_formula_tensor(formula_dir, "x_t", xt, t_idx)
                    save_formula_tensor(formula_dir, "x0_given_t", x0_t, t_idx)
                    save_formula_tensor(formula_dir, "I_minus_AdagA_x0_given_t", i_minus_a_dag_a_x0_t, t_idx)
                    save_formula_tensor(formula_dir, "x0_hat_given_t", x0_t_hat, t_idx)
                    save_formula_tensor(formula_dir, "x_t_minus_1", xt_next, t_idx)
                if getattr(args, "save_step_outputs", False):
                    stride = max(1, int(getattr(args, "step_output_stride", 1)))
                    if (t_idx % stride) == 0:
                        step_dir = os.path.join(args.image_folder, "step_analysis")
                        save_step_prediction_batch(step_dir, "x0_hat", x0_t_hat, idx_so_far, t_idx, i, j)
                        save_step_prediction_batch(step_dir, "x0", x0_t, idx_so_far, t_idx, i, j)

                x0_preds.append(x0_t.to('cpu'))
                x0t_preds.append(x0_t_hat.to('cpu'))
                xs.append(xt_next.to('cpu'))

    # UHRCT 조건 확인 후 다운샘플링
    if "UHRCT" in args.config and "down2x" in args.image_folder:
        xs_last_downsampled = downsample_tensor([xs[-1]])
        x0_preds_downsampled = downsample_tensor(x0_preds)
        x0t_preds_downsampled = downsample_tensor(x0t_preds)
        return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
    else:
        return [xs[-1]], x0_preds, x0t_preds, start_step

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, temp_y_original=None, temp_y_cgls=None, cls_fn=None, classes=None, config=None, args=None, idx_so_far=0, prev_projection_output=None):
    with torch.no_grad():
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling # 20
        n = x.size(0)
        x0_preds = []
        x0t_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                                config.time_travel.travel_length, 
                                config.time_travel.travel_repeat,
                                )

        time_pairs = list(zip(times[:-1], times[1:]))
        
        if args.setup == "ddnm_orig":
            start_step = 0
        elif args.setup == "ddnm_fixedSteps":
            start_step = args.startStep
        elif args.setup == "ddnm_pas":
            # Use temp_y_original (upsampled LR) for start step selection
            start_step = select_start_step(args, b, x, y, temp_y_original, model, time_pairs, skip, eta, A_funcs, sigma_y, idx_so_far=idx_so_far)
        
        if args.minimum_PAS_startStep >=0 and start_step > args.minimum_PAS_startStep:
            time_pairs = time_pairs[args.minimum_PAS_startStep:]
        else:
            time_pairs = time_pairs[start_step:]

        # Use temp_y_cgls for initialization if available, otherwise use temp_y_original
        temp_y_for_init = temp_y_cgls if temp_y_cgls is not None else temp_y_original

        # reverse diffusion sampling
        save_formula = getattr(args, "save_ddnm_formula", False) and idx_so_far == getattr(args, "formula_proj_idx", 0)
        formula_dir = os.path.join(args.image_folder, "ddnm_formula", f"proj_{idx_so_far:02d}")

        total_steps = len(time_pairs)
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            i, j = i*skip, j*skip

            # Always initialize from temp_y if cgls_path is provided or for PAS/fixedSteps modes
            if t_idx == 0 and (args.setup in ["ddnm_fixedSteps", "ddnm_pas"] or args.cgls_path is not None):
                # temp_y: reshape y into xs[-1] shape
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                # xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
                xt = xs[-1] * (1 - at).sqrt() + at.sqrt() * temp_y_for_init
                xs.append(xt.to('cpu'))

            if j<0: j=-1 

            if j < i: # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                
                xt_base = xs[-1].to(x.device)
                use_temp_y_this_step = _should_use_temp_y_for_step(args, t_idx)
                xt = _compose_xt_for_denoiser(xt_base, at, temp_y_for_init, use_temp_y_this_step)

                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=x.device) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                a_dag_a_x0_t = A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1))
                ).reshape(*x0_t.size())
                i_minus_a_dag_a_x0_t = x0_t - a_dag_a_x0_t

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
                sigma_y_t = compute_sigma_y_t(args, b, t, sigma_y)
                eta_t = compute_eta_t(args, t_idx, total_steps)

                skip_measure = args.ddnm_step_before is not None and args.ddnm_step_before >= 0 and t_idx >= args.ddnm_step_before
                if skip_measure:
                    x0_t_hat = x0_t
                else:
                    # Eq. 17
                    x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y_t, sigma_t, eta_t).reshape(*x0_t.size())

                x0_t_hat = apply_late_highfreq_controls(x0_t_hat, prev_projection_output, args, t_idx, total_steps)

                # Eq. 51
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y_t, sigma_t, eta_t, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                if save_formula:
                    save_formula_tensor(formula_dir, "x_t", xt, t_idx)
                    save_formula_tensor(formula_dir, "x0_given_t", x0_t, t_idx)
                    save_formula_tensor(formula_dir, "I_minus_AdagA_x0_given_t", i_minus_a_dag_a_x0_t, t_idx)
                    save_formula_tensor(formula_dir, "x0_hat_given_t", x0_t_hat, t_idx)
                    save_formula_tensor(formula_dir, "x_t_minus_1", xt_next, t_idx)
                if getattr(args, "save_step_outputs", False):
                    stride = max(1, int(getattr(args, "step_output_stride", 1)))
                    if (t_idx % stride) == 0:
                        step_dir = os.path.join(args.image_folder, "step_analysis")
                        save_step_prediction_batch(step_dir, "x0_hat", x0_t_hat, idx_so_far, t_idx, i, j)
                        save_step_prediction_batch(step_dir, "x0", x0_t, idx_so_far, t_idx, i, j)

                x0_preds.append(x0_t.to('cpu'))
                x0t_preds.append(x0_t_hat.to('cpu'))
                xs.append(xt_next.to('cpu'))

    # UHRCT 조건 확인 후 다운샘플링
    if "UHRCT" in args.config and "down2x" in args.image_folder:
        xs_last_downsampled = downsample_tensor([xs[-1]])
        x0_preds_downsampled = downsample_tensor(x0_preds)
        x0t_preds_downsampled = downsample_tensor(x0t_preds)
        return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
    else:
        return [xs[-1]], x0_preds, x0t_preds, start_step

def select_start_step(args, b, x, y, temp_y, model, time_pairs, skip, eta, A_funcs, sigma_y, idx_so_far=0):
    n = x.size(0)
    
    # Check if adaptive threshold is enabled
    if hasattr(args, 'l2thr_min') and hasattr(args, 'l2thr_max') and \
       args.l2thr_min >= 0 and args.l2thr_max >= 0:
        # Import the adaptive threshold function from diffusion module
        from guided_diffusion.diffusion_miccai26 import get_adaptive_threshold, get_linear_adaptive_threshold
        l2thr = get_linear_adaptive_threshold(idx_so_far, args.l2thr_min, args.l2thr_max)
        # l2thr = get_adaptive_threshold(idx_so_far, args.l2thr_min, args.l2thr_max)
    else:
        l2thr = args.l2thr
    
    x0_preds = []
    x0t_preds = []
    l2_values = []  # Store L2 values for plotting
    
    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)

    for t_idx, (i, j) in tqdm(enumerate(time_pairs), desc="Select start step"):
        i, j = i*skip, j*skip

        if j<0: j=-1

        t = (torch.ones(n) * i).to(x.device).long()
        at = compute_alpha(b, t.long())
        # xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
        xt = x * (1 - at).sqrt() + at.sqrt() * temp_y
        
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
                        
        et = model(xt, t)

        if et.size(1) == 6:
            et = et[:, :3]
        
        # Eq. 12
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Eq. 17
        if sigma_y==0:
            x0_t_hat = x0_t - A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(*x0_t.size())
        else:
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

        # x0_preds.append(x0_t.to('cpu'))
        # x0t_preds.append(x0_t_hat.to('cpu'))

        x0t_pred_01 = (x0_t_hat + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        x0t_change_l2 = torch.norm(temp_y_01 - x0t_pred_01, p=2)
        l2_values.append(x0t_change_l2.item())
        
        if x0t_change_l2.item() < l2thr:
            start_idx = t_idx
            break
    else:
        start_idx = len(time_pairs) - 1
    
    # Save L2 monitoring plot and CSV if enabled
    if args.l2_monitor:
        os.makedirs(os.path.join(args.image_folder, "x0t_change_l2"), exist_ok=True)
        
        # Save CSV with projection index, selected step, and all l2 values
        csv_path = os.path.join(args.image_folder, "x0t_change_l2", "l2_values.csv")
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Write header
                header = ['projection_idx', 'selected_step'] + [f'step_{i}' for i in range(len(l2_values))]
                writer.writerow(header)
            # Write data
            row = [idx_so_far, start_idx] + l2_values
            writer.writerow(row)
        
        # Fixed plot dimensions: x-axis 0-50, y-axis auto but consistent figure size
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(l2_values)), l2_values, 'b-', linewidth=2, label='L2 Distance')
        plt.axhline(y=l2thr, color='r', linestyle='--', linewidth=2, label=f'Threshold ({l2thr})')
        plt.axvline(x=start_idx, color='g', linestyle='--', linewidth=2, label=f'Start Step ({start_idx})')
        plt.xlabel('Time Step Index')
        plt.ylabel('L2 Distance')
        plt.title(f'L2 Distance vs Time Step (Projection {idx_so_far:02d})')
        plt.xlim(0, 50)  # Fixed x-axis range
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(args.image_folder, "x0t_change_l2", f"proj_{idx_so_far:02d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    return start_idx

# form RePaint
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
