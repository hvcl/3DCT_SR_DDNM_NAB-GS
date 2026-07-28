import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np

class_num = 951

def downsample_tensor(tensor_list, scale_factor=0.5):
    """
    텐서 리스트를 받아 각 텐서의 H, W를 scale_factor만큼 downsample하는 함수.
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

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


# ==================== MPES: Multi-Path Ensemble Sampling ====================
def mpes_sample(x, model, b, eta, A_funcs, y, temp_y, args, config, single_sample_fn):
    """
    Multi-Path Ensemble Sampling (MPES)
    Generate K independent trajectories with different seeds, then aggregate.
    
    Args:
        x: initial noise
        model: diffusion model
        b: beta schedule
        eta: DDIM eta parameter
        A_funcs: degradation operators
        y: degraded observation
        temp_y: GT LR projection for reference
        args: arguments containing mpes_k and mpes_mode
        config: config namespace
        single_sample_fn: function that runs one sampling trajectory
    
    Returns:
        aggregated result (x_final, x0_preds, x0t_preds, start_step)
    """
    original_seed = args.seed
    results = []
    
    print(f"[MPES] Running {args.mpes_k} ensemble trajectories...")
    
    for k in range(args.mpes_k):
        # Set different seed for each trajectory
        current_seed = original_seed + 1000 * k
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        
        print(f"[MPES] Trajectory {k+1}/{args.mpes_k} with seed {current_seed}")
        
        # Run single sampling trajectory
        x_k, x0_preds_k, x0t_preds_k, start_step_k = single_sample_fn(
            x.clone(), model, b, eta, A_funcs, y, temp_y, args, config
        )
        
        results.append({
            'x': x_k,
            'x0_preds': x0_preds_k,
            'x0t_preds': x0t_preds_k,
            'start_step': start_step_k
        })
    
    # Aggregate results based on mode
    if args.mpes_mode == "mean":
        print("[MPES] Aggregating via MEAN")
        x_final = [torch.stack([r['x'][i] for r in results]).mean(dim=0) for i in range(len(results[0]['x']))]
        x0_preds_final = [torch.stack([r['x0_preds'][i] for r in results]).mean(dim=0) for i in range(len(results[0]['x0_preds']))]
        x0t_preds_final = [torch.stack([r['x0t_preds'][i] for r in results]).mean(dim=0) for i in range(len(results[0]['x0t_preds']))]
        start_step_final = int(np.mean([r['start_step'] for r in results]))
        
    elif args.mpes_mode == "median":
        print("[MPES] Aggregating via MEDIAN")
        x_final = [torch.stack([r['x'][i] for r in results]).median(dim=0)[0] for i in range(len(results[0]['x']))]
        x0_preds_final = [torch.stack([r['x0_preds'][i] for r in results]).median(dim=0)[0] for i in range(len(results[0]['x0_preds']))]
        x0t_preds_final = [torch.stack([r['x0t_preds'][i] for r in results]).median(dim=0)[0] for i in range(len(results[0]['x0t_preds']))]
        start_step_final = int(np.median([r['start_step'] for r in results]))
        
    elif args.mpes_mode == "best_psnr":
        print("[MPES] Selecting BEST trajectory by PSNR")
        # Simple heuristic: choose trajectory with lowest final reconstruction error
        best_idx = 0
        best_error = float('inf')
        for idx, r in enumerate(results):
            error = torch.norm(r['x'][0] - temp_y).item()
            if error < best_error:
                best_error = error
                best_idx = idx
        
        print(f"[MPES] Selected trajectory {best_idx+1} (error: {best_error:.4f})")
        x_final = results[best_idx]['x']
        x0_preds_final = results[best_idx]['x0_preds']
        x0t_preds_final = results[best_idx]['x0t_preds']
        start_step_final = results[best_idx]['start_step']
    
    else:
        raise ValueError(f"Unknown mpes_mode: {args.mpes_mode}")
    
    # Restore original seed
    torch.manual_seed(original_seed)
    np.random.seed(original_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(original_seed)
    
    return x_final, x0_preds_final, x0t_preds_final, start_step_final


# ==================== Data Consistency Guidance ====================
def apply_dc_guidance(x0_t, y, A_funcs, args, t_idx, total_steps):
    """
    Apply Data Consistency Guidance
    
    Args:
        x0_t: predicted x0 at current timestep
        y: GT LR projection (degraded observation)
        A_funcs: degradation operators
        args: arguments containing dc_lambda, dc_start_step, dc_end_step, dc_annealing
        t_idx: current timestep index
        total_steps: total number of sampling steps
    
    Returns:
        x0_t_corrected: x0 with DC guidance applied
    """
    # Check if we should apply DC guidance at this step
    if t_idx < args.dc_start_step or t_idx > args.dc_end_step:
        return x0_t
    
    # Lambda annealing: stronger guidance at later steps (smaller t)
    if args.dc_annealing:
        # Anneal from 0.5*lambda to 1.5*lambda as t decreases
        progress = (total_steps - t_idx) / total_steps  # 0 at start, 1 at end
        lambda_t = args.dc_lambda * (0.5 + progress)
    else:
        lambda_t = args.dc_lambda
    
    # Compute data consistency error
    b, c, h, w = x0_t.size()
    hwc = c * h * w
    
    # Forward projection: A(x0_t)
    x0_t_flat = x0_t.reshape((b, hwc))
    Ax0 = A_funcs.A(x0_t_flat)  # shape: [b, hwc_lr]
    
    # Consistency error: A(x0) - y
    dc_error = Ax0 - y.reshape(y.size(0), -1)
    
    # Backproject error: A^†(A(x0) - y)
    dc_correction = A_funcs.A_pinv(dc_error)
    dc_correction = dc_correction.reshape(*x0_t.size())
    
    # Apply guidance: x0_corrected = x0 - lambda * A^†(A(x0) - y)
    x0_t_corrected = x0_t - lambda_t * dc_correction
    
    return x0_t_corrected


# ==================== Main Diffusion Functions ====================
def ddnm_diffusion(x, model, b, eta, A_funcs, y, temp_y=None, cls_fn=None, classes=None, config=None, args=None):
    """
    DDNM Diffusion with optional MPES and DC Guidance
    """
    # If MPES is enabled, delegate to ensemble sampler
    if args.mpes:
        return mpes_sample(
            x, model, b, eta, A_funcs, y, temp_y, args, config,
            single_sample_fn=ddnm_diffusion_single
        )
    else:
        return ddnm_diffusion_single(x, model, b, eta, A_funcs, y, temp_y, args, config)


def ddnm_diffusion_single(x, model, b, eta, A_funcs, y, temp_y, args, config, cls_fn=None, classes=None):
    """
    Single trajectory DDNM diffusion (with optional DC guidance)
    """
    with torch.no_grad():
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        x0t_preds = []
        xs = [x]

        # generate time schedule
        times_select = get_schedule_jump(config.time_travel.T_sampling, 1, 1)
        time_pairs_select = list(zip(times_select[:-1], times_select[1:]))
        start_step = select_start_step(args, b, x, y, temp_y, model, time_pairs_select, skip, eta, A_funcs, sigma_y=0)

        times = get_schedule_jump(50 - start_step,
                                   config.time_travel.travel_length,
                                   config.time_travel.travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))
        total_steps = len(time_pairs)

        # reverse diffusion sampling
        for t_idx, (i, j) in tqdm(enumerate(time_pairs), desc="Diffusion sampling"):
            i, j = i*skip, j*skip
            if t_idx == 0:
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                xt = xs[-1] * (1 - at).sqrt() + at.sqrt() * temp_y
                xs.append(xt.to('cpu'))

            if j<0: j=-1

            if j < i:  # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')

                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # ===== NEW: Apply DC Guidance if enabled =====
                if args.dc_guidance:
                    x0_t = apply_dc_guidance(x0_t, y, A_funcs, args, t_idx, total_steps)

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                x0t_preds.append(x0_t_hat.to('cpu'))
                xs.append(xt_next.to('cpu'))

        # UHRCT 조건 확인 후 다운샘플링
        if "UHRCT" in args.config:
            xs_last_downsampled = downsample_tensor([xs[-1]])
            x0_preds_downsampled = downsample_tensor(x0_preds)
            x0t_preds_downsampled = downsample_tensor(x0t_preds)
            return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
        else:
            return [xs[-1]], x0_preds, x0t_preds, start_step


def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, temp_y=None, cls_fn=None, classes=None, config=None, args=None):
    """
    DDNM+ Diffusion with optional MPES and DC Guidance
    """
    # If MPES is enabled, delegate to ensemble sampler
    if args.mpes:
        return mpes_sample(
            x, model, b, eta, A_funcs, y, temp_y, args, config,
            single_sample_fn=lambda x, model, b, eta, A_funcs, y, temp_y, args, config: 
                ddnm_plus_diffusion_single(x, model, b, eta, A_funcs, y, sigma_y, temp_y, args, config)
        )
    else:
        return ddnm_plus_diffusion_single(x, model, b, eta, A_funcs, y, sigma_y, temp_y, args, config)


def ddnm_plus_diffusion_single(x, model, b, eta, A_funcs, y, sigma_y, temp_y, args, config, cls_fn=None, classes=None):
    """
    Single trajectory DDNM+ diffusion (with optional DC guidance)
    """
    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        x0t_preds = []
        xs = [x]

        times = get_schedule_jump(config.time_travel.T_sampling,
                                   config.time_travel.travel_length,
                                   config.time_travel.travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))
        start_step = select_start_step(args, b, x, y, temp_y, model, time_pairs, skip, eta, A_funcs, sigma_y)
        time_pairs = time_pairs[start_step:]
        total_steps = len(time_pairs)

        for t_idx, (i, j) in tqdm(enumerate(time_pairs), desc="Diffusion+ sampling"):
            i, j = i*skip, j*skip
            if t_idx == 0:
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                xt = xs[-1] * (1 - at).sqrt() + at.sqrt() * temp_y
                xs.append(xt.to('cpu'))

            if j<0: j=-1

            if j < i:
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')

                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # ===== NEW: Apply DC Guidance if enabled =====
                if args.dc_guidance:
                    x0_t = apply_dc_guidance(x0_t, y, A_funcs, args, t_idx, total_steps)

                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1),
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t.to('cpu'))
                x0t_preds.append(x0_t_hat.to('cpu'))
                xs.append(xt_next.to('cpu'))

        if "UHRCT" in args.config:
            xs_last_downsampled = downsample_tensor([xs[-1]])
            x0_preds_downsampled = downsample_tensor(x0_preds)
            x0t_preds_downsampled = downsample_tensor(x0t_preds)
            return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
        else:
            return [xs[-1]], x0_preds, x0t_preds, start_step


def select_start_step(args, b, x, y, temp_y, model, time_pairs, skip, eta, A_funcs, sigma_y):
    """
    Select start step based on L2 threshold
    """
    n = x.size(0)
    l2thr = args.l2thr
    x0_preds = []
    x0t_preds = []

    for t_idx, (i, j) in tqdm(enumerate(time_pairs), desc="Select start step"):
        i, j = i*skip, j*skip
        if j<0: j=-1

        t = (torch.ones(n) * i).to(x.device).long()
        at = compute_alpha(b, t.long())
        xt = x * (1 - at).sqrt() + at.sqrt() * temp_y

        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())

        et = model(xt, t)
        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        if sigma_y==0:
            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())
        else:
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

        x0_preds.append(x0_t.to('cpu'))
        x0t_preds.append(x0_t_hat.to('cpu'))

    # L2 norm change calculation
    temp_y = temp_y.to('cpu')
    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)

    x0t_changes_l2 = []
    for i, (x0_pred, x0t_pred) in enumerate(zip(x0_preds, x0t_preds)):
        x0t_pred_01 = (x0t_pred + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        x0t_change_l2 = torch.norm(temp_y_01 - x0t_pred_01, p=2)
        x0t_changes_l2.append(x0t_change_l2)

    # Set start step when L2 norm change is less than l2thr
    start_step = 0
    for i, x0t_change in enumerate(x0t_changes_l2):
        if x0t_change < l2thr:
            start_step = i
            break

    return start_step


# Form RePaint
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
