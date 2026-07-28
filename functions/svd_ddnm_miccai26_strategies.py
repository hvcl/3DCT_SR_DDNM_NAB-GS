from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

class_num = 951

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

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, model, b, eta, A_funcs, y, temp_y_original=None, temp_y_cgls=None, cls_fn=None, classes=None, config=None, args=None, idx_so_far=0):
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

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() # shape: (1, 3, 128, 128)
                skip_measure = args.ddnm_step_before is not None and args.ddnm_step_before >= 0 and t_idx >= args.ddnm_step_before
                if skip_measure:
                    x0_t_hat = x0_t
                else:
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
    if "UHRCT" in args.config and "down2x" in args.image_folder:
        xs_last_downsampled = downsample_tensor([xs[-1]])
        x0_preds_downsampled = downsample_tensor(x0_preds)
        x0t_preds_downsampled = downsample_tensor(x0t_preds)
        return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
    else:
        return [xs[-1]], x0_preds, x0t_preds, start_step

# 기존 ddnm_plus_diffusion 함수 수정 부분 (간단한 변경)

from adaptive_pas_strategies import select_start_step_unified, get_projection_angle

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, temp_y_original=None, 
                        temp_y_cgls=None, cls_fn=None, classes=None, config=None, args=None, idx_so_far=0):
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
            # ========== 수정 부분: 새로운 adaptive PAS 전략 지원 ==========
            if hasattr(args, 'pas_strategy') and args.pas_strategy in ['angle_aware', 'ssim_based', 'variance_tracking']:
                # 새로운 adaptive strategies 사용
                start_step = select_start_step_unified(args, b, x, y, temp_y_original, model, 
                                                       time_pairs, skip, eta, A_funcs, sigma_y, 
                                                       idx_so_far=idx_so_far, total_projs=100)
            else:
                # 기존 L2-based PAS 사용 (backward compatible)
                from adaptive_pas_strategies import select_start_step_original
                start_step = select_start_step_original(args, b, x, y, temp_y_original, model, 
                                                        time_pairs, skip, eta, A_funcs, sigma_y, 
                                                        idx_so_far=idx_so_far)
        
        if args.minimum_PAS_startStep >=0 and start_step > args.minimum_PAS_startStep:
            time_pairs = time_pairs[args.minimum_PAS_startStep:]
        else:
            time_pairs = time_pairs[start_step:]

        # Use temp_y_cgls for initialization if available, otherwise use temp_y_original
        temp_y_for_init = temp_y_cgls if temp_y_cgls is not None else temp_y_original

        # reverse diffusion sampling
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            i, j = i*skip, j*skip

            # Always initialize from temp_y if cgls_path is provided or for PAS/fixedSteps modes
            if t_idx == 0 and (args.setup in ["ddnm_fixedSteps", "ddnm_pas"] or args.cgls_path is not None):
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                xt = xs[-1] * (1 - at).sqrt() + at.sqrt() * temp_y_for_init
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
                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                skip_measure = args.ddnm_step_before is not None and args.ddnm_step_before >= 0 and t_idx >= args.ddnm_step_before
                if skip_measure:
                    x0_t_hat = x0_t
                else:
                    x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

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
