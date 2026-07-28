import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity

########################################
# 공통 유틸
########################################

def compute_alpha(betas, t):
    """
    betas: (T,) 1D tensor
    t: (N,) long tensor
    return: (N, 1, 1, 1) alpha_bar_t
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
    alpha_t = alphas_cumprod[t]                    # (N,)
    return alpha_t.view(-1, 1, 1, 1)


def get_projection_angle(proj_idx, total_projs=100, angle_range=180):
    """
    proj_idx: 0~(total_projs-1)
    return angle in degrees (0~angle_range)
    """
    return (proj_idx / total_projs) * angle_range


def is_frontal_view(angle, frontal_range=30):
    """
    Frontal: 0 ± frontal_range, 180 ± frontal_range, 90 ± frontal_range
    [0, frontal_range] U [180-frontal_range, 180] U [90-frontal_range, 90+frontal_range]
    """
    a = angle % 180.0
    return (a < frontal_range) or (a > 180.0 - frontal_range)


def is_lateral_view(angle, lateral_range=30):
    """
    Lateral: 90±lateral_range
    """
    a = angle % 180.0
    return (90.0 - lateral_range) < a < (90.0 + lateral_range)


########################################
# 1) 기존 L2 기반 PAS (원본)
########################################

def select_start_step_original(args, b, x, y, temp_y, model,
                               time_pairs, skip, eta, A_funcs,
                               sigma_y, idx_so_far=0):
    n = x.size(0)
    l2thr = args.l2thr

    l2_values = []

    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)

    for t_idx, (i, j) in tqdm(enumerate(time_pairs), desc="Select start step"):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        t = (torch.ones(n) * i).to(x.device).long()
        at = compute_alpha(b, t)

        xt = x * (1 - at).sqrt() + at.sqrt() * temp_y

        next_t = (torch.ones(n) * j).to(x.device).long()
        at_next = compute_alpha(b, next_t)

        et = model(xt, t)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        if sigma_y == 0:
            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())
        else:
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            x0_t_hat = x0_t - A_funcs.Lambda(
                A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta
            ).reshape(*x0_t.size())

        x0t_pred_01 = (x0_t_hat + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        x0t_change_l2 = torch.norm(temp_y_01 - x0t_pred_01, p=2)
        l2_values.append(x0t_change_l2.item())

        if x0t_change_l2.item() < l2thr:
            start_idx = t_idx
            break
    else:
        start_idx = len(time_pairs) - 1

    if args.l2_monitor:
        os.makedirs(os.path.join(args.image_folder, "x0t_change_l2"), exist_ok=True)
        csv_path = os.path.join(args.image_folder, "x0t_change_l2", "l2_values.csv")
        file_exists = os.path.isfile(csv_path)

        import csv
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                header = ['projection_idx', 'selected_step'] + [f'step_{i}' for i in range(len(l2_values))]
                writer.writerow(header)
            row = [idx_so_far, start_idx] + l2_values
            writer.writerow(row)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(l2_values)), l2_values, 'b-', linewidth=2, label='L2 Distance')
        plt.axhline(y=l2thr, color='r', linestyle='--', linewidth=2, label=f'Threshold ({l2thr})')
        plt.axvline(x=start_idx, color='g', linestyle='--', linewidth=2, label=f'Start Step ({start_idx})')
        plt.xlabel('Time Step Index')
        plt.ylabel('L2 Distance')
        plt.title(f'L2 Distance vs Time Step (Projection {idx_so_far:02d})')
        plt.xlim(0, 50)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(args.image_folder, "x0t_change_l2", f"proj_{idx_so_far:02d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    return start_idx


########################################
# 2) Angle-aware L2 threshold
########################################

def select_start_step_angle_aware(args, b, x, y, temp_y, model,
                                  time_pairs, skip, eta, A_funcs,
                                  sigma_y, idx_so_far=0, total_projs=100):
    n = x.size(0)
    l2thr_base = args.l2thr

    angle = get_projection_angle(idx_so_far, total_projs)
    if is_frontal_view(angle, frontal_range=30):
        mul = getattr(args, "angle_aware_frontal_increase", 1.2)
        l2thr = l2thr_base * mul
        view_type = "Frontal"
    elif is_lateral_view(angle, lateral_range=30):
        l2thr = l2thr_base
        view_type = "Lateral"
    else:
        l2thr = l2thr_base
        view_type = "Intermediate"

    l2_values = []

    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)

    for t_idx, (i, j) in tqdm(enumerate(time_pairs),
                              desc=f"PAS angle-aware (proj {idx_so_far}, {view_type}, angle {angle:.1f})"):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        t = (torch.ones(n) * i).to(x.device).long()
        at = compute_alpha(b, t)
        xt = x * (1 - at).sqrt() + at.sqrt() * temp_y

        next_t = (torch.ones(n) * j).to(x.device).long()
        at_next = compute_alpha(b, next_t)

        et = model(xt, t)
        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        if sigma_y == 0:
            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())
        else:
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            x0_t_hat = x0_t - A_funcs.Lambda(
                A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta
            ).reshape(*x0_t.size())

        x0t_pred_01 = (x0_t_hat + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        x0t_change_l2 = torch.norm(temp_y_01 - x0t_pred_01, p=2)
        l2_values.append(x0t_change_l2.item())

        if x0t_change_l2.item() < l2thr:
            start_idx = t_idx
            break
    else:
        start_idx = len(time_pairs) - 1

    if args.l2_monitor:
        os.makedirs(os.path.join(args.image_folder, "angle_aware_l2"), exist_ok=True)
        csv_path = os.path.join(args.image_folder, "angle_aware_l2", "l2_values.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['proj_idx', 'angle', 'view', 'selected_step', 'thr'] + \
                         [f'step_{i}' for i in range(len(l2_values))]
                writer.writerow(header)
            row = [idx_so_far, f"{angle:.1f}", view_type, start_idx, f"{l2thr:.3f}"] + \
                  [f"{v:.4f}" for v in l2_values]
            writer.writerow(row)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(l2_values)), l2_values, 'b-', linewidth=2, label='L2 Distance')
        plt.axhline(y=l2thr, color='r', linestyle='--', linewidth=2,
                    label=f'Threshold ({l2thr:.3f})')
        plt.axvline(x=start_idx, color='g', linestyle='--', linewidth=2,
                    label=f'Start Step ({start_idx})')
        plt.xlabel('Time Step Index')
        plt.ylabel('L2 Distance')
        plt.title(f'Angle-aware PAS (Proj {idx_so_far:02d}, {view_type}, {angle:.1f}°)')
        plt.xlim(0, 50)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(args.image_folder, "angle_aware_l2", f"proj_{idx_so_far:02d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    return start_idx


########################################
# 3) SSIM-based PAS
########################################

def select_start_step_ssim_based(args, b, x, y, temp_y, model,
                                 time_pairs, skip, eta, A_funcs,
                                 sigma_y, idx_so_far=0, total_projs=100):
    n = x.size(0)

    angle = get_projection_angle(idx_so_far, total_projs)
    if is_frontal_view(angle):
        ssim_thr = getattr(args, "ssim_threshold_frontal", 0.96)
        view_type = "Frontal"
    elif is_lateral_view(angle):
        ssim_thr = getattr(args, "ssim_threshold_lateral", 0.94)
        view_type = "Lateral"
    else:
        ssim_thr = getattr(args, "ssim_threshold_intermediate", 0.95)
        view_type = "Intermediate"

    ssim_values = []
    l2_values = []

    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)
    temp_np = temp_y_01.detach().cpu().numpy()  # (B,C,H,W) or (B,1,H,W)
    if temp_np.shape[1] > 1:
        temp_np = temp_np[:, 0]  # first channel

    for t_idx, (i, j) in tqdm(enumerate(time_pairs),
                              desc=f"PAS SSIM (proj {idx_so_far}, {view_type}, angle {angle:.1f})"):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        t = (torch.ones(n) * i).to(x.device).long()
        at = compute_alpha(b, t)
        xt = x * (1 - at).sqrt() + at.sqrt() * temp_y

        next_t = (torch.ones(n) * j).to(x.device).long()
        at_next = compute_alpha(b, next_t)

        et = model(xt, t)
        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        if sigma_y == 0:
            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())
        else:
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            x0_t_hat = x0_t - A_funcs.Lambda(
                A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta
            ).reshape(*x0_t.size())

        x0t_pred_01 = (x0_t_hat + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        x0_np = x0t_pred_01.detach().cpu().numpy()
        if x0_np.shape[1] > 1:
            x0_np = x0_np[:, 0]

        ssim_list = []
        for b_idx in range(x0_np.shape[0]):
            ssim_val = structural_similarity(
                temp_np[b_idx], x0_np[b_idx], data_range=1.0
            )
            ssim_list.append(ssim_val)
        ssim_val = float(np.mean(ssim_list))
        ssim_values.append(ssim_val)

        l2_val = torch.norm(temp_y_01 - x0t_pred_01, p=2).item()
        l2_values.append(l2_val)

        if ssim_val >= ssim_thr:
            start_idx = t_idx
            break
    else:
        start_idx = len(time_pairs) - 1

    if args.l2_monitor:
        base_dir = os.path.join(args.image_folder, "ssim_based_pas")
        os.makedirs(base_dir, exist_ok=True)

        csv_path = os.path.join(base_dir, "ssim_values.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['proj_idx', 'angle', 'view', 'selected_step', 'ssim_thr'] + \
                         [f'ssim_{i}' for i in range(len(ssim_values))]
                writer.writerow(header)
            row = [idx_so_far, f"{angle:.1f}", view_type, start_idx, f"{ssim_thr:.3f}"] + \
                  [f"{v:.4f}" for v in ssim_values]
            writer.writerow(row)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ssim_values)), ssim_values, 'b-', linewidth=2, label='SSIM')
        plt.axhline(y=ssim_thr, color='r', linestyle='--', linewidth=2,
                    label=f'Threshold ({ssim_thr:.3f})')
        plt.axvline(x=start_idx, color='g', linestyle='--', linewidth=2,
                    label=f'Start Step ({start_idx})')
        plt.xlabel('Time Step Index')
        plt.ylabel('SSIM')
        plt.title(f'SSIM-based PAS (Proj {idx_so_far:02d}, {view_type}, {angle:.1f}°)')
        plt.ylim(0.8, 1.0)
        plt.xlim(0, 50)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(base_dir, f"proj_{idx_so_far:02d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    return start_idx


########################################
# 4) Variance tracking PAS
########################################

def select_start_step_variance_tracking(args, b, x, y, temp_y, model,
                                        time_pairs, skip, eta, A_funcs,
                                        sigma_y, idx_so_far=0, total_projs=100):
    n = x.size(0)
    l2thr = args.l2thr
    window_size = getattr(args, "variance_window_size", 5)
    var_thr = getattr(args, "variance_threshold", 1e-3)

    angle = get_projection_angle(idx_so_far, total_projs)
    if is_frontal_view(angle):
        view_type = "Frontal"
    elif is_lateral_view(angle):
        view_type = "Lateral"
    else:
        view_type = "Intermediate"

    l2_values = []
    var_values = []

    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)

    for t_idx, (i, j) in tqdm(enumerate(time_pairs),
                              desc=f"PAS Var (proj {idx_so_far}, {view_type}, angle {angle:.1f})"):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        t = (torch.ones(n) * i).to(x.device).long()
        at = compute_alpha(b, t)
        xt = x * (1 - at).sqrt() + at.sqrt() * temp_y

        next_t = (torch.ones(n) * j).to(x.device).long()
        at_next = compute_alpha(b, next_t)

        et = model(xt, t)
        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        if sigma_y == 0:
            x0_t_hat = x0_t - A_funcs.A_pinv(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())
        else:
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            x0_t_hat = x0_t - A_funcs.Lambda(
                A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta
            ).reshape(*x0_t.size())

        x0t_pred_01 = (x0_t_hat + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        l2_val = torch.norm(temp_y_01 - x0t_pred_01, p=2).item()
        l2_values.append(l2_val)

        if len(l2_values) >= window_size:
            recent = l2_values[-window_size:]
            var = float(np.var(recent))
        else:
            var = np.inf
        var_values.append(var)

        if (l2_val < l2thr) and (var < var_thr):
            start_idx = t_idx
            break
    else:
        start_idx = len(time_pairs) - 1

    if args.l2_monitor:
        base_dir = os.path.join(args.image_folder, "variance_pas")
        os.makedirs(base_dir, exist_ok=True)

        csv_path = os.path.join(base_dir, "variance_values.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['proj_idx', 'angle', 'view', 'selected_step', 'l2_thr', 'var_thr'] + \
                         [f'l2_{i}' for i in range(len(l2_values))] + \
                         [f'var_{i}' for i in range(len(var_values))]
                writer.writerow(header)
            row = [idx_so_far, f"{angle:.1f}", view_type, start_idx,
                   f"{l2thr:.3f}", f"{var_thr:.5f}"] + \
                  [f"{v:.4f}" for v in l2_values] + \
                  [("nan" if np.isinf(v) else f"{v:.6f}") for v in var_values]
            writer.writerow(row)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(range(len(l2_values)), l2_values, 'b-', linewidth=2, label='L2')
        ax1.axhline(y=l2thr, color='r', linestyle='--', linewidth=2,
                    label=f'L2 thr ({l2thr:.3f})')
        ax1.axvline(x=start_idx, color='g', linestyle='--', linewidth=2,
                    label=f'Start {start_idx}')
        ax1.set_ylabel('L2')
        ax1.set_xlim(0, 50)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        valid_var = [np.nan if np.isinf(v) else v for v in var_values]
        ax2.plot(range(len(valid_var)), valid_var, 'orange', linewidth=2, label='Var(L2)')
        ax2.axhline(y=var_thr, color='r', linestyle='--', linewidth=2,
                    label=f'Var thr ({var_thr:.5f})')
        ax2.axvline(x=start_idx, color='g', linestyle='--', linewidth=2,
                    label=f'Start {start_idx}')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Variance')
        ax2.set_xlim(0, 50)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(base_dir, f"proj_{idx_so_far:02d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    return start_idx


########################################
# 5) Unified wrapper
########################################

def select_start_step_unified(args, b, x, y, temp_y, model,
                              time_pairs, skip, eta, A_funcs,
                              sigma_y, idx_so_far=0, total_projs=100):
    """
    args.pas_strategy: 'original', 'angle_aware', 'ssim_based', 'variance_tracking'
    """
    strategy = getattr(args, "pas_strategy", "original")

    if strategy == "angle_aware":
        return select_start_step_angle_aware(
            args, b, x, y, temp_y, model,
            time_pairs, skip, eta, A_funcs,
            sigma_y, idx_so_far=idx_so_far, total_projs=total_projs
        )
    elif strategy == "ssim_based":
        return select_start_step_ssim_based(
            args, b, x, y, temp_y, model,
            time_pairs, skip, eta, A_funcs,
            sigma_y, idx_so_far=idx_so_far, total_projs=total_projs
        )
    elif strategy == "variance_tracking":
        return select_start_step_variance_tracking(
            args, b, x, y, temp_y, model,
            time_pairs, skip, eta, A_funcs,
            sigma_y, idx_so_far=idx_so_far, total_projs=total_projs
        )
    else:
        return select_start_step_original(
            args, b, x, y, temp_y, model,
            time_pairs, skip, eta, A_funcs,
            sigma_y, idx_so_far=idx_so_far
        )
