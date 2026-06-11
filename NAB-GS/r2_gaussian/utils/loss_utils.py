#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def resize_to(vol_pred, target_shape):
    # vol_pred, vol_lr: (D,H,W)
    vp = vol_pred.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    D, H, W = target_shape.shape
    
    y = F.adaptive_avg_pool3d(vp, output_size=(D, H, W))

    return y.squeeze(0).squeeze(0)    # (D,H,W)

def l1_loss_consistency(vol_pred, gt):
    return torch.abs((resize_to(vol_pred, gt) - gt)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def compute_power_spectrum(image):
    f_img = torch.fft.fft2(image, norm='ortho')
    f_img_shifted = torch.fft.fftshift(f_img)
    power = f_img_shifted.abs() ** 2
    return power, f_img_shifted

def compute_energy_threshold(tensor, target_ratio=0.9999):
    flat = tensor.flatten()
    sorted_vals, _ = torch.sort(flat, descending=True)
    cumulative = torch.cumsum(sorted_vals, dim=0)
    total = cumulative[-1]
    idx = (cumulative / total >= target_ratio).nonzero()[0].item()
    return sorted_vals[idx].item()

def get_common_mask(power_A, target_ratio=0.9999):
    threshold = compute_energy_threshold(power_A, target_ratio)
    mask = (power_A > threshold).float()
    return mask

def reconstruct_image_from_freq(freq_domain):
    f_ishift = torch.fft.ifftshift(freq_domain)
    img = torch.fft.ifft2(f_ishift, norm='ortho').real
    return img


def LR_con_HR_freq_loss(lr_image, pred, gt, weight, alpha):
    lr_image = lr_image.unsqueeze(0)
    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)
    power_lr, _ = compute_power_spectrum(lr_image)
    mask = get_common_mask(power_lr, target_ratio=0.9995)

    _, f_pred = compute_power_spectrum(pred)
    _, f_gt = compute_power_spectrum(gt)

    
    f_pred_masked = f_pred * mask
    pred_lr = reconstruct_image_from_freq(f_pred_masked)

    f_pred_residual = f_pred * (1 - mask)
    pred_hr = reconstruct_image_from_freq(f_pred_residual)

    f_gt_residual = f_gt * (1 - mask)
    gt_hr = reconstruct_image_from_freq(f_gt_residual)
    
    lr_loss = F.l1_loss(lr_image, pred_lr)
    hr_loss = F.l1_loss(gt_hr, pred_hr)

    loss = weight * (alpha * lr_loss + (1-alpha) * hr_loss)
    # loss = weight * hr_loss

    return loss

def get_3d_gaussian_kernel(kernel_size=5, sigma=1.0, channels=1, device='cpu'):
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss /= gauss.sum()
    
    kernel_3d = gauss[:, None, None] * gauss[None, :, None] * gauss[None, None, :]
    kernel_3d = kernel_3d.expand(channels, 1, kernel_size, kernel_size, kernel_size)
    
    return kernel_3d

def volume_consistency(pred, lr_vol, scale=4, weight=1.0):
    pred = pred.unsqueeze(0).unsqueeze(0)
    lr_vol = lr_vol.unsqueeze(0).unsqueeze(0)
    device = pred.device
    print(pred.shape)

    kernel_size = 5
    sigma = 1.0
    kernel = get_3d_gaussian_kernel(kernel_size, sigma, channels=1, device=device)

    # pred_padded = F.pad(pred, pad=[2,2,2,2,2,2], mode='reflect') 
    # pred_blurred = F.conv3d(pred_padded, kernel, groups=1)
    pred_blurred = F.conv3d(pred, kernel, groups=1)

    downsampled_pred = F.interpolate(pred_blurred, size=lr_vol.shape[2:], mode='trilinear', align_corners=True)

    # downsampled_pred = F.avg_pool3d(pred, (scale, scale, scale))

    loss = weight * F.l1_loss(downsampled_pred, lr_vol)

    return loss

def matching_prior_loss(gaussian, render_pkg_A, render_pkg_B, K_B, R_B, t_B, weight):
    '''
    Compare 3D Gaussians visible from view A to their spatially nearest neighbors in view B,
    by projecting A's points into B's image plane and finding the closest corresponding 3D point.

    Args:
    render_pkg_A, render_pkg_B: Dicts containing:
        - "viewspace_points": (N, 3) float tensor of Gaussian centers in 3D
        - "visibility_filter": (N,) boolean tensor indicating visible points
    K_B, R_B, t_B: Camera B's intrinsic matrix (3x3), rotation (3x3), and translation (3,)
    weight: scalar multiplier for loss scaling
    
    Process Summary:
        1. Extract visible 3D points from view A.
        2. Transform them into camera B space using R_B and t_B.
        3. Project them to pixel coordinates using K_B.
        4. For each projected 2D point, find the nearest 3D point in view B (based on screen space).
        5. Compute L1 loss between original 3D points from view A and matched 3D points from view B.
    '''
    K_B = torch.from_numpy(K_B).cuda().float()
    R_B = torch.from_numpy(R_B).cuda().float()
    t_B = torch.from_numpy(t_B).cuda().float()

    pts_A = render_pkg_A["viewspace_points"]     # (N_total, 3)
    vis_A = render_pkg_A["visibility_filter"]    
    
    mask = vis_A > 0.5                            
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pts_A.device)

    X_A = pts_A[mask.view(-1)]                   # (N_visible, 3)
    density_A = gaussian._density[mask.view(-1)]

    X_B_cam = (R_B @ X_A.T + t_B[:, None]).T     # (N, 3)
    x_proj = (K_B @ X_B_cam.T)                   # (3, N)
    x_proj = x_proj[:2, :] / x_proj[2:3, :]      # (2, N)
    x_proj = x_proj.T                            # (N, 2)

    viewspace_B = render_pkg_B["viewspace_points"]  # (M, 3)
    screen_xy_B = compute_screen_xy(viewspace_B, K_B, R_B, t_B)         # (M, 2)

    # sample viewspace points
    dists = torch.cdist(x_proj, screen_xy_B)
    idx = torch.argmin(dists, dim=1)

    X_B_pred = viewspace_B[idx] # 3D location
    density_B = gaussian._density[idx]
    # X_B_pred = sample_viewspace_points(viewspace_B, screen_xy_B, x_proj)  # (N, 3)

    loss_loation = F.l1_loss(X_A, X_B_pred)
    loss_density = F.l1_loss(density_A, density_B)

    loss = loss_loation + loss_density

    return weight * loss


def compute_screen_xy(viewspace_points, K, R, t):
    '''
    Projects 3D points into 2D screen space using camera intrinsics and extrinsics.
    '''
    X_cam = (R @ viewspace_points.T + t[:, None]).T  # (M, 3)
    xy = (K @ X_cam.T)                               # (3, M)
    xy = xy[:2, :] / xy[2:3, :]                      # (2, M)
    return xy.T  # (M, 2)

def sample_viewspace_points(viewspace_points_B, screen_xy_B, xy_proj):
    '''
    For each 2D projected point (from A), find the nearest 2D screen-space point in B,
    and return its corresponding 3D position.
    '''
    dists = torch.cdist(xy_proj, screen_xy_B)   # (N, M)
    idx = torch.argmin(dists, dim=1)            # (N,)
    matched = viewspace_points_B[idx]           # (N, 3)
    return matched

def sinkhorn_loss_2d(
    pred,          # [1,H,W] or [H,W]  (현재 GS projection or SR projection)
    target,        # [1,H,W] or [H,W]  (LR projection or 기준 projection)
    eps=0.05,      # entropic regularization (0.01~0.1 정도 권장)
    n_iters=50,
    down_size=96,
    eps_prob=1e-8,
):
    """
    - pred, target: intensity image (0~1 근처라고 가정)
    - 이미지를 다운샘플해서 64x64 grid에서 OT를 계산 (full 512^2 x 512^2는 너무 큼)
    - log-domain Sinkhorn으로 NaN/Inf를 줄임
    """

    # 1) shape 정리: [1,H,W]
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    # 2) 다운샘플링 (gradient는 pred, target 둘 다 통과함)
    pred_ds   = F.interpolate(pred.unsqueeze(0),   size=(down_size, down_size),
                              mode='bilinear', align_corners=False).squeeze(0)  # [1,h,w]
    target_ds = F.interpolate(target.unsqueeze(0), size=(down_size, down_size),
                              mode='bilinear', align_corners=False).squeeze(0)  # [1,h,w]

    # 3) 음수 clamp + flatten + 확률분포화
    #    (Sinkhorn은 non-negative prob. measure를 가정)
    pred_flat   = pred_ds.clamp_min(0.0).reshape(-1)
    target_flat = target_ds.clamp_min(0.0).reshape(-1)

    # 완전히 0이면 확률분포가 안 되므로 작은 값 추가
    pred_sum   = pred_flat.sum()
    target_sum = target_flat.sum()
    if pred_sum < eps_prob:
        pred_flat = torch.ones_like(pred_flat) / pred_flat.numel()
    else:
        pred_flat = pred_flat / (pred_sum + eps_prob)

    if target_sum < eps_prob:
        target_flat = torch.ones_like(target_flat) / target_flat.numel()
    else:
        target_flat = target_flat / (target_sum + eps_prob)

    a = pred_flat            # source weights, shape [N]
    b = target_flat          # target weights, shape [N]

    N = a.shape[0]

    # 4) 위치 좌표 (0~1 정규화)
    h = w = down_size
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, h, device=pred.device),
        torch.linspace(0, 1, w, device=pred.device),
        indexing="ij"
    )
    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # [N,2]

    # cost matrix C_ij = ||x_i - y_j||^2  (먼저 [0,1]로 정규화 되어 있어서 range 제한)
    # N=64*64=4096 => C: [4096, 4096] ~ 16M entries (GPU 메모리 가능)
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)   # [N,1,2]-[1,N,2] = [N,N,2]
    C = (diff ** 2).sum(dim=-1)                        # [N,N]
    C = C / (C.max() + 1e-8)                           # 0~1 범위로 스케일

    # 5) log-domain Sinkhorn
    # logK = -C/eps
    logK = -C / eps

    # log a, log b
    log_a = torch.log(a + eps_prob)
    log_b = torch.log(b + eps_prob)

    # 초기 u, v (log-domain에서 0)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(n_iters):
        # log_u^{l+1} = log a - logsumexp(logK + log_v[None,:], dim=1)
        log_u = log_a - torch.logsumexp(logK + log_v.unsqueeze(0), dim=1)
        # log_v^{l+1} = log b - logsumexp(logK + log_u[:,None], dim=0)
        log_v = log_b - torch.logsumexp(logK + log_u.unsqueeze(1), dim=0)

        # 수치 안정성 체크 (optional)
        if not torch.isfinite(log_u).all() or not torch.isfinite(log_v).all():
            # NaN/Inf가 발생하면 큰 loss를 리턴해서 optimizer가 방향을 바꾸게 함
            return torch.tensor(1e3, device=pred.device)

    # 6) transport plan π = diag(u) K diag(v) 의 기대 cost 계산
    #    log π_ij = log_u[i] + log_v[j] + logK[i,j]
    log_pi = log_u.unsqueeze(1) + log_v.unsqueeze(0) + logK  # [N,N]
    # 기대 cost = Σ_ij π_ij C_ij
    pi = torch.exp(log_pi)    # log-domain에서 나온 것이라 상대적으로 안정
    # 혹시라도 아주 작은/큰 값들에서 NaN 나면 clamp
    pi = pi.clamp_min(0.0)

    loss_ot = (pi * C).sum()

    # 혹시 여전히 비유한정이면 fallback
    if not torch.isfinite(loss_ot):
        loss_ot = torch.tensor(1e3, device=pred.device)

    return loss_ot
