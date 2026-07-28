import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import torch.nn.functional as F


class_num = 951

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def cond_noise_sampling(src_noise, level=3):
    # this function is used to generate Warped Noise
    # originally from: https://github.com/mskwak01/How-I-Warped-Your-Noise-Unofficial/blob/main/How_I_Warped_Your_Noise_Unofficial.ipynb
    
    B, C, H, W = src_noise.shape

    up_factor = 2 ** level

    upscaled_means = F.interpolate(src_noise, scale_factor=(up_factor, up_factor), mode='nearest')

    up_H = up_factor * H
    up_W = up_factor * W

    """
        1) Unconditionally sample a discrete Nk x Nk Gaussian sample
    """

    raw_rand = torch.randn(B, C, up_H, up_W)

    """
        2) Remove its mean from it
    """

    Z_mean = raw_rand.unfold(2, up_factor, up_factor).unfold(3, up_factor, up_factor).mean((4, 5))
    Z_mean = F.interpolate(Z_mean, scale_factor=up_factor, mode='nearest')
    mean_removed_rand = raw_rand - Z_mean

    """
        3) Add the pixel value to it
    """

    up_noise = upscaled_means / up_factor + mean_removed_rand

    return up_noise

def ddnm_diffusion_with_warped_noise(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    """
    DDNM inference function modified to use Warped Noise instead of Gaussian random noise.
    """
    with torch.no_grad():
        # Setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = []

        # Generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling,
                                  config.time_travel.travel_length,
                                  config.time_travel.travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))

        # Initialize Warped Noise
        warped_noise = cond_noise_sampling(x, level=3)  # Replace Gaussian noise with Warped Noise
        xs.append(warped_noise)

        # Reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0:
                j = -1

            if j < i:  # Normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')

                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et -= (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else:  # Time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')

                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


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
