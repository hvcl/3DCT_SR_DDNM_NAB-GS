import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion_temporal(x, model, b, eta, A_funcs, y, y_prev=None, y_after=None, n_prev=None, n_next=None, cls_fn=None, classes=None, config=None):
    with torch.no_grad():
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]
        xs_prev = [x] if n_prev is None else [n_prev]
        xs_after = [x] if n_next is None else [n_next]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        # print(times)
        # [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1]
        
        time_pairs = list(zip(times[:-1], times[1:])) # [49, 48], [48, 47], ..., [0, -1]
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda') # shape: [1, 3, 512, 512]
                xt_prev = None
                if y_prev is not None:
                    xt_prev = xs_prev[-1].to('cuda') # shape: [1, 3, 512, 512]
                xt_after = None
                if y_after is not None:
                    xt_after = xs_after[-1].to('cuda') # shape: [1, 3, 512, 512]
                
                # if cls_fn == None: # here
                #     if y_prev is not None:
                #         y_prev_t = (at.sqrt() * y_prev + (1 - at).sqrt() * torch.randn_like(xs[0])).to('cuda')
                #     else:
                #         y_prev_t = None
                #     if y_after is not None:
                #         y_after_t = (at_next.sqrt() * y_after + (1 - at_next).sqrt() * torch.randn_like(xs[0])).to('cuda')
                #     else:
                #         y_after_t = None
                    
                et = model(xt, t, y_prev_t=xt_prev, y_after_t=xt_after)
                xt_b_size = et.shape[0]

                # case 1: only xt is used
                # case 2: xt and xt_prev are used
                # case 3: xt, xt_prev, and xt_after are used
                # case 4: xt and xt_prev are used

                et_prev = None
                et_after = None

                if xt_b_size == 2:
                    if y_prev is not None:
                        et, et_prev = torch.chunk(et, 2, dim=0)
                        et_after = None
                    else:
                        et, et_after = torch.chunk(et, 2, dim=0)
                        et_prev = None
                elif xt_b_size == 3:
                    et, et_prev, et_after = torch.chunk(et, 3, dim=0)

                    # if x_prev is not None: # add noise to x_prev_t
                    #     x_prev_t = (at.sqrt() * x_prev + (1 - at).sqrt() * torch.randn_like(xs[0])).to('cuda')
                    #     et = model(xt, t, x_prev=x_prev_t)
                    # else:
                    #     et = model(xt, t)
                # else:
                #     classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                #     et = model(xt, t, classes)
                #     et = et[:, :3]
                #     et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]
                    et_prev = et_prev[:, :3] if et_prev is not None else None
                    et_after = et_after[:, :3] if et_after is not None else None

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_prev_t = (xt_prev - et_prev * (1 - at).sqrt()) / at.sqrt() if y_prev is not None else None
                x0_after_t = (xt_after - et_after * (1 - at).sqrt()) / at.sqrt() if y_after is not None else None

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())
                x0_prev_t_hat = x0_prev_t - A_funcs.A_pinv(
                    A_funcs.A(x0_prev_t.reshape(x0_prev_t.size(0), -1)) - y_prev.reshape(y_prev.size(0), -1)
                ).reshape(*x0_prev_t.size()) if y_prev is not None else None
                x0_after_t_hat = x0_after_t - A_funcs.A_pinv(
                    A_funcs.A(x0_after_t.reshape(x0_after_t.size(0), -1)) - y_after.reshape(y_after.size(0), -1)
                ).reshape(*x0_after_t.size()) if y_after is not None else None

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et
                xt_prev_next = at_next.sqrt() * x0_prev_t_hat + c1 * torch.randn_like(x0_prev_t) + c2 * et_prev if y_prev is not None else None
                xt_after_next = at_next.sqrt() * x0_after_t_hat + c1 * torch.randn_like(x0_after_t) + c2 * et_after if y_after is not None else None

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
                xs_prev.append(xt_prev_next.to('cpu')) if y_prev is not None else None
                xs_after.append(xt_after_next.to('cpu')) if y_after is not None else None
            # else: # time-travel back -> X
            #     next_t = (torch.ones(n) * j).to(x.device)
            #     at_next = compute_alpha(b, next_t.long())
            #     x0_t = x0_preds[-1].to('cuda')
                
            #     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

            #     xs.append(xt_next.to('cpu'))

    # if x_prev is not None:
    #     xs[-1] = xs[-1].chuck(2, dim=0)
    #     x0_preds[-1] = x0_preds[-1].chuck(2, dim=0)
    return [xs[-1]], [x0_preds[-1]]

def ddnm_plus_diffusion_temporal(x, model, b, eta, A_funcs, y, y_prev=None, y_after=None, sigma_y=0.0, cls_fn=None, classes=None, config=None):
    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]
        xs_prev = [x] if y_prev is not None else None
        xs_after = [x] if y_after is not None else None

        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))
        
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i:
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                xt_prev = xs_prev[-1].to('cuda') if y_prev is not None else None
                xt_after = xs_after[-1].to('cuda') if y_after is not None else None
                
                et = model(xt, t, y_prev_t=xt_prev, y_after_t=xt_after)
                xt_b_size = et.shape[0]

                et_prev = et_after = None
                if xt_b_size == 2:
                    if y_prev is not None:
                        et, et_prev = torch.chunk(et, 2, dim=0)
                    else:
                        et, et_after = torch.chunk(et, 2, dim=0)
                elif xt_b_size == 3:
                    et, et_prev, et_after = torch.chunk(et, 3, dim=0)

                if et.size(1) == 6:
                    et = et[:, :3]
                    et_prev = et_prev[:, :3] if et_prev is not None else None
                    et_after = et_after[:, :3] if et_after is not None else None

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_prev_t = (xt_prev - et_prev * (1 - at).sqrt()) / at.sqrt() if y_prev is not None else None
                x0_after_t = (xt_after - et_after * (1 - at).sqrt()) / at.sqrt() if y_after is not None else None

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                x0_prev_t_hat = x0_prev_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_prev_t.reshape(x0_prev_t.size(0), -1)) - y_prev.reshape(y_prev.size(0), -1)
                ).reshape(x0_prev_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_prev_t.size()) if y_prev is not None else None

                x0_after_t_hat = x0_after_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_after_t.reshape(x0_after_t.size(0), -1)) - y_after.reshape(y_after.size(0), -1)
                ).reshape(x0_after_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_after_t.size()) if y_after is not None else None

                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                xt_prev_next = at_next.sqrt() * x0_prev_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_prev_t).reshape(x0_prev_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et_prev.reshape(et_prev.size(0), -1)).reshape(*x0_prev_t.size()) if y_prev is not None else None

                xt_after_next = at_next.sqrt() * x0_after_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_after_t).reshape(x0_after_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et_after.reshape(et_after.size(0), -1)).reshape(*x0_after_t.size()) if y_after is not None else None

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
                if y_prev is not None:
                    xs_prev.append(xt_prev_next.to('cpu'))
                if y_after is not None:
                    xs_after.append(xt_after_next.to('cpu'))

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
