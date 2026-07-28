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

# def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
def ddnm_diffusion(x, model, b, eta, A_funcs, y, sigma_y, thr_tau, cls_fn=None, classes=None, config=None, args=None):
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

        if "lessSteps" in args.image_folder:
            startStep = int(args.image_folder.split("startStep_")[-1].split("-")[0])
            times = times[startStep:] # 10번째부터 시작 (T=800)
        
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # reverse diffusion sampling
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            i, j = i*skip, j*skip

            if i < thr_tau:
                model.enable_freeu()
                use_freeu = True
                if (model.s1==1.0 and model.s2==1.0 and model.b1==1.0 and model.b2==1.0):
                    use_freeu = False
            else:
                use_freeu = False

            if t_idx == 0:
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
                xs.append(xt.to('cpu'))

            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t, use_freeu=use_freeu, freeu_layers=args.freeu_layers)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes, use_freeu=use_freeu, freeu_layers=args.freeu_layers)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() # shape: (1, 3, 128, 128)

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                x0t_preds.append(x0_t_hat.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], x0_preds, x0t_preds

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, thr_tau, cls_fn=None, classes=None, config=None, args=None):
    with torch.no_grad():
        
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling # 20
        skip_tau = thr_tau // skip * 5
        n = x.size(0)
        x0_preds = []
        x0t_preds = []
        xs = [x]

        # generate time schedule
        if "fine_sampling" in args.image_folder:
            times = get_schedule_jump_thr(config.time_travel.T_sampling, 
                                        config.time_travel.travel_length, 
                                        config.time_travel.travel_repeat,
                                        thr_tau, 
                                        skip
                                        )
        else:
            times = get_schedule_jump(config.time_travel.T_sampling, 
                                   config.time_travel.travel_length, 
                                   config.time_travel.travel_repeat,
                                  )
        
        if "lessSteps" in args.image_folder:
            startStep = int(args.image_folder.split("startStep_")[-1].split("-")[0])
            times = times[startStep:] # 10번째부터 시작 (T=800)

        time_pairs = list(zip(times[:-1], times[1:]))        
        
        # reverse diffusion sampling
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            if not "fine_sampling" in args.image_folder:
                i, j = i*skip, j*skip

            if i < thr_tau:
                model.enable_freeu()
                use_freeu = True
                if (model.s1==1.0 and model.s2==1.0 and model.b1==1.0 and model.b2==1.0):
                    use_freeu = False
            else:
                use_freeu = False

            if t_idx == 0:
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
                xs.append(xt.to('cpu'))

            if j<0: j=-1 

            if j < i: # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t, use_freeu=use_freeu, freeu_layers=args.freeu_layers)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes, use_freeu=use_freeu, freeu_layers=args.freeu_layers)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # Eq. 17
                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                # Eq. 51
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t.to('cpu'))
                x0t_preds.append(x0_t_hat.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))
                
    return [xs[-1]], x0_preds, x0t_preds


def get_schedule_jump_thr(T_sampling, travel_length, travel_repeat, thr_tau, skip):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t > (thr_tau // skip):
        t = t - 1
        ts.append(t*skip)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t*skip)

    # thr_tau 아래의 time step을 세분화하는 부분
    if thr_tau == 60:
        num_segments = 15  # 원하는 세분화 정도 (예: 5등분)
    elif thr_tau == 200:
        num_segments = 40
    else:
        num_segments = 15 # 기본값 설정

    segment_length = thr_tau // num_segments  # 세분화된 간격 계산
    
    # thr_tau/skip 부터 시작하여 0에 가까워지도록 time step 추가
    current_t = thr_tau
    while current_t > 0:
        current_t -= (segment_length) # skip 값으로 나눠서 실제 step size 반영
        if current_t < 0:
            break
        ts.append(current_t)
    
    ts.append(0)
    ts.append(-1)

    # _check_times(ts, -1, T_sampling)
    return ts


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
