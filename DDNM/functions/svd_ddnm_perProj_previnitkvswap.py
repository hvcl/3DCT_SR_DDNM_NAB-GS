import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

class_num = 951


def _predict_eps_with_optional_ref(model, xt, t, ref_y, at=None, base_y=None, cls_fn=None, classes=None):
    """
    If ref_y is None, run the plain single-image denoiser.
    Otherwise, build a 2-sample batch:
      [0] noised reference, [1] current target xt
    and use KV-swap inside the model on that pair.
    """
    if ref_y is None:
        if cls_fn is None:
            et = model(xt, t)
        else:
            classes_single = torch.ones(xt.size(0), dtype=torch.long, device=xt.device) * class_num
            et = model(xt, t, classes_single)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(xt, t, classes_single)
        return et[:, :3] if et.size(1) == 6 else et

    noised_ref = at.sqrt() * ref_y + (1 - at).sqrt() * torch.randn_like(ref_y).to(ref_y.device)
    x_concat = torch.cat([noised_ref, xt], dim=0)
    t_concat = torch.cat([t, t], dim=0)

    if cls_fn is None:
        et_concat = model(x_concat, t_concat)
        et = et_concat[1:2]
    else:
        classes_concat = torch.ones(x_concat.size(0), dtype=torch.long, device=xt.device) * class_num
        et_concat = model(x_concat, t_concat, classes_concat)
        et = et_concat[1:2, :3]
        et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(xt, t, classes_concat[1:2])

    return et[:, :3] if et.size(1) == 6 else et

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

def ddnm_diffusion(x, model, b, eta, A_funcs, y, temp_y=None, ref_y=None, cls_fn=None, classes=None, config=None, args=None, init_start_step=0):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        x0t_preds = []
        xs = [x]
        # generate time schedule
        # times = get_schedule_jump(config.time_travel.T_sampling, 
        #                         config.time_travel.travel_length, 
        #                         config.time_travel.travel_repeat,
        #                         )
        start_step = 0
        if getattr(args, "use_pas", False):
            times_select = get_schedule_jump(config.time_travel.T_sampling, 1, 1)
            time_pairs_select = list(zip(times_select[:-1], times_select[1:]))
            start_step = select_start_step(args, b, x, y, temp_y, model, time_pairs_select, skip, eta, A_funcs, sigma_y=0)
            times = get_schedule_jump(50 - start_step, config.time_travel.travel_length, config.time_travel.travel_repeat)
            time_pairs = list(zip(times[:-1], times[1:]))
        else:
            times = get_schedule_jump(
                config.time_travel.T_sampling,
                config.time_travel.travel_length,
                config.time_travel.travel_repeat,
            )
            time_pairs = list(zip(times[:-1], times[1:]))
            if init_start_step > 0:
                start_step = int(init_start_step)
                time_pairs = time_pairs[start_step:]

        # reverse diffusion sampling
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            i, j = i*skip, j*skip

            if t_idx == 0:
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                # xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
                x_current = xs[-1].to(x.device)
                xt = x_current * (1 - at).sqrt() + at.sqrt() * temp_y
                xs.append(xt.to('cpu'))

            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to(x.device)
                
                et = _predict_eps_with_optional_ref(
                    model,
                    xt,
                    t,
                    ref_y,
                    at=at,
                    base_y=temp_y,
                    cls_fn=cls_fn,
                    classes=classes,
                )

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
            # else: # time-travel back
            #     next_t = (torch.ones(n) * j).to(x.device)
            #     at_next = compute_alpha(b, next_t.long())
            #     x0_t = x0_preds[-1].to('cuda')
                
            #     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

            #     xs.append(xt_next.to('cpu'))

    # UHRCT 조건 확인 후 다운샘플링
    if "UHRCT" in args.config: # and "down2x" in args.image_folder:
        xs_last_downsampled = downsample_tensor([xs[-1]])
        x0_preds_downsampled = downsample_tensor(x0_preds)
        x0t_preds_downsampled = downsample_tensor(x0t_preds)
        return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
    else:
        return [xs[-1]], x0_preds, x0t_preds, start_step

    # return [xs[-1]], x0_preds, x0t_preds, start_step

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, temp_y=None, ref_y=None, cls_fn=None, classes=None, config=None, args=None, init_start_step=0):
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
        start_step = 0
        if getattr(args, "use_pas", False):
            start_step = select_start_step(args, b, x, y, temp_y, model, time_pairs, skip, eta, A_funcs, sigma_y)
            time_pairs = time_pairs[start_step:]
        elif init_start_step > 0:
            start_step = int(init_start_step)
            time_pairs = time_pairs[start_step:]

        # reverse diffusion sampling
        for t_idx, (i, j) in tqdm(enumerate(time_pairs)):
            i, j = i*skip, j*skip

            if t_idx == 0:
                # temp_y: reshape y into xs[-1] shape
                t = (torch.ones(n) * i).to(x.device).long()
                at = compute_alpha(b, t.long())
                # xt = (at.sqrt() * xs[-1] + (1 - at).sqrt() * xs[-1])
                x_current = xs[-1].to(x.device)
                xt = x_current * (1 - at).sqrt() + at.sqrt() * temp_y
                xs.append(xt.to('cpu'))

            if j<0: j=-1 

            if j < i: # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                
                xt = xs[-1].to(x.device)

                et = _predict_eps_with_optional_ref(
                    model,
                    xt,
                    t,
                    ref_y,
                    at=at,
                    base_y=temp_y,
                    cls_fn=cls_fn,
                    classes=classes,
                )

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
            # else: # time-travel back
            #     next_t = (torch.ones(n) * j).to(x.device)
            #     at_next = compute_alpha(b, next_t.long())
            #     x0_t = x0_preds[-1].to('cuda')
                
            #     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

            #     xs.append(xt_next.to('cpu'))

    # UHRCT 조건 확인 후 다운샘플링
    if "UHRCT" in args.config: # and "down2x" in args.image_folder:
        xs_last_downsampled = downsample_tensor([xs[-1]])
        x0_preds_downsampled = downsample_tensor(x0_preds)
        x0t_preds_downsampled = downsample_tensor(x0t_preds)
        return xs_last_downsampled, x0_preds_downsampled, x0t_preds_downsampled, start_step
    else:
        return [xs[-1]], x0_preds, x0t_preds, start_step

    # return [xs[-1]], x0_preds, x0t_preds, start_step

def select_start_step(args, b, x, y, temp_y, model, time_pairs, skip, eta, A_funcs, sigma_y):
    n = x.size(0)
    l2thr = args.l2thr
    
    x0_preds = []
    x0t_preds = []
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

        x0_preds.append(x0_t.to('cpu'))
        x0t_preds.append(x0_t_hat.to('cpu'))

    # Save the difference graph between x0t_pred and temp_y
    # normalize value range: [0, 1]
    temp_y = temp_y.to('cpu')
    temp_y_01 = (temp_y + 1.0) / 2.0
    temp_y_01 = torch.clamp(temp_y_01, 0.0, 1.0)
    # temp_y_01 = temp_y_01.squeeze(0).permute(1, 2, 0) # [1, 3, 512, 512]
    
    x0_dir = os.path.join(args.image_folder, "x0")
    if not os.path.exists(x0_dir):
        os.makedirs(x0_dir)
    x0t_dir = os.path.join(args.image_folder, "x0t")
    if not os.path.exists(x0t_dir):
        os.makedirs(x0t_dir)

    # x0t_changes_l1 = []
    x0t_changes_l2 = []
    for i, (x0_pred, x0t_pred) in enumerate(zip(x0_preds, x0t_preds)): # [20:]):
        # x0_pred_01 = (x0_pred + 1.0) / 2.0
        # x0_pred_01 = torch.clamp(x0_pred_01, 0.0, 1.0)
        # x0_pred_01 = x0_pred_01.squeeze(0).permute(1, 2, 0) # shape: [3, 512, 512] -> [512, 512, 3]
        
        x0t_pred_01 = (x0t_pred + 1.0) / 2.0
        x0t_pred_01 = torch.clamp(x0t_pred_01, 0.0, 1.0)
        # x0t_pred_01 = x0t_pred_01.squeeze(0).permute(1, 2, 0)

        # x0t_change_l1 = torch.norm(temp_y_01 - x0t_pred_01, p=1)
        x0t_change_l2 = torch.norm(temp_y_01 - x0t_pred_01, p=2)

        # x0t_changes_l1.append(x0t_change_l1)
        x0t_changes_l2.append(x0t_change_l2)

        # save x0, x0t
        # torchvision.utils.save_image(x0_pred_01, os.path.join(x0_dir, f"x0_{i:02d}.png"))
        # torchvision.utils.save_image(x0t_pred_01, os.path.join(x0t_dir, f"x0t_{i:02d}.png"))
    
    # x0t_change graph directory
    x0t_change_dir = os.path.join(args.image_folder, "x0t_change")
    if not os.path.exists(x0t_change_dir):
        os.makedirs(x0t_change_dir)

    # x0t_pred 변화량 그래프
    # from matplotlib import pyplot as plt
    # # plt.figure(figsize=(10, 5))
    # # plt.plot(x0t_changes_l1, label="L1")
    # # plt.title("x0t_changes_l1")
    # # plt.xlabel("Timestep")
    # # plt.ylabel("L1 norm change")
    # # plt.savefig(os.path.join(x0t_change_dir, "x0t_changes_l1.png"))
    # # plt.close()
    
    # if sigma_y>0:
    #     plt_title = "L2 norm change 4x"
    #     plt_grad_title = "L2 norm change gradient 4x"
    # else:
    #     plt_title = "L2 norm change 8x"
    #     plt_grad_title = "L2 norm change gradient 8x"

    # plt.figure(figsize=(10, 5))
    # plt.plot(x0t_changes_l2, label="L2")
    # plt.title(plt_title)
    # plt.xlabel("Timestep")
    # plt.ylabel("L2 norm change")
    # plt.savefig(os.path.join(x0t_change_dir, "x0t_changes_l2.png"))
    # plt.close()

    # # x0t_pred 변화량의 gradient 그래프
    # plt.figure(figsize=(10, 5))
    # plt.plot(torch.tensor(x0t_changes_l2).diff(), label="L2")
    # plt.title(plt_grad_title)
    # plt.xlabel("Timestep")
    # plt.ylabel("L2 norm change gradient")
    # plt.savefig(os.path.join(x0t_change_dir, "x0t_changes_l2_gradient.png"))
    # plt.close()

    # # save as txt (gradient and l2 norm changes), not in torch tensor -> as each line is a value
    # with open(os.path.join(x0t_change_dir, "x0t_changes_l2.txt"), "w") as f:
    #     for item in x0t_changes_l2:
    #         f.write(f"{item}\n")
    # with open(os.path.join(x0t_change_dir, "x0t_changes_l2_gradient.txt"), "w") as f:
    #     for item in torch.tensor(x0t_changes_l2).diff():
    #         f.write(f"{item}\n")
    
    # set start step when L2 norm change is less than L2thr
    start_step = 0
    
    for i, x0t_change in enumerate(x0t_changes_l2):
        if x0t_change < l2thr:
            start_step = i
            break
    
    return start_step

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
