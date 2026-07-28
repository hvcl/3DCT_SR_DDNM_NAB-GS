import os
import torch
import numpy as np
import glob
import pickle
import json
import csv
import tqdm
import lpips
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from datasets import get_dataset_cascade

def load_npy_if_exists(path):
    if path and os.path.exists(path):
        a = np.load(path)
        # collapse channel dim if present
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0, :, :]
        return a.astype(np.float32)
    return None

def build_gt_stack(args, config):
    _, test_dataset = get_dataset_cascade(args, config)
    gt_list = []
    for item in test_dataset:
        # gt = item["gt_norm_vol"] * 255. # range: [0,255] -> 그대로 stack하면 됨.
        gt = item["gt_img"] * 255. # range: [0,255] -> 그대로 stack하면 됨.
        gt_list.append(gt)
    gt_stack = np.stack(gt_list, axis=0)
    gt_stack = gt_stack.astype(np.float32)
    return gt_stack

def compute_and_save_metrics_cascade(args, config, stage2_dir=None):
    """
    Compute PSNR/SSIM/LPIPS for:
        - stage1 (loaded from a pickled GS file here)
        - any stage2 files saved as stage2_lvl{idx}_preGF.npy and stage2_lvl{idx}_postGF.npy
    Saves per-stage CSV and JSON summary and a combined metrics JSON into args.image_folder.
    """
    combined_path = os.path.join(args.image_folder, "metrics_combined_summary.json")
    if os.path.exists(combined_path):
        try:
            with open(combined_path, "r") as f:
                metrics = json.load(f)
            print("Loaded existing metrics from", combined_path)
            return metrics
        except Exception:
            pass

    # --- load stage1 from hardcoded candidate (existing behavior) ---
    candidate = "/Alexandrite/jhnoh/r2_gaussian/NAB_GS_mela0050.pickle"
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Expected stage1 pickle not found: {candidate}")
    with open(candidate, 'rb') as f:
        data = pickle.load(f)
    out_stage1 = data["train"]["projections"].astype(np.float32)  # (N, 512, 512)

    # min-max norm to 0..255 (all projections at a time)
    out_stage1 = (out_stage1 - np.min(out_stage1)) / (np.max(out_stage1) - np.min(out_stage1)) * 255.0

    # # per-projection min-max normalize -> 0..255 (preserve current pipeline behavior)
    # for i in range(out_stage1.shape[0]):
    #     mn = float(np.min(out_stage1[i])); mx = float(np.max(out_stage1[i]))
    #     out_stage1[i] = (out_stage1[i] - mn) / (mx - mn) * 255.0

    stage2_pre_dict = {}
    if stage2_dir is not None and os.path.isdir(stage2_dir):
        pre_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_preGF.npy")))
        for p in pre_files:
            base = os.path.basename(p)
            try:
                idx = int(base.split("stage2_lvl")[1].split("_")[0])
            except Exception:
                idx = len(stage2_pre_dict)
            stage2_pre_dict[idx] = load_npy_if_exists(p)

    gt_stack = build_gt_stack(args, config)  # expected in 0..255

    # optional LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').to(config.device).eval()
    print("LPIPS model loaded.")

    # helper prepare
    def ensure_255(arr):
        a = arr.astype(np.float32)
        if np.nanmax(a) <= 1.01:
            a = a * 255.0
        return a

    def compute_and_save(name, pred_stack):
        if gt_stack is None or pred_stack is None:
            print(f"SKIP metrics {name}: GT or pred not available")
            return None
        pred = ensure_255(pred_stack)
        N = min(pred.shape[0], gt_stack.shape[0])
        ps = np.full(N, np.nan, dtype=np.float32)
        ss = np.full(N, np.nan, dtype=np.float32)
        lp = np.full(N, np.nan, dtype=np.float32)

        for i in tqdm.tqdm(range(N), desc=f"Computing metrics for {name}"):
            gt_i = gt_stack[i].astype(np.float32) # [3, 512, 512]
            gt_i = gt_i[0]
            pred_i = pred[i].astype(np.float32) # [512, 512]

            ps[i] = psnr_fn(gt_i, pred_i, data_range=255.0)
            ss[i] = ssim_fn(gt_i, pred_i, data_range=255.0)

            if lpips_model is not None:
                # lpips expects torch tensor in [-1,1] and 3 channels
                p_t = torch.from_numpy(pred_i / 255.0).float()
                g_t = torch.from_numpy(gt_i / 255.0).float()
                if p_t.ndim == 2:
                    p_t = p_t.unsqueeze(0)
                if g_t.ndim == 2:
                    g_t = g_t.unsqueeze(0)
                # make 3 channels by repeating
                p_t3 = p_t.unsqueeze(0).repeat(1, 3, 1, 1).to(config.device)
                g_t3 = g_t.unsqueeze(0).repeat(1, 3, 1, 1).to(config.device)
                # to [-1,1]
                p_t3 = (p_t3 * 2.0) - 1.0
                g_t3 = (g_t3 * 2.0) - 1.0
                with torch.no_grad():
                    lpv = lpips_model(p_t3, g_t3)
                lp[i] = float(lpv.cpu().numpy().ravel()[0])

        summary = {
            "n": int(N),
            "psnr_mean": float(np.nanmean(ps)),
            "psnr_std": float(np.nanstd(ps)),
            "ssim_mean": float(np.nanmean(ss)),
            "ssim_std": float(np.nanstd(ss)),
            "lpips_mean": float(np.nanmean(lp)),
            "lpips_std": float(np.nanstd(lp))
        }

        # save per-frame CSV
        csv_path = os.path.join(args.image_folder, f"metrics_{name}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            header = ["idx", "psnr", "ssim"]
            if lpips_model is not None:
                header.append("lpips")
            writer.writerow(header)
            for j in range(N):
                row = [j, float(ps[j]) if not np.isnan(ps[j]) else "", float(ss[j]) if not np.isnan(ss[j]) else "", float(lp[j]) if not np.isnan(lp[j]) else ""]
                writer.writerow(row)

        # save json summary
        with open(os.path.join(args.image_folder, f"summary_{name}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # print in requested format
        print(f"{name} -> PSNR {summary['psnr_mean']:.4f} _ {summary['psnr_std']:.4f}, SSIM {summary['ssim_mean']:.4f} _ {summary['ssim_std']:.4f} LPIPS {summary['lpips_mean']:.4f} _ {summary['lpips_std']:.4f}")

        return summary

    metrics = {}
    metrics['out_gs'] = compute_and_save("out_gs", out_stage1)

    # stage2 levels (sorted by idx)
    for idx in sorted(stage2_pre_dict.keys()):
        pre = stage2_pre_dict.get(idx, None)
        name_pre = f"stage2_sr{idx+1}_pre"
        metrics[name_pre] = compute_and_save(name_pre, pre)

    # write combined json
    with open(combined_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def compute_and_save_metrics(args, config, stage1_dir, stage2_dir=None, sigma_t=1.0):
    """
    Compute PSNR/SSIM using:
        - stage1_dir: folder with stage1 whole.npy and whole_gaussian_filter1d_sigma_t{sigma}.npy
        - stage2_dir: folder containing stage2_lvl{idx}_preGF.npy and stage2_lvl{idx}_postGF.npy
    Only stage2_lvl* files are considered (do NOT map stage2_final.npy into extra level).
    Results saved as metrics_{name}.csv and summary_{name}.json in args.image_folder and a combined json.
    """
    # avoid duplicate work: if combined summary already exists, load & return it
    combined_path = os.path.join(args.image_folder, "metrics_combined_summary.json")
    if os.path.exists(combined_path):
        try:
            with open(combined_path, "r") as f:
                metrics = json.load(f)
            print("Loaded existing metrics from", combined_path)
            return metrics
        except Exception:
            # fall through to recompute if file corrupted
            pass

    def load_stack(path):
        if path and os.path.exists(path):
            a = np.load(path)
            # accept (N,H,W) or (N,1,H,W)
            if a.ndim == 4 and a.shape[1] == 1:
                a = a[:, 0, :, :]
            return a.astype(np.float32)
        return None

    # load stage1 stacks
    stage1_whole = load_stack(os.path.join(stage1_dir, "whole.npy"))
    stage1_sm = load_stack(os.path.join(stage1_dir, f"whole_gaussian_filter1d_sigma_t{sigma_t}.npy"))

    # discover stage2 pre/post files under stage2_dir
    stage2_pre_dict = {}
    stage2_post_dict = {}
    if stage2_dir is not None and os.path.isdir(stage2_dir):
        # find files matching stage2_lvl{idx}_preGF.npy and _postGF.npy, and stage2_final.npy
        pre_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_preGF.npy")))
        post_files = sorted(glob.glob(os.path.join(stage2_dir, "stage2_lvl*_postGF.npy")))
        # map by level index extracted
        for p in pre_files:
            base = os.path.basename(p)
            # expect stage2_lvl{idx}_preGF.npy
            try:
                idx = int(base.split("stage2_lvl")[1].split("_")[0])
            except Exception:
                idx = len(stage2_pre_dict)
            stage2_pre_dict[idx] = np.load(p).astype(np.float32)
        for p in post_files:
            base = os.path.basename(p)
            try:
                idx = int(base.split("stage2_lvl")[1].split("_")[0])
            except Exception:
                idx = len(stage2_post_dict)
            stage2_post_dict[idx] = np.load(p).astype(np.float32)
        # also allow single final file
        final_p = os.path.join(stage2_dir, "stage2_final.npy")
        if os.path.exists(final_p):
            stage2_post_dict[max(stage2_post_dict.keys())+1 if len(stage2_post_dict)>0 else 0] = np.load(final_p).astype(np.float32)

    # helper to build GT stack from dataset
    def build_gt_stack():
        try:
            _, test_dataset = get_dataset_cascade(args, config)
            gt_list = []
            for item in test_dataset:
                gt = item["gt_img"] * 255.
                gt_list.append(gt)
            if len(gt_list) > 0:
                return np.stack(gt_list, axis=0)
        except Exception as e:
            print("compute_and_save_metrics: failed to build GT stack:", e)
        return None

    gt_stack = build_gt_stack()

    # normalize pred -> 0..255 for PSNR/SSIM computation if needed
    def prepare_pred(pred):
        p = pred.astype(np.float32)
        if np.nanmax(p) <= 1.01:
            p = p * 255.0
        return p

    def compute_metrics(name, pred_stack):
        if gt_stack is None:
            print(f"SKIP metrics {name}: GT not available")
            return None
        pred = prepare_pred(pred_stack)
        N = min(gt_stack.shape[0], pred.shape[0])
        ps = np.full(N, np.nan, dtype=np.float32)
        ss = np.full(N, np.nan, dtype=np.float32)
        for i in range(N):
            gt_i = gt_stack[i].astype(np.float32)
            pred_i = pred[i].astype(np.float32)
            try:
                ps[i] = psnr_fn(gt_i, pred_i, data_range=255.0)
            except Exception:
                ps[i] = np.nan
            try:
                ss[i] = ssim_fn(gt_i, pred_i, data_range=255.0)
            except Exception:
                ss[i] = np.nan
        summary = {
            "n": int(N),
            "psnr_mean": float(np.nanmean(ps)),
            "psnr_std": float(np.nanstd(ps)),
            "ssim_mean": float(np.nanmean(ss)),
            "ssim_std": float(np.nanstd(ss))
        }
        # save per-frame CSV
        csv_path = os.path.join(args.image_folder, f"metrics_{name}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "psnr", "ssim"])
            for j in range(N):
                writer.writerow([j, float(ps[j]) if not np.isnan(ps[j]) else "", float(ss[j]) if not np.isnan(ss[j]) else ""])
        # save json summary
        with open(os.path.join(args.image_folder, f"summary_{name}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"{name} -> PSNR {summary['psnr_mean']:.4f} _ {summary['psnr_std']:.4f}, SSIM {summary['ssim_mean']:.4f} _ {summary['ssim_std']:.4f}")
        return summary

    metrics = {}
    if stage1_whole is not None:
        metrics['stage1_whole'] = compute_metrics("stage1_whole", stage1_whole)
    if stage1_sm is not None:
        metrics['stage1_smoothed'] = compute_metrics("stage1_smoothed", stage1_sm)

    # handle stage2 levels in sorted order
    for idx in sorted(set(list(stage2_pre_dict.keys()) + list(stage2_post_dict.keys()))):
        pre = stage2_pre_dict.get(idx, None)
        post = stage2_post_dict.get(idx, None)
        name_pre = f"stage2_sr{idx+1}_pre"
        name_post = f"stage2_sr{idx+1}_post"
        # name as stage2_sr{level}_{pre/post} where level indexes are 1-based for readability
        if pre is not None:
            metrics[name_pre] = compute_metrics(name_pre, pre)
        if post is not None:
            metrics[name_post] = compute_metrics(name_post, post)

    # combined summary
    # with open(os.path.join(args.image_folder, "metrics_combined_summary.json"), "w") as f:
    with open(combined_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
