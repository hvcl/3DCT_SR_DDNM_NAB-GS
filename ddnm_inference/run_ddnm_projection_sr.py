#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_HF_REPO = "YOUR_HF_USERNAME/ddnm-xray512-ct-projection-prior"
DEFAULT_MODEL_FILE = "ema_0.9999_620000.pt"
SIDE_RELATIVE_PATH = Path("exp/logs/CheX-ray14_CheXpert_512x512-2gb/ema_0.9999_620000.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DDNM projection SR for one CT projection stack stored as .npy."
    )
    parser.add_argument("--ddnm-root", type=Path, required=True, help="Path to the local DDNM repository.")
    parser.add_argument("--input-npy", type=Path, required=True, help="LR projection stack, shape [N,H,W].")
    parser.add_argument("--gt-pickle", type=Path, required=True, help="HR GT pickle used by DDNM for shape/range/evaluation.")
    parser.add_argument("--case-id", default="mela_0050", help="Case id passed to DDNM --path_y.")
    parser.add_argument("--scale", type=int, choices=[4, 8], required=True, help="Projection SR scale.")
    parser.add_argument("--output-name", default="", help="Output folder under <ddnm-root>/exp/image_samples.")
    parser.add_argument("--setup", choices=["ddnm_orig"], default="ddnm_orig", help="Paper setting. Other internal ablations are not exposed in the release wrapper.")
    parser.add_argument("--eta", type=float, default=0.990)
    parser.add_argument("--sigma-y", type=float, default=None, help="Defaults: 4x=0.001, 8x=0.0025.")
    parser.add_argument("--clip-max", type=float, default=1.05)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--subset-start", "--subset_start", dest="subset_start", type=int, default=0)
    parser.add_argument("--subset-end", "--subset_end", dest="subset_end", type=int, default=-1, help="Use >0 for quick tests, e.g. 1.")
    parser.add_argument("--work-dir", type=Path, default=Path("ddnm_work"))
    parser.add_argument("--model-checkpoint", type=Path, default=None, help="Local diffusion checkpoint. Overrides HF download.")
    parser.add_argument("--hf-model-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--workspace-ddnm", type=Path, default=Path(".ddnm_workspace/DDNM"))
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without launching DDNM.")
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def write_pickle(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_degraded_pickle(input_npy: Path, gt_pickle: Path, out_pickle: Path) -> None:
    arr = np.load(input_npy).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"--input-npy must have shape [N,H,W], got {arr.shape}")
    gt = load_pickle(gt_pickle)
    payload = dict(gt)
    payload["image"] = arr
    payload["train"] = dict(gt.get("train", {}))
    payload["train"]["projections"] = arr
    payload["numTrain"] = int(arr.shape[0])
    payload["val"] = dict(gt.get("val", {}))
    write_pickle(out_pickle, payload)


def hf_download(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the model. "
            "Install it or pass --model-checkpoint."
        ) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def ensure_side_checkpoint(ddnm_root: Path, workspace_ddnm: Path, model_checkpoint: Path) -> Path:
    expected = workspace_ddnm / SIDE_RELATIVE_PATH
    if expected.exists():
        return expected

    if not workspace_ddnm.exists():
        workspace_ddnm.parent.mkdir(parents=True, exist_ok=True)
        try:
            workspace_ddnm.symlink_to(ddnm_root.resolve(), target_is_directory=True)
        except FileExistsError:
            pass

    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        expected.unlink()
    expected.symlink_to(model_checkpoint.resolve())
    return expected


def main() -> int:
    args = parse_args()
    ddnm_root = args.ddnm_root.resolve()
    if not (ddnm_root / "main_miccai26.py").exists():
        raise FileNotFoundError(f"DDNM root does not contain main_miccai26.py: {ddnm_root}")
    if not args.input_npy.exists():
        raise FileNotFoundError(args.input_npy)
    if not args.gt_pickle.exists():
        raise FileNotFoundError(args.gt_pickle)

    sigma_y = args.sigma_y if args.sigma_y is not None else (0.001 if args.scale == 4 else 0.0025)
    output_name = args.output_name or f"{args.case_id}_ddnm_x{args.scale}_{args.setup}"

    work_dir = args.work_dir.resolve()
    degraded_pickle = work_dir / "inputs" / f"{args.case_id}_x{args.scale}_degraded_from_npy.pickle"
    make_degraded_pickle(args.input_npy.resolve(), args.gt_pickle.resolve(), degraded_pickle)

    side_ckpt: Path | str
    model_checkpoint = args.model_checkpoint
    if args.dry_run and model_checkpoint is None:
        side_ckpt = f"DRY_RUN: would download {args.hf_model_repo}/{args.hf_model_file}"
    elif model_checkpoint is None:
        model_checkpoint = hf_download(args.hf_model_repo, args.hf_model_file)
        model_checkpoint = model_checkpoint.resolve()
        if not model_checkpoint.exists():
            raise FileNotFoundError(model_checkpoint)
        side_ckpt = ensure_side_checkpoint(ddnm_root, args.workspace_ddnm, model_checkpoint)
    else:
        model_checkpoint = model_checkpoint.resolve()
        if not model_checkpoint.exists():
            raise FileNotFoundError(model_checkpoint)
        side_ckpt = ensure_side_checkpoint(ddnm_root, args.workspace_ddnm, model_checkpoint)

    cmd = [
        sys.executable,
        "main_miccai26.py",
        "--ni",
        "--config",
        "MELA.yml",
        "--path_y",
        args.case_id,
        "--degraded_path",
        str(degraded_pickle),
        "--gt_path",
        str(args.gt_pickle.resolve()),
        "--eta",
        str(args.eta),
        "--deg",
        "sr_averagepooling",
        "--deg_scale",
        str(float(args.scale)),
        "--sigma_y",
        str(sigma_y),
        "--setup",
        args.setup,
        "--clip_max",
        str(args.clip_max),
        "-i",
        output_name,
        "--ckpt",
        "SIDE",
    ]
    if args.subset_start >= 0:
        cmd += ["--subset_start", str(args.subset_start)]
    if args.subset_end > 0:
        cmd += ["--subset_end", str(args.subset_end)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = str(ddnm_root) + os.pathsep + env.get("PYTHONPATH", "")

    print("Prepared DDNM degraded pickle:", degraded_pickle)
    print("Resolved SIDE checkpoint:", side_ckpt)
    print("Output directory:", ddnm_root / "exp/image_samples" / output_name)
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return 0

    return subprocess.call(cmd, cwd=ddnm_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
