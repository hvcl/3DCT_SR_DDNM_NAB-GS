import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--ckpt", type=str, default="SIDE", help="Path to the ckpt file"
        # SIDE | 
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--path_y",
        type=str,
        required=True,
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--degraded_path",
        type=str,
        default="",
        help="Optional override pickle path for degraded input projections.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="",
        help="Optional override pickle path for GT projections.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--deg_scale", type=float, default=0., help="deg_scale"
    )    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=0
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--save_ddnm_formula",
        action="store_true",
        help="Save intermediate tensors used in DDNM equations."
    )
    parser.add_argument(
        "--formula_proj_idx",
        type=int,
        default=0,
        help="Projection index to save DDNM-formula intermediates for."
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        type=str,
        default="gaussian",
        help="gaussian | 3d_gaussian | poisson | speckle"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )
    parser.add_argument(
        "--noise_warp",
        action="store_true"
    )
    
    # New MICCAI26 arguments
    parser.add_argument(
        "--setup",
        type=str,
        default="ddnm_orig",
        choices=["ddnm_orig", "ddnm_pas", "ddnm_fixedSteps"],
    )
    parser.add_argument(
        "--startStep", 
        type=int, 
        default=-1,
        help="Fixed start step to skip l2thr detection. If >= 0, skip this many steps from the beginning"
        # when setup is ddnm_fixedSteps
    )
    parser.add_argument(
        "--l2thr", 
        type=float, 
        default=0,
        help="l2 threshold for PAS start step selection (baseline for adaptive threshold)"
        # when setup is ddnm_pas
    )
    parser.add_argument(
        "--l2thr_min",
        type=float,
        default=-1,
        help="Minimum l2 threshold for adaptive threshold across projections. If -1, disable adaptive threshold."
    )
    parser.add_argument(
        "--l2thr_max",
        type=float,
        default=-1,
        help="Maximum l2 threshold for adaptive threshold across projections."
    )
    parser.add_argument(
        "--minimum_PAS_startStep",
        type=int,
        default=-1,
        help="Minimum start step for PAS to ensure at least N steps are executed. "
             "If threshold fails, use this as lower bound. Default -1 (no minimum)."
    )
    parser.add_argument(
        "--cgls_path",
        type=str,
        default=None,
        help="Path to whole_cgls.npy file [N, 512, 512] with pixel values in [0, 1] range"
    )
    parser.add_argument(
        "--l2_monitor",
        action="store_true",
        help="Monitor and save l2 threshold plots during start step selection"
    )
    parser.add_argument(
        "--ddnm_step_before",
        type=int,
        default=-1,
        help="If set to k (0-49), disable measurement consistency for steps >= k (use pure diffusion output for last steps). Default -1: off."
    )
    parser.add_argument(
        "--shared_noise",
        action="store_true",
        help="If set, use the same random noise for all projections (reproducible per-projection denormalization)."
    )
    parser.add_argument(
        "--noise_control",
        type=str,
        default="none",
        choices=["none", "shared", "warped", "slerp"],
        help="Noise control variant for projection sequence. 'shared' and 'warped' map to existing modes; 'slerp' uses ISCS-style spherical interpolation between endpoint noises.",
    )
    parser.add_argument(
        "--warped_noise",
        action="store_true",
        help="If set with --shared_noise, apply optical flow-based noise warping between consecutive projections."
    )
    parser.add_argument(
        "--slerp_endpoint_seed_0",
        type=int,
        default=1234,
        help="Seed for the first SLERP endpoint noise.",
    )
    parser.add_argument(
        "--slerp_endpoint_seed_1",
        type=int,
        default=5678,
        help="Seed for the second SLERP endpoint noise.",
    )

    parser.add_argument(
        "--clip_max",
        type=float,
        default=1.0,
        help="Maximum value used when un-normalizing and saving images (default=1.0)."
    )

    args = parser.parse_args()

    if args.cgls_path is not None:
        args.deg_scale = args.deg_scale / 2

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # Add file logging to output directory
    try:
        file_handler = logging.FileHandler(os.path.join(args.image_folder, "run.log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {os.path.join(args.image_folder, 'run.log')}")
    except Exception as e:
        logger.error(f"Failed to attach file logger: {e}")
            
    config_save = config
    config_save['args'] = {}
    for arg in vars(args):
        config_save['args'][arg] = getattr(args, arg)
    
    with open(os.path.join(args.image_folder, "configs.yml"), "w") as f:
        yaml.dump(config_save, f, default_flow_style=False, indent=4)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    from guided_diffusion.diffusion_miccai26 import Diffusion
    try:
        runner = Diffusion(args, config)
        runner.sample(args.simplified)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
