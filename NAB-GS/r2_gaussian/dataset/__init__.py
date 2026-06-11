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
import os
import sys
import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

sys.path.append("./")
from r2_gaussian.gaussian import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from r2_gaussian.utils.camera_utils import cameraList_from_camInfos
from r2_gaussian.utils.general_utils import t2a
from scipy.ndimage import map_coordinates

class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.vol_lr = scene_info.vol_lr
        self.vol_up = scene_info.vol_up
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc, res_error_map=False):
        # offOrigin = torch.tensor(self.scanner_cfg["offOrigin"]).cuda()
        # dVoxel = torch.tensor(self.scanner_cfg["dVoxel"]).cuda()
        # sVoxel = torch.tensor(self.scanner_cfg["sVoxel"]).cuda()

        # lr_vol = F.relu(self.vol_lr)

        # vol_indicies = (self.gaussians._xyz - offOrigin + sVoxel/2) / dVoxel
        # lr_vol_ = lr_vol.unsqueeze(0).unsqueeze(0)
        # D, H, W = lr_vol.shape
        # vol_indicies[:, 0] = vol_indicies[:, 0] / (D - 1) * 2 - 1  # x → W
        # vol_indicies[:, 1] = vol_indicies[:, 1] / (H - 1) * 2 - 1  # y → H
        # vol_indicies[:, 2] = vol_indicies[:, 2] / (W - 1) * 2 - 1  # z → D
        # grid = vol_indicies[:, [0, 1, 2]].view(1, -1, 1, 1, 3)

        # # Interpolate
        # lr_vol_density = F.grid_sample(lr_vol_, grid, mode='bilinear', align_corners=True)
        # lr_vol_density = lr_vol_density.view(-1, 1)  # (N,)             
        # self.gaussians._density = self.gaussians._density + (lr_vol_density * 0.15)
        
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            if res_error_map is False:
                vol_pred = queryfunc(self.gaussians)["vol"]
                # np.save(
                #     osp.join(point_cloud_path, "vol_pred_only.npy"),
                #     t2a(vol_pred),
                # )
                vol_gt = self.vol_gt
                vol_up = self.vol_up
                vol_up = F.relu(vol_up)
                vol_pred = vol_pred + vol_up
                vol_pred = F.relu(vol_pred)
                np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
                np.save(
                    osp.join(point_cloud_path, "vol_pred.npy"),
                    t2a(vol_pred),
                )
            else:
                vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                t2a(vol_pred),
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
