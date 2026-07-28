from __future__ import annotations

import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml

import pickle
import scipy.io
import SimpleITK as sitk
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise

import cv2
import matplotlib.pyplot as plt

import argparse

# CUDA_VISIBLE_DEVICES=0 python dataGenerator/generateData.py --mode test128_clamp --config up512

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test512_clamp", type=str,
                        help="Type of dataset")
    parser.add_argument("--ctName", default="mela_0257", type=str,
                        help="Name of CT")
    parser.add_argument("--config", default="gt64", type=str,
                        help="Name of config file")
    parser.add_argument("--outputName", default="chest_50", type=str,
                        help="Name of output data")
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    
    pickle_list = os.listdir('/workspace/data/MELA_selected/test512_clamp/')

    for pickle_data in pickle_list:

        volName = pickle_data[:9]
        mode = args.mode
        outputName = args.outputName
        configName = args.config
        
        ## add noise -> norm
        if args.config == "gt64":
            deg_scale = 8
        elif args.config == "gt128":
            deg_scale = 4

        volPath = f"/workspace/downsample/down_addNoise_{deg_scale}x/{volName}.nii.gz"
        configPath = f"/workspace/data_config/pickle/{configName}.yml"
        outputPath = f"/workspace/data/pickle_data/down_addNoise_{deg_scale}x_proj100/{volName}.pickle"

        ## norm -> add noise
        # volPath = f"/workspace/downsample/8x/{volName}.nii.gz"
        # configPath = f"/workspace/data_config/pickle/{configName}.yml"
        # outputPath = f"/workspace/data/pickle_data/downsample_8x_proj100_axis_fixed/{volName}.pickle"

        print(volPath)
        generator(volPath, configPath, outputPath, True)
    print('Finish')

# %% Geometry
class ConeGeometry_special(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] / 1000  # Distance Source Detector      (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"]) / 1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"][::-1])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data["offDetector"][1], data["offDetector"][0], 0]) / 1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


def convert_to_attenuation(data: np.array, rescale_slope: float, rescale_intercept: float):
    """
    CT scan is measured using Hounsfield units (HU). We need to convert it to attenuation.

    The HU is first computed with rescaling parameters:
        HU = slope * data + intercept

    Then HU is converted to attenuation:
        mu = mu_water + HU/1000x(mu_water-mu_air)
        mu_water = 0.206
        mu_air=0.0004

    Args:
    data (np.array(X, Y, Z)): CT data.
    rescale_slope (float): rescale slope.
    rescale_intercept (float): rescale intercept.

    Returns:
    mu (np.array(X, Y, Z)): attenuation map.

    """
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
    return mu

def get_volume_from_file(file):
    vol = sitk.GetArrayFromImage(file)
    vol = np.transpose(vol, (1, 2, 0))
    vol = vol.astype(np.float32)  # TIGRE requires float32
    return vol

def loadImage(dirname, lrVol, nVoxels, convert, rescale_slope, rescale_intercept, normalize=True):
    """
    Load CT image.
    """
    # Load Volume File
    gt_file = sitk.ReadImage(dirname)
    nVoxels = np.array(gt_file.GetSize()) if nVoxels is None else np.array(nVoxels)
    
    # Volume Information
    stats = sitk.StatisticsImageFilter()
    stats.Execute(gt_file)
    image_min = stats.GetMinimum()
    image_max = stats.GetMaximum()
    image_mean = stats.GetMean()
    gt_file = sitk.Clamp(gt_file, lowerBound=-512, upperBound=image_max)
    # gt_file = sitk.Clamp(gt_file, lowerBound=-500, upperBound=2000)
    
    print("Range of CT image is [%f, %f], mean: %f" % (image_min, image_max, image_mean))
    
    # Normalization
    # '''
    if normalize and image_min !=0 and image_max != 1:
        print("Normalize range to [0, 1]")
        gt_file = normalize_volume(gt_file)
    # '''
    
    # Generate array of GT volume
    if not np.all(np.array(gt_file.GetSize()) == nVoxels):
        print(f"Given volume is downsampled with scale {nVoxels/np.array(gt_file.GetSize())}")
        gt_file = downsample_volume(gt_file, np.array(gt_file.GetSize())/nVoxels)
    
    gt_vol = get_volume_from_file(gt_file)
    
    # Generate LR Volume
    lrVol = np.array((64, 64, 64)) if lrVol is None else np.array(lrVol)
    if np.any(nVoxels != lrVol):
        print(f"Resize GT ct image from {nVoxels[0]}x{nVoxels[1]}x{nVoxels[2]} to "
              f"{lrVol[0]}x{lrVol[1]}x{lrVol[2]}")
        lr_file = downsample_volume(gt_file, nVoxels/lrVol)
        lr_vol = get_volume_from_file(lr_file)
        print(lr_vol.shape)
    else:
        lr_vol = None
    
    # Convert HU into attenuation if needed
    if convert:
        print("Convert from HU to attenuation")
        gt_image = convert_to_attenuation(gt_vol, rescale_slope, rescale_intercept)
        lr_image = None if lr_vol is None else convert_to_attenuation(lr_vol, rescale_slope, rescale_intercept)
    else:
        gt_image = gt_vol
        lr_image = lr_vol

    return gt_image, lr_image


def generator(volPath, configPath, outputPath, show=False):
    """
    Generate projections given CT image and configuration.
    """
    # Load configuration
    with open(configPath, "r") as handle:
        data = yaml.safe_load(handle)

    # Load CT image
    geo = ConeGeometry_special(data)
    gt, lr = loadImage(volPath, data['lrVol'], data["nVoxel"], data["convert"],
                    data["rescale_slope"], data["rescale_intercept"], data["normalize"])
    data["image"] = gt.copy()
    
    if lr is None:
        print("Target shape is same with original volume.")
        img = gt
    else:
        scale = np.array(gt.shape) / np.array(lr.shape)    
        upsample = 'trilinear'
        print(f"Volume {lr.shape} is upsampled into {gt.shape} by {upsample}.")
        if upsample == 'trilinear':
            img = scipy.ndimage.zoom(lr, scale, order=1)
        elif upsample == 'cubic':
            img = scipy.ndimage.zoom(lr, scale, order=3, prefilter=False)
        elif upsample == 'prefilter':
            img = scipy.ndimage.zoom(lr, scale, order=3)
        else:
            print('No upsample method was selected')
            img = scipy.ndimage.zoom(lr, scale, order=0)
        data["upsampled"] = img.copy()
    
    data["numTrain"]= 100 # temporarily
    print("numTrain: ", data["numTrain"])
    # exit()

    # Generate training images
    if data["randomAngle"] is False:
        data["train"] = {"angles": np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numTrain"]+1)[:-1] + data["startAngle"]/ 180 * np.pi}
    else:
        data["train"] = {"angles": np.sort(np.random.rand(data["numTrain"]) * data["totalAngle"] / 180 * np.pi) + data["startAngle"]/ 180 * np.pi}
    projections = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), geo, data["train"]["angles"])[:, ::-1, :]
    if data["noise"] != 0 and data["normalize"]:
        print("Add noise to projections")
        noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, data["noise"]]))
        data["train"]["projections"] = noise_projections
    else:
        data["train"]["projections"] = projections
    
    print(len(data["train"]["angles"]))
    
    # Generate validation images
    '''
    data["val"] = {"angles": np.sort(np.random.rand(data["numVal"]) * 180 / 180 * np.pi) + data["startAngle"]/ 180 * np.pi}
    projections = tigre.Ax(np.transpose(img, (2, 1, 0))
                           .copy(), geo, data["val"]["angles"])[:, ::-1, :]
    if data["noise"] != 0 and data["normalize"]:
        print("Add noise to projections")
        noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, data["noise"]]))
        data["val"]["projections"] = noise_projections
    else:
        data["val"]["projections"] = projections

    if show:
        print("Display ct image")
        tigre.plotimg(img.transpose((2,0,1)), dim="z")
        print("Display training images")
        tigre.plotproj(data["train"]["projections"][:, ::-1, :])
        print("Display validation images")
        tigre.plotproj(data["val"]["projections"][:, ::-1, :])
    '''

    # Save data
    os.makedirs(osp.dirname(outputPath), exist_ok=True)
    with open(outputPath, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    print(f"Save files in {outputPath}")

def downsample_volume(volume, scale_factor=2):
    print('--------------------------')
    print(scale_factor)
    if isinstance(scale_factor, (int, float)):
        print('--------------------------')
        volume_dim = volume.ndim
        scale_factor = [scale_factor] * volume_dim
    
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()

    new_spacing = np.multiply(scale_factor, original_spacing)

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(volume.GetOrigin())
    resampler.SetOutputDirection(volume.GetDirection())
    resampler.SetDefaultPixelValue(volume.GetPixelIDValue())

    downsampled_volume = resampler.Execute(volume)

    return downsampled_volume

def normalize_volume(volume):
    stats = sitk.StatisticsImageFilter()
    stats.Execute(volume)
    min_value = stats.GetMinimum()
    max_value = stats.GetMaximum()

    normalized_volume = sitk.Cast((volume - min_value) / (max_value - min_value), sitk.sitkFloat32)

    return normalized_volume

if __name__ == "__main__":
    main()
