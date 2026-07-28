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
import lpips
import torch
import argparse
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import tqdm
# CUDA_VISIBLE_DEVICES=0 python dataGenerator/generateData.py --mode test128_clamp --config up512
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default='./data/MELA_selected/test512_clamp/', type=str, help="Data Path")
    parser.add_argument("--savePath", default="./downsample/4x/", type=str, help="data save path")
    parser.add_argument("--lrVol", default=128, type=int, help="lr volume size")
    parser.add_argument("--nVoxel", default=512, type=int, help="hr volume size")
    parser.add_argument("--convert", default=False, type=bool, help="convert to attenuation")
    parser.add_argument("--reSlope", default=1.0, type=float, help="rescale slope")
    parser.add_argument("--reIntercept", default=0.0, type=float, help="rescale intercept")
    parser.add_argument("--normalize", default=True, type=bool, help="Normalization")
    parser.add_argument("--ctName", default="mela_0257", type=str, help="Name of CT")
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    pickle_list = os.listdir(args.dataPath)
    os.makedirs(args.savePath, exist_ok=True)
    for pickle_data in pickle_list:
        volName = pickle_data[:9]
        volPath = f"{args.dataPath}{volName}.nii.gz"
        outputPath = f"{args.savePath}/{volName}.nii.gz"
        print('Save Path: ', outputPath)
        gt, img = generator(args, volPath, outputPath)
    print('Finish')

def generator(args, volPath, outputPath):
    """
    Generate projections given CT image and configuration.
    """
    # Load CT image
    gt, lr = loadImage(volPath, args.lrVol, args.nVoxel, args.convert, args.reSlope, args.reIntercept, args.normalize)
    img = CTnoise.add(lr, Poisson=1e3, Gaussian=np.array([0, 0.]))
    sitk.WriteImage(sitk.GetImageFromArray(img), os.path.join(outputPath))
    print(f"Generated image saved to {outputPath}")
    return gt, img

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
    if normalize and image_min !=0 and image_max != 1:
        print("Normalize range to [0, 1]")
        gt_file = normalize_volume(gt_file)
    gt_vol = get_volume_from_file(gt_file)

    # Generate LR Volume
    lrVol = np.array((64, 64, 64)) if lrVol is None else np.array(lrVol)
    if np.any(nVoxels != lrVol):
        print(f"Resize GT ct image from {nVoxels}x{nVoxels}x{nVoxels} to "
              f"{lrVol}x{lrVol}x{lrVol}")
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

def downsample_volume(volume, scale_factor=2):
    scale_factor = [scale_factor, scale_factor, scale_factor]
    if isinstance(scale_factor, (int, float)):
        volume_dim = volume.ndim
        scale_factor = [scale_factor] * volume_dim
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_spacing = np.multiply(scale_factor, original_spacing)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    blurred_volume = sitk.SmoothingRecursiveGaussian(volume, 2.5) # sigma = 2.5
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(blurred_volume.GetOrigin())
    resampler.SetOutputDirection(blurred_volume.GetDirection())
    resampler.SetDefaultPixelValue(blurred_volume.GetPixelIDValue())
    downsampled_volume = resampler.Execute(blurred_volume)
    # noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
    # noise_filter.SetMean(0.0)
    # noise_filter.SetStandardDeviation(0.05)
    # downsampled_volume = noise_filter.Execute(downsampled_volume)
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