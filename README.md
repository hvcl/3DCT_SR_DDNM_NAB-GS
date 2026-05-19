# Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians

<div align="center">

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026%20Early%20Accept-brightgreen?style=for-the-badge)](https://conferences.miccai.org/2026/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.15151-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2508.15151)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-blue?style=for-the-badge&logo=github)](https://github.com)
[![Korea University](https://img.shields.io/badge/Korea%20University-HVCL-8B0000?style=for-the-badge)](https://hvcl.korea.ac.kr)

**Jeonghyun Noh\* · Hyun-Jic Oh\* · Won-Ki Jeong†**

*Korea University &nbsp;|&nbsp; \* Equal contribution &nbsp;|&nbsp; † Corresponding author*

`{wjdgus0967, hyunjic0127, wkjeong}@korea.ac.kr`

</div>

---

## TL;DR

> We propose a **zero-shot 3D CT super-resolution** framework that boosts spatial resolution without any paired HR-LR training data. By leveraging a diffusion model trained on abundant 2D X-ray images to super-resolve CT projections, and then reconstructing a high-resolution 3D volume via a novel **Negative Alpha Blending 3D Gaussian Splatting (NAB-GS)**, our method achieves state-of-the-art performance at **4× upscaling** and demonstrates strong clinical potential.

---

## Overview

<div align="center">
  <img src="main_figure.png" alt="Method Overview" width="100%"/>
  <br>
  <em><b>Figure 1.</b> Overview of the proposed framework. (a) LR CT projections are super-resolved using a diffusion model (DDNM) trained on 2D X-ray data. (b) The resulting HR projections guide 3D CT reconstruction via NAB-GS, which jointly learns positive and negative residual Gaussian densities to reconstruct a high-fidelity HR volume.</em>
</div>

---

## Abstract

Computed tomography (CT) is important in clinical diagnosis, but acquiring high-resolution (HR) CT is constrained by radiation exposure risks. While deep learning-based super-resolution (SR) methods have shown promise for reconstructing HR CT from low-resolution (LR) inputs, **supervised approaches require paired datasets that are often unavailable**. Zero-shot methods address this limitation by operating on single LR inputs; however, they frequently fail to recover fine structural details due to limited LR information within individual volumes.

To overcome these limitations, we propose a novel **zero-shot 3D CT SR framework** that integrates diffusion-based upsampled 2D projection priors into the 3D reconstruction process. Our framework consists of two stages:

1. **LR CT Projection SR**: A diffusion model trained on abundant 2D X-ray data upsamples LR CT projections via DDNM, enriching the scarce high-frequency information in LR inputs.
2. **3D CT Reconstruction via NAB-GS**: Our novel Negative Alpha Blending 3D Gaussian Splatting reconstructs the HR volume by modeling both positive and negative Gaussian densities to learn the signed residual field between diffusion-generated HR projections and upsampled LR projections.

---

## Method

### Stage 1 — LR Projection SR using Diffusion Model

<div align="center">
  <img src="main_figure.png" alt="Stage 1" width="80%"/>
  <br>
  <em>Stage 1: A 2D diffusion model trained on X-ray data super-resolves LR CT projections via the DDNM (Denoising Diffusion Null-space Model) framework.</em>
</div>

- Given an LR CT volume, **multi-view LR projections** are generated via volume projection.
- A **diffusion model** pre-trained on large-scale 2D X-ray datasets is applied to each projection.
- The DDNM process iteratively refines the noisy estimate $\mathbf{x}_t$, enforcing data consistency through the null-space decomposition:

$$\hat{\mathbf{x}}_{0|t} = (\mathbf{I} - \mathbf{A}^\dagger \mathbf{A}) \mathbf{x}_{0|t} + \mathbf{A}^\dagger \mathbf{y}$$

where $\mathbf{y}$ is the LR projection (measurement), $\mathbf{A}$ is the degradation operator, and $\mathbf{A}^\dagger$ is its pseudo-inverse.

- The resulting **HR projections** serve as 2D priors for the subsequent 3D reconstruction stage.

---

### Stage 2 — 3D CT Reconstruction via NAB-GS

The core idea is to learn a **signed residual field** between the diffusion-generated HR projections and the projections of the upsampled LR volume, using 3D Gaussian Splatting extended with a novel **Negative Alpha Blending** mechanism.

#### Key Components

| Component | Description |
|---|---|
| **CT Space Radiative Gaussian** | 3D Gaussians initialized in CT space and transformed to world space for X-ray rendering |
| **Negative Alpha Blending (NAB)** | Relaxes the non-negativity constraint of standard 3DGS, allowing Gaussians to carry negative density for signed residual learning |
| **Adaptive Control** | Dynamically adjusts Gaussian density and coverage during optimization |
| **X-ray Rasterizer** | Renders residual projections from Gaussians; combined with upsampled LR projections for image residual learning ($\mathcal{L}_\text{recon}$) |
| **Voxelizer** | Converts Gaussian representation to voxel grid for volume residual learning ($\mathcal{L}_\text{tv}$) |

#### Positive & Negative Gaussians

Standard 3D Gaussian Splatting enforces **non-negative opacity**. In CT SR, the residual between HR and LR can be **positive or negative** depending on local structure. NAB-GS explicitly models:

- 🔵 **Positive density Gaussians** — regions where the HR volume has higher attenuation than the upsampled LR
- 🔴 **Negative density Gaussians** — regions where LR over-estimates attenuation

#### HR Volume Generation

$$\text{Output HR Volume} = \text{Clipping}(\text{Max}(0, x)) \Big( \text{Learned Residual Field} + \text{Upsampled LR Volume} \Big)$$

---

## Results

### Quantitative Comparison

Results on two public CT datasets at **4× super-resolution**:

| Method | Type | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| Trilinear Interpolation | Traditional | — | — | — |
| ESRGAN | Supervised | — | — | — |
| SelfSR | Zero-shot | — | — | — |
| NeRF-based SR | Zero-shot | — | — | — |
| **Ours (NAB-GS)** | **Zero-shot** | **Best** | **Best** | **Best** |

> *Full quantitative results are available in the paper. Our method achieves **superior quantitative and qualitative performance** on both datasets.*

### Qualitative Results

Our framework recovers fine structural details (e.g., bone trabeculae, vessel boundaries) that zero-shot baselines fail to reconstruct, while preserving anatomical plausibility validated by clinical expert evaluation.

### Clinical Expert Evaluation

Expert radiologist evaluations confirm the clinical potential of our framework at **4× upscaling**, demonstrating:
- Improved delineation of fine anatomical structures
- Reduction of aliasing artifacts common in LR reconstructions
- Diagnostic-quality outputs without paired training data

---

## Getting Started

> **Code is coming soon!** We are preparing a clean, reproducible release. Star ⭐ this repository to get notified.

### Requirements *(Planned)*

```bash
# Environment setup
conda create -n nabgs python=3.10
conda activate nabgs

# Install dependencies
pip install -r requirements.txt
```

### Pretrained Models *(Planned)*

| Model | Description | Link |
|---|---|---|
| Diffusion Model (X-ray) | 2D projection SR prior | TBD |
| NAB-GS | 3D reconstruction module | TBD |

### Inference *(Planned)*

```bash
# Stage 1: LR projection SR
python stage1_projection_sr.py --input_volume /path/to/lr_ct.nii.gz --output_dir ./hr_projections

# Stage 2: 3D reconstruction via NAB-GS
python stage2_nabgs_recon.py --projections ./hr_projections --output /path/to/hr_ct.nii.gz
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{noh2026zeroshot,
  title     = {Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians},
  author    = {Noh, Jeonghyun and Oh, Hyun-Jic and Jeong, Won-Ki},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```

---

## Acknowledgements

This work was conducted at the **High-performance Visual Computing Lab (HVCL), Korea University**.

We gratefully acknowledge the public CT datasets used for evaluation, and the authors of DDNM and 3D Gaussian Splatting for their foundational contributions.

---

<div align="center">
  <sub>© 2026 Jeonghyun Noh, Hyun-Jic Oh, Won-Ki Jeong | Korea University</sub>
  <br>
  <sub>⭐ Star this repo to stay updated on the code release!</sub>
</div>
