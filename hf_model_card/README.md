---
license: apache-2.0
library_name: pytorch
tags:
  - diffusion
  - ddnm
  - ct
  - x-ray
  - super-resolution
---

# DDNM X-ray512 CT Projection Prior

Hugging Face model repository:

`Hyun-Jic/ddnm-xray512-ct-projection-prior`

This repository is intended to host the diffusion checkpoint used by the DDNM projection super-resolution stage of:

> Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians.

## Model Description

This checkpoint is a 512x512 2D diffusion prior trained on chest X-ray domain images and used as the image prior inside DDNM for CT projection super-resolution. It is not a standalone CT volume reconstruction model. In the full pipeline, a low-resolution CT volume is first converted into 2D projection images; DDNM then uses this diffusion prior to enhance each projection; the enhanced projections are subsequently used by the 3D reconstruction stage.

## Intended Use

- 2D CT projection super-resolution through DDNM.
- Zero-shot 3D CT super-resolution pipelines where projection-domain priors are used before volume reconstruction.
- Research use and reproduction of the projection-prior stage.

## Training Data

The diffusion prior was trained on chest X-ray domain images from:

- CheX-ray14
- CheXpert

The model is used as a natural/medical X-ray image prior for projection-domain restoration. It was not trained directly on the target CT volumes used for downstream evaluation.

## Architecture and Sampling Settings

The checkpoint follows an improved-diffusion / guided-diffusion style UNet configuration:

```yaml
image_size: 512
in_channels: 3
out_channels: 3
num_channels: 256
num_res_blocks: 2
attention_resolutions: "32,16,8"
num_heads: 4
num_head_channels: 64
dropout: 0.0
learn_sigma: false
use_scale_shift_norm: true
use_fp16: true
resblock_updown: true
beta_schedule: linear
beta_start: 0.0001
beta_end: 0.02
num_diffusion_timesteps: 1000
```

DDNM projection SR settings used in the release wrapper:

| Scale | degradation | eta | sigma_y | sampling steps |
|---|---|---:|---:|---:|
| 4x | sr_averagepooling | 0.990 | 0.0010 | 50 |
| 8x | sr_averagepooling | 0.990 | 0.0025 | 50 |

## Files

Upload the DDNM/SIDE checkpoint as:

```text
ema_0.9999_620000.pt
```

The GitHub wrapper expects this file by default.

## Usage

```bash
pip install huggingface_hub

python ddnm_inference/run_ddnm_projection_sr.py \
  --hf-model-repo Hyun-Jic/ddnm-xray512-ct-projection-prior \
  --hf-model-file ema_0.9999_620000.pt \
  --ddnm-root /path/to/DDNM \
  --input-npy examples/mela_0050/mela_0050_projection_4x_128x128.npy \
  --gt-pickle /path/to/MELA_GT_512_rmbed/mela_0050_rmbed.pickle \
  --case-id mela_0050 \
  --scale 4
```

## Limitations

- The model is a 2D projection prior, not a complete 3D CT reconstructor.
- Output quality depends on the DDNM degradation operator, projection normalization, and downstream reconstruction.
- Clinical use is not intended without additional validation.

## Notes

The checkpoint is large and should be stored on Hugging Face rather than in the GitHub repository.
