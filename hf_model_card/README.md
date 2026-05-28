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

Recommended Hugging Face model repository name:

`<hf-username-or-org>/ddnm-xray512-ct-projection-prior`

This repository is intended to host the diffusion checkpoint used by the DDNM projection super-resolution stage of:

> Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians.

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
  --hf-model-repo <hf-username-or-org>/ddnm-xray512-ct-projection-prior \
  --hf-model-file ema_0.9999_620000.pt \
  --ddnm-root /path/to/DDNM \
  --input-npy examples/mela_0050/mela_0050_projection_4x_128x128.npy \
  --gt-pickle /path/to/MELA_GT_512_rmbed/mela_0050_rmbed.pickle \
  --case-id mela_0050 \
  --scale 4
```

## Notes

The checkpoint is large and should be stored on Hugging Face rather than in the GitHub repository.
