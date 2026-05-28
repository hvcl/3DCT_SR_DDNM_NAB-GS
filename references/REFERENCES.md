# References and Code Acknowledgements

This repository includes release wrappers around the DDNM projection-prior stage and the NAB-GS reconstruction stage. The following open-source projects were important references.

## DDNM

- Paper: *Denoising Diffusion Null-Space Model for Inverse Problems*
- Code: https://github.com/wyhuai/DDNM

We use the DDNM inverse-problem formulation for 2D CT projection super-resolution.

## Improved Diffusion / Guided Diffusion

- Improved Denoising Diffusion Probabilistic Models: https://github.com/openai/improved-diffusion
- Guided Diffusion: https://github.com/openai/guided-diffusion

The X-ray projection prior was trained using an improved-diffusion style 512x512 diffusion model.

## 3D Gaussian Splatting

- Code: https://github.com/graphdeco-inria/gaussian-splatting

NAB-GS builds on the general 3D Gaussian representation idea and adapts it to signed residual CT reconstruction.

## TIGRE

- Code: https://github.com/CERN/TIGRE

TIGRE is used as a CT geometry/projection reference in related preprocessing and reconstruction utilities.

## Project

If you use this code, please cite:

```bibtex
@article{noh2025zero,
  title={Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians},
  author={Noh, Jeonghyun and Oh, Hyun-Jic and Jeong, Won-Ki},
  journal={arXiv preprint arXiv:2508.15151},
  year={2025}
}
```
