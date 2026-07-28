import math
import torch
import torch.nn.functional as F
import numpy as np
from functions.svd_operators import A_functions

class HighFreqDegradation(A_functions):
    """
    High-frequency measurement operator for stage2.

    - A(x): returns flattened HF component = inverse-FFT( FFT(x) * mask ) flattened
    - A_pinv(y): for our simple operator we treat it as identity (returns image-shaped HF)
    - Lambda(...) and Lambda_noise(...) implement the simple behaviour expected by ddnm_plus:
        Lambda(residual_flat, ...) -> returns flattened correction to subtract from x0_t
        Lambda_noise(noise_flat, ...) -> maps random noise to appropriate shaped noise term

    Args:
        channels: number of channels (1 or 3)
        img_dim: spatial image size (H==W)
        cutoff_radius: float (0..1 or absolute pixels). If <=1 treat as fraction of half-size.
        device: torch.device or device string
        img_folder: optional (keeps backward compatibility)
    """
    def __init__(self, channels, img_dim, cutoff_radius, device, img_folder=None, hf_boost=1.0, min_gain=0.5, max_gain=1.2):
        super().__init__()
        self.channels = int(channels)
        self.img_dim = int(img_dim)
        self.device = torch.device(device)
        # HF gain control
        self.hf_boost = float(hf_boost)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        # allow cutoff_radius either fraction (<=1) or pixel radius (>1)
        if cutoff_radius <= 1.0:
            self.cutoff_px = float(cutoff_radius) * (self.img_dim // 2)
        else:
            self.cutoff_px = float(cutoff_radius)
        # build real-valued radial mask in frequency domain: 1 inside highfreq annulus, 0 elsewhere
        cy, cx = self.img_dim // 2, self.img_dim // 2
        y, x = np.ogrid[:self.img_dim, :self.img_dim]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # highfreq mask: 1 where dist >= cutoff_px (keep outer high freqs),
        # small epsilon to avoid strictly zero region -> allow tiny leakage
        mask_np = (dist >= (self.cutoff_px)).astype(np.float32)
        # smooth mask edges slightly with a Gaussian to avoid ringing
        edge_width = max(1.0, self.cutoff_px * 0.05)
        if edge_width > 0:
            mask_np = gaussian_smooth_mask(mask_np, sigma=edge_width)
        mask = torch.from_numpy(mask_np).to(self.device).float()
        mask = 1 - mask
        # store as shape (1,1,H,W) for broadcasting to [B,C,H,W] complex spectra
        self.mask = mask.unsqueeze(0).unsqueeze(0)

    def _fft2(self, x):
        # x: real tensor [B,C,H,W]
        # return complex tensor of same shape
        X = torch.fft.fft2(x, norm='ortho')
        X = torch.fft.fftshift(X, dim=(-2, -1))
        return X

    def _ifft2(self, X):
        X = torch.fft.ifftshift(X, dim=(-2, -1))
        x = torch.fft.ifft2(X, norm='ortho')
        return x.real

    # Compatibility/placeholder methods expected by svd_operators.A_functions
    def V(self, vec): return vec
    def Vt(self, vec): return vec
    def U(self, vec): return vec
    def Ut(self, vec): return vec
    def singulars(self):
        return torch.ones(self.channels * self.img_dim * self.img_dim, device=self.device)
    def add_zeros(self, vec): return vec

    def A(self, x):
        """
        x: tensor [B, C, H, W] or flattened [B, C*H*W]
        returns flattened HF image [B, C*H*W] (real)
        Applies auto-scaling to HF component to preserve contrast.
        """
        if x.dim() == 2:
            # already flattened vector
            return x
        if x.dim() != 4:
            raise ValueError("HighFreqDegradation.A expects x [B,C,H,W] or flattened [B,N].")
        # ensure mask on same device/dtype
        mask = self.mask.to(x.device).to(x.dtype)
        X = self._fft2(x)  # complex
        # broadcast mask to channels
        mask_b = mask.repeat(1, self.channels, 1, 1)
        X_hf = X * mask_b
        x_hf = self._ifft2(X_hf)  # real, shape [B,C,H,W]

        # Automatic HF gain scaling to avoid washed-out contrast:
        # compute per-sample RMS/std for original and HF; compute gain = (orig_std / hf_std) * hf_boost
        # clamp gain to [min_gain, max_gain]
        try:
            eps = 1e-12
            # compute std over channels+spatial dims -> shape (B,)
            x_flat = x.view(x.shape[0], -1)
            hf_flat = x_hf.view(x_hf.shape[0], -1)
            x_std = torch.std(x_flat, dim=1, unbiased=False).clamp(min=eps)  # (B,)
            hf_std = torch.std(hf_flat, dim=1, unbiased=False).clamp(min=eps)  # (B,)
            gain = (x_std / hf_std) * float(self.hf_boost)  # (B,)
            # clamp
            gain = torch.clamp(gain, self.min_gain, self.max_gain)  # (B,)
            # reshape to broadcast to [B,C,H,W]
            gain = gain.view(-1, 1, 1, 1).to(x_hf.device).to(x_hf.dtype)
            x_hf = x_hf * gain
        except Exception:
            # if anything fails, fall back to no scaling
            pass

        B, C, H, W = x_hf.shape
        return x_hf.reshape(B, -1)

    def A_pinv(self, y):
        """
        Return image-shaped HF in the same numeric domain as input (no remapping).
        If `y` is flattened [B, C*H*W], reshape to [B,C,H,W].
        Ensures zero-mean per sample (HF should be zero-mean) but preserves input scale.
        """
        if y.dim() == 4:
            # already image-shaped
            B, C, H, W = y.shape
            # ensure shape matches expected
            if C != self.channels or H != self.img_dim or W != self.img_dim:
                # try to broadcast / adapt if possible (not ideal)
                raise ValueError(f"A_pinv: unexpected image shape {y.shape}, expected (B,{self.channels},{self.img_dim},{self.img_dim})")
            y_img = y
        elif y.dim() == 2:
            B = y.size(0)
            expected = self.channels * self.img_dim * self.img_dim
            if y.size(1) != expected:
                raise ValueError(f"A_pinv: flattened size {y.size(1)} != expected {expected}")
            y_img = y.view(B, self.channels, self.img_dim, self.img_dim)
        else:
            raise ValueError("HighFreqDegradation.A_pinv expects dim 2 or 4")

        # enforce zero-mean HF per-sample (remove DC) but preserve scale
        mean_per_sample = y_img.view(y_img.shape[0], -1).mean(dim=1).view(-1,1,1,1)
        y_img = y_img - mean_per_sample.to(y_img.device)
        return y_img  # image-shaped [B,C,H,W]

    # alias for At (some code calls At)
    def At(self, y):
        return self.A_pinv(y)

    def Lambda(self, residual_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None):
        """
        Map residual_flat (flattened or image-shaped) -> image-shaped correction.
        Return same numeric domain as inputs, shape [B,C,H,W].
        """
        if residual_flat is None:
            raise ValueError("Lambda: residual_flat is None")
        if isinstance(residual_flat, torch.Tensor):
            if residual_flat.dim() == 2:
                r_img = self.A_pinv(residual_flat)
            elif residual_flat.dim() == 4:
                r_img = residual_flat
            else:
                raise ValueError("Lambda: unexpected residual dim")
        else:
            raise TypeError("Lambda: residual_flat must be torch.Tensor")

        # scaling heuristic: attenuate if sigma_y large
        scale = 1.0 / (1.0 + float(sigma_y) * 100.0)
        r_img = r_img * scale
        return r_img  # image-shaped

    def Lambda_noise(self, noise_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None, et_flat=None):
        """
        Map random noise -> image-shaped noise (same shape as Lambda).
        """
        if noise_flat.dim() == 2:
            n_img = self.A_pinv(noise_flat)
        elif noise_flat.dim() == 4:
            n_img = noise_flat
        else:
            raise ValueError("Lambda_noise: unexpected dim")
        scale = 1.0 / (1.0 + float(sigma_y) * 100.0)
        return n_img * scale
def gaussian_smooth_mask(mask_arr, sigma=2.0):
    """
    Simple gaussian smoothing for numpy 2D mask.
    """
    # small separable gaussian blur using FFT via scipy if available, fallback to manual conv
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(mask_arr.astype(np.float32), sigma=sigma)
    except Exception:
        # fallback: simple gaussian kernel convolution (inefficient but works)
        ksize = int(max(3, math.ceil(sigma * 6)))
        if ksize % 2 == 0:
            ksize += 1
        coords = np.arange(ksize) - ksize // 2
        g1 = np.exp(-(coords**2) / (2 * sigma * sigma))
        g1 = g1 / g1.sum()
        g2 = np.outer(g1, g1)
        pad = ksize // 2
        img = np.pad(mask_arr.astype(np.float32), pad, mode='reflect')
        out = np.zeros_like(mask_arr, dtype=np.float32)
        H, W = mask_arr.shape
        for i in range(H):
            for j in range(W):
                patch = img[i:i+ksize, j:j+ksize]
                out[i,j] = np.sum(patch * g2)
        return out