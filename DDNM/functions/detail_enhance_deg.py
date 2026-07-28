import math
import torch
import torch.nn.functional as F

from functions.svd_operators import A_functions


def _gaussian_kernel1d(sigma, truncate=3.0, device="cpu", dtype=torch.float32):
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


class _GaussianLowpassBase(A_functions):
    """
    Gaussian lowpass anchor for detail enhancement.

    - A(x): lowpass(x)
    - Lambda(...): only low-frequency residual is enforced
    - Lambda_noise(...): mostly low-frequency noise, plus a very small structured
      high-frequency term from the diffusion prediction
    """

    def __init__(
        self,
        channels,
        img_dim,
        sigma,
        device,
        highfreq_gain=1.0,
        lowfreq_noise_scale=0.0,
        random_highfreq_scale=0.0,
        structured_highfreq_scale=0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.img_dim = int(img_dim)
        self.device = torch.device(device)
        self.sigma = float(sigma)
        self.highfreq_gain = float(highfreq_gain)
        self.lowfreq_noise_scale = float(lowfreq_noise_scale)
        self.random_highfreq_scale = float(random_highfreq_scale)
        self.structured_highfreq_scale = float(structured_highfreq_scale)

        kernel1d = _gaussian_kernel1d(self.sigma, device=self.device)
        kernel2d = torch.outer(kernel1d, kernel1d)
        kernel2d = kernel2d / kernel2d.sum()
        self.kernel_size = int(kernel2d.shape[0])
        self.kernel = kernel2d.view(1, 1, self.kernel_size, self.kernel_size)

    def V(self, vec):
        return vec

    def Vt(self, vec):
        return vec

    def U(self, vec):
        return vec

    def Ut(self, vec):
        return vec

    def singulars(self):
        return torch.ones(self.channels * self.img_dim * self.img_dim, device=self.device)

    def add_zeros(self, vec):
        return vec

    def _to_image(self, x):
        if x.dim() == 4:
            return x
        if x.dim() == 2:
            return x.view(x.shape[0], self.channels, self.img_dim, self.img_dim)
        raise ValueError(f"Unexpected tensor shape: {x.shape}")

    def _gaussian_lowpass(self, x):
        kernel = self.kernel.to(x.device).to(x.dtype).repeat(self.channels, 1, 1, 1)
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel, padding=0, groups=self.channels)

    def _highpass(self, x):
        return x - self._gaussian_lowpass(x)

    def A(self, x):
        x_img = self._to_image(x)
        low = self._gaussian_lowpass(x_img)
        return low.reshape(low.shape[0], -1)

    def A_pinv(self, y):
        return self._gaussian_lowpass(self._to_image(y))

    def At(self, y):
        return self.A_pinv(y)

    def Lambda(self, residual_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None):
        residual = self._to_image(residual_flat)
        low = self._gaussian_lowpass(residual)
        scale = 1.0 / (1.0 + float(sigma_y) * 100.0)
        return low * scale

    def Lambda_noise(self, noise_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None, et_flat=None):
        noise = self._to_image(noise_flat)
        noise_low = self._gaussian_lowpass(noise)
        noise_high = self._highpass(noise)

        eta_value = 1.0 if eta is None else float(eta)
        low_scale = self.lowfreq_noise_scale / (1.0 + float(sigma_y) * 50.0)
        rand_high_scale = self.random_highfreq_scale * self.highfreq_gain * max(1e-3, 1.0 - eta_value)

        out = noise_low * low_scale + noise_high * rand_high_scale

        if et_flat is not None:
            et_img = self._to_image(et_flat)
            structured_high = self._highpass(et_img)
            structured_scale = self.structured_highfreq_scale * self.highfreq_gain / (1.0 + float(sigma_y) * 10.0)
            out = out + structured_high * structured_scale

        return out


class DetailEnhanceLowpass(_GaussianLowpassBase):
    """
    Backward-compatible name.
    cutoff_radius is mapped to a Gaussian sigma.
    """

    def __init__(self, channels, img_dim, cutoff_radius, device, highfreq_gain=1.0):
        if cutoff_radius <= 1.0:
            sigma = max(0.8, cutoff_radius * (img_dim // 8))
        else:
            sigma = max(0.8, float(cutoff_radius) / 24.0)
        super().__init__(channels, img_dim, sigma, device, highfreq_gain=highfreq_gain)


class DetailEnhanceGaussian(_GaussianLowpassBase):
    """
    Explicit Gaussian detail-enhancement operator controlled directly by sigma.
    """

    def __init__(self, channels, img_dim, sigma, device, highfreq_gain=1.0):
        super().__init__(channels, img_dim, sigma, device, highfreq_gain=highfreq_gain)


class DetailEnhanceLowRef(_GaussianLowpassBase):
    """
    Low-only correction operator.

    - Measurement consistency only constrains low-frequency content.
    - Final output is expected to be composed as:
      low(pred) + (before - low(before))
    """

    def __init__(self, channels, img_dim, sigma, device, low_anchor_strength=1.0):
        super().__init__(channels, img_dim, sigma, device, highfreq_gain=1.0)
        self.low_anchor_strength = float(low_anchor_strength)

    def compose_low_target(self, before_x, first_x, low_ref_alpha=0.5):
        before_img = self._to_image(before_x)
        first_img = self._to_image(first_x)
        low_before = self._gaussian_lowpass(before_img)
        low_first = self._gaussian_lowpass(first_img)
        alpha = float(low_ref_alpha)
        return (1.0 - alpha) * low_before + alpha * low_first

    def compose_final(self, refined_x, before_x):
        refined_img = self._to_image(refined_x)
        before_img = self._to_image(before_x)
        refined_low = self._gaussian_lowpass(refined_img)
        before_low = self._gaussian_lowpass(before_img)
        before_midhigh = before_img - before_low
        return refined_low + before_midhigh

    def Lambda(self, residual_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None):
        residual = self._to_image(residual_flat)
        low = self._gaussian_lowpass(residual)
        scale = self.low_anchor_strength / (1.0 + float(sigma_y) * 100.0)
        return low * scale

    def Lambda_noise(self, noise_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None, et_flat=None):
        noise = self._to_image(noise_flat)
        return torch.zeros_like(noise)


class DetailEnhanceBandpass(A_functions):
    """
    3-band detail enhancement operator.

    low band: softly anchored via `low_anchor_strength`
    mid band: partially anchored and enhanced via `mid_gain`
    very-high band: excluded from consistency to avoid ringing/patterns
    """

    def __init__(
        self,
        channels,
        img_dim,
        sigma_low,
        sigma_mid,
        device,
        mid_gain=1.0,
        low_anchor_strength=1.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.img_dim = int(img_dim)
        self.device = torch.device(device)
        self.sigma_low = float(max(sigma_low, sigma_mid + 1e-3))
        self.sigma_mid = float(max(0.5, sigma_mid))
        self.mid_gain = float(mid_gain)
        self.low_anchor_strength = float(low_anchor_strength)

        self.low_kernel = self._make_kernel(self.sigma_low)
        self.mid_kernel = self._make_kernel(self.sigma_mid)

    def _make_kernel(self, sigma):
        kernel1d = _gaussian_kernel1d(sigma, device=self.device)
        kernel2d = torch.outer(kernel1d, kernel1d)
        kernel2d = kernel2d / kernel2d.sum()
        return kernel2d.view(1, 1, kernel2d.shape[0], kernel2d.shape[1])

    def V(self, vec):
        return vec

    def Vt(self, vec):
        return vec

    def U(self, vec):
        return vec

    def Ut(self, vec):
        return vec

    def singulars(self):
        return torch.ones(self.channels * self.img_dim * self.img_dim, device=self.device)

    def add_zeros(self, vec):
        return vec

    def _to_image(self, x):
        if x.dim() == 4:
            return x
        if x.dim() == 2:
            return x.view(x.shape[0], self.channels, self.img_dim, self.img_dim)
        raise ValueError(f"Unexpected tensor shape: {x.shape}")

    def _blur(self, x, kernel):
        kernel = kernel.to(x.device).to(x.dtype).repeat(self.channels, 1, 1, 1)
        pad = kernel.shape[-1] // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel, padding=0, groups=self.channels)

    def _decompose(self, x):
        low = self._blur(x, self.low_kernel)
        smooth_mid = self._blur(x, self.mid_kernel)
        mid = smooth_mid - low
        high = x - smooth_mid
        return low, mid, high

    def _compose_measurement(self, x):
        low, mid, _ = self._decompose(x)
        return low + self.mid_gain * mid

    def compose_from_refs(self, before_x, first_x):
        before_img = self._to_image(before_x)
        first_img = self._to_image(first_x)
        low_before, _, _ = self._decompose(before_img)
        _, mid_first, _ = self._decompose(first_img)
        return low_before + self.mid_gain * mid_first

    def A(self, x):
        x_img = self._to_image(x)
        meas = self._compose_measurement(x_img)
        return meas.reshape(meas.shape[0], -1)

    def A_pinv(self, y):
        return self._compose_measurement(self._to_image(y))

    def At(self, y):
        return self.A_pinv(y)

    def Lambda(self, residual_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None):
        residual = self._to_image(residual_flat)
        low, mid, _ = self._decompose(residual)
        meas_residual = self.low_anchor_strength * low + self.mid_gain * mid
        scale = 1.0 / (1.0 + float(sigma_y) * 100.0)
        return meas_residual * scale

    def Lambda_noise(self, noise_flat, at_sqrt=None, sigma_y=0.0, sigma_t=None, eta=None, et_flat=None):
        # Intentionally disable stochastic injection first.
        noise = self._to_image(noise_flat)
        return torch.zeros_like(noise)
