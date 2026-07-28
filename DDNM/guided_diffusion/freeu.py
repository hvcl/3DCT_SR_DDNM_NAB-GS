import torch
import torch.nn.functional as F
import torch.fft as fft

def Fourier_filter(x, threshold, scale):
    x_freq = fft.fftn(x.float(), dim=(-2, -1))  # float()로 변환
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(x.dtype)

def apply_freeu(h, hidden_size, b1, b2, s1, s2):
    if h.shape[1] == hidden_size:
        hidden_mean = h.mean(1, keepdim=True)
        hidden_max = hidden_mean.amax(dim=(2, 3), keepdim=True)
        hidden_min = hidden_mean.amin(dim=(2, 3), keepdim=True)
        hidden_mean = (hidden_mean - hidden_min) / (hidden_max - hidden_min)
        h[:, :hidden_size//2] = h[:, :hidden_size//2] * ((b1 - 1) * hidden_mean + 1)
        h[:, hidden_size//2:] = Fourier_filter(h[:, hidden_size//2:], threshold=1, scale=s1)
    elif h.shape[1] == hidden_size // 2:
        hidden_mean = h.mean(1, keepdim=True)
        hidden_max = hidden_mean.amax(dim=(2, 3), keepdim=True)
        hidden_min = hidden_mean.amin(dim=(2, 3), keepdim=True)
        hidden_mean = (hidden_mean - hidden_min) / (hidden_max - hidden_min)
        h[:, :hidden_size//4] = h[:, :hidden_size//4] * ((b2 - 1) * hidden_mean + 1)
        h[:, hidden_size//4:] = Fourier_filter(h[:, hidden_size//4:], threshold=1, scale=s2)
    return h

def apply_freeu_single(h, b, s):
    hidden_size = h.shape[1]
    hidden_mean = h.mean(1, keepdim=True)
    hidden_max = hidden_mean.amax(dim=(2, 3), keepdim=True)
    hidden_min = hidden_mean.amin(dim=(2, 3), keepdim=True)
    hidden_mean = (hidden_mean - hidden_min) / (hidden_max - hidden_min)
    h[:, :hidden_size//2] = h[:, :hidden_size//2] * ((b - 1) * hidden_mean + 1)
    h[:, hidden_size//2:] = Fourier_filter(h[:, hidden_size//2:], threshold=1, scale=s)
    # h = h * ((b - 1) * hidden_mean + 1)
    # h = Fourier_filter(h, threshold=1, scale=s)
    return h