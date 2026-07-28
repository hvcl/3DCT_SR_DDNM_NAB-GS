# Modified Phase 1: Per-projection Latent Optimization
# This file replaces the DDNM approach with latent code optimization

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def ddim_reverse_step(x_t, noise_pred, t_curr, t_next, alphas_cumprod):
    """
    DDIM reverse diffusion step
    x_t: current latent (B, C, H, W)
    noise_pred: predicted noise from model
    t_curr: current timestep
    t_next: next (previous in reverse) timestep
    alphas_cumprod: cumulative product of alphas
    """
    alpha_t = alphas_cumprod[t_curr]
    alpha_next = alphas_cumprod[t_next]
    
    sigma_t = (1.0 - alpha_next).sqrt()
    sigma_t_curr = (1.0 - alpha_t).sqrt()
    
    # Predicted x0
    x_0 = (x_t - sigma_t_curr * noise_pred) / alpha_t.sqrt()
    
    # DDIM step
    x_next = alpha_next.sqrt() * x_0 + sigma_t * noise_pred
    
    return x_next, x_0


def optimize_projection_latent(
    y_lr,                  # LR projection (normalized to [-1, 1])
    diffusion_model,
    betas,
    alphas_cumprod,
    A_funcs,
    num_opt_steps=50,
    lr=0.01,
    delta_t=100,           # DDIM step size
    device='cuda',
    verbose=False
):
    """
    Per-projection latent optimization using gradient descent
    
    Args:
        y_lr: LR projection in model domain ([-1, 1]) shape (1, C, H, W)
        diffusion_model: pretrained diffusion model
        betas: noise schedule
        alphas_cumprod: cumulative alpha schedule
        A_funcs: degradation operator (provides downsampling)
        num_opt_steps: number of optimization steps
        lr: learning rate
        delta_t: DDIM step size (100 = 10 reverse steps for 1000 total)
        device: torch device
        verbose: print progress
    
    Returns:
        x_0_final: generated SR projection
        z_T_final: final optimized initial noise
    """
    
    # Initialize z_T (initial noise, learnable)
    z_T = torch.randn(1, y_lr.shape[1], 512, 512, device=device)
    z_T.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.Adam([z_T], lr=lr)
    
    # Diffusion timesteps for DDIM (accelerated: 10 steps)
    timesteps = list(range(0, 1000, delta_t))
    
    loss_history = []
    
    for step in range(num_opt_steps):
        # Forward pass: DDIM reverse
        x_t = z_T.clone()
        
        for idx in range(len(timesteps) - 1):
            t_curr = timesteps[idx]
            t_next = timesteps[idx + 1]
            
            t_tensor = torch.tensor([t_curr] * x_t.shape[0], device=device)
            
            with torch.no_grad():
                noise_pred = diffusion_model(x_t, t_tensor)
            
            if noise_pred.shape[1] == 6:
                noise_pred = noise_pred[:, :3]
            
            x_t_next, _ = ddim_reverse_step(
                x_t, noise_pred, 
                t_curr, t_next,
                alphas_cumprod
            )
            x_t = x_t_next
        
        x_0_generated = x_t  # Final image
        
        # Loss computation
        # 1. Data consistency loss
        x_0_downsampled = A_funcs.A(x_0_generated.reshape(1, -1)).reshape(1, 3, 128, 128)
        loss_data = F.l1_loss(x_0_downsampled, y_lr)
        
        # 2. Manifold regularization (keep z_T magnitude controlled)
        loss_manifold = 0.01 * torch.norm(z_T)
        
        loss_total = loss_data + loss_manifold
        loss_history.append(loss_total.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if verbose and step % 10 == 0:
            print(f"  Opt step {step}/{num_opt_steps}, Loss: {loss_total.item():.6f}")
        
        # Early stopping if converged
        if step > 20:
            recent_loss = np.mean(loss_history[-5:])
            prev_loss = np.mean(loss_history[-10:-5])
            if abs(recent_loss - prev_loss) < 1e-5:
                if verbose:
                    print(f"  Converged at step {step}")
                break
    
    # Final generation with optimized z_T
    with torch.no_grad():
        x_t = z_T.clone()
        
        for idx in range(len(timesteps) - 1):
            t_curr = timesteps[idx]
            t_next = timesteps[idx + 1]
            
            t_tensor = torch.tensor([t_curr] * x_t.shape[0], device=device)
            
            noise_pred = diffusion_model(x_t, t_tensor)
            
            if noise_pred.shape[1] == 6:
                noise_pred = noise_pred[:, :3]
            
            x_t_next, _ = ddim_reverse_step(
                x_t, noise_pred,
                t_curr, t_next,
                alphas_cumprod
            )
            x_t = x_t_next
        
        x_0_final = x_t
    
    return x_0_final, z_T.detach()


def latent_optimization_diffusion(
    model, 
    betas, 
    alphas_cumprod,
    A_funcs,
    y_batch,
    temp_y_batch,
    config,
    args,
    num_opt_steps_range=[20, 50, 100],
    lr_range=[0.001, 0.01, 0.05],
    delta_t_range=[50, 100, 200],
):
    """
    Wrapper for hyperparameter search in latent optimization
    
    Args:
        model: diffusion model
        betas, alphas_cumprod: diffusion schedules
        A_funcs: degradation operators
        y_batch: LR projections batch
        temp_y_batch: temporary y (initial noise distribution)
        config: diffusion config
        args: command line args
        num_opt_steps_range: range of optimization steps to try
        lr_range: range of learning rates
        delta_t_range: range of DDIM step sizes
    
    Returns:
        results: list of (num_opt_steps, lr, delta_t, x_0_output)
    """
    
    results = []
    total_configs = len(num_opt_steps_range) * len(lr_range) * len(delta_t_range)
    config_idx = 0
    
    for num_opt_steps in num_opt_steps_range:
        for lr in lr_range:
            for delta_t in delta_t_range:
                config_idx += 1
                print(f"\n=== Config {config_idx}/{total_configs}: "
                      f"num_opt_steps={num_opt_steps}, lr={lr}, delta_t={delta_t} ===")
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                x_0_opt, z_T_opt = optimize_projection_latent(
                    y_batch[0:1],
                    model,
                    betas,
                    alphas_cumprod,
                    A_funcs,
                    num_opt_steps=num_opt_steps,
                    lr=lr,
                    delta_t=delta_t,
                    device=device,
                    verbose=True
                )
                
                results.append({
                    'num_opt_steps': num_opt_steps,
                    'lr': lr,
                    'delta_t': delta_t,
                    'x_0_output': x_0_opt,
                    'z_T_output': z_T_opt
                })
    
    return results
