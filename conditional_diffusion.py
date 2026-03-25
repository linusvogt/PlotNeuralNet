import math
from typing import List, Optional, Tuple, Sequence

import numpy as np
import xarray as xr
from tqdm.autonotebook import tqdm
from pathlib import Path
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TracerDataset(Dataset):
    """
    Wrap an xarray.Dataset with dims (sample, lat, lon) and multiple variables.

    Returns dicts with keys:
     - 'y': (1,H,W) normalized target
     - 'cond': (C,H,W) normalized predictors
     - 'mask': (1,H,W) binary float mask with 1 for ocean, 0 for land (same for all samples)
    """

    def __init__(self, ds: xr.Dataset, predictors: List[str], target: str, dtype=np.float32):
        """Initialize torch Dataset from stacked xarray Dataset.

        Args:
            ds (xr.Dataset): dataset with dims (sample, lat, lon) and several variables
                (both predictors and target)
            predictors (list of str): predictor variables (to condition on in DDPM)
            target (str): target variable
            dtype: default=float32
        """

        # make sure all necessary variables are in dataset
        assert target in ds.variables
        for p in predictors:
            assert p in ds.variables
        assert target not in predictors

        # list of all variables
        variables = predictors + [target]

        # check ds dimensions and transpose to correct order
        dims_list = [dim for dim in ds.dims]
        assert set(dims_list) == {'sample', 'lat', 'lon'}, f'{dims_list=}'
        ds = ds.transpose('sample', 'lat', 'lon')

        self.predictors = predictors
        self.target = target
        self.dtype = dtype

        # ocean mask from first sample (True on ocean, False on land)
        self.mask = np.isfinite(ds[target].isel(sample=0).values)

        # compute *area-weighted* mean/std for each var
        da_area = xr.open_dataset('/scratch/lv2429/gridarea_r360x180.nc')
        da_area = da_area.cell_area.fillna(0.).transpose('lat', 'lon')
        self.means = {var: float(ds[var].weighted(da_area).mean()) for var in variables}
        self.stds = {var: float(ds[var].weighted(da_area).std()) for var in variables}

        # precompute normalized arrays and set land cells to 0
        self._preproc = {}
        for var in variables:
            da_preproc = (ds[var] - self.means[var]) / self.stds[var]  # normalize
            da_preproc = da_preproc.fillna(0.)
            self._preproc[var] = da_preproc.values.astype(self.dtype)

        self.n_samples = len(ds.sample)
        self.mask_tensor = torch.from_numpy(self.mask.astype(np.float32))[None, ...]  # (1,H,W)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        y = self._preproc[self.target][idx:idx+1]  # (1,H,W)
        cond = np.concatenate([self._preproc[p][idx:idx+1] for p in self.predictors], axis=0)
        return {
            "y": torch.from_numpy(y).float(),
            "cond": torch.from_numpy(cond).float(),
        }


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,) long or float
        returns: (B, dim)
        """
        device = timesteps.device
        half = self.dim // 2
        # Avoid div-by-zero if dim=1 (not a realistic setting here, but safe)
        if half <= 1:
            emb = timesteps.float()[:, None]
            return F.pad(emb, (0, self.dim - emb.shape[-1]))

        freq = math.log(10000.0) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device) * -freq)  # (half,)
        t = timesteps.float()[:, None]                                # (B,1)
        emb = t * freq[None, :]                                       # (B,half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)               # (B,2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.act = nn.SiLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_ch)

        # time embedding -> channel-wise bias (FiLM-like shift)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.gn2(self.conv2(h)))
        return h + self.res_conv(x)


class SimpleUNetCond(nn.Module):
    """
    Conditional UNet backbone for DDPM-style diffusion on 2D fields.

    This network predicts a single target channel (e.g. noise ε or velocity v)
    from a multi-channel input consisting of the noisy target and conditioning
    variables. Time is injected through a sinusoidal embedding followed by an MLP
    and applied in every residual block as a channel-wise bias (FiLM-like shift).

    Architecture
    ------------
    - Encoder/decoder UNet with skip connections via **addition** (not concatenation).
    - Each resolution level contains `num_res_blocks` residual blocks.
    - Downsampling is performed with average pooling (factor 2).
    - Upsampling is bilinear followed by a 1×1 projection if channel counts differ.
    - A two-block bottleneck operates at the lowest resolution.
    - Final 1×1 convolution maps features to a single output channel.

    Parameters
    ----------
    in_ch : int
        Number of input channels. Typically:
        in_ch = 1 (noisy target) + N_cond (conditioning variables).
    base_ch : int, default=64
        Base channel width of the UNet. The actual channels at each level are
        `base_ch * ch_mults[level]`. Controls overall model capacity.
    ch_mults : Sequence[int], default=(1, 2, 4)
        Multiplicative factors defining channel width at each resolution level.
        Length of this sequence equals the number of UNet levels (depth).
    time_emb_dim : int, default=256
        Dimensionality of the timestep embedding. Used in all ResBlocks.
    num_res_blocks : int, default=2
        Number of residual blocks per resolution level (both encoder and decoder).
        Increasing this increases model depth and capacity.
    gn_groups : int, default=8
        Number of groups for GroupNorm in residual blocks.

    Notes
    -----
    - Skip connections use **elementwise addition**, so encoder and decoder
      feature maps must share the same channel count at each level.
    - Spatial size must be divisible by 2^(len(ch_mults)) for exact symmetry,
      though minor mismatches are handled via interpolation.
    - Output shape is (B, 1, H, W), matching a single predicted target channel.
    """
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 64,
        ch_mults: Sequence[int] = (1, 2, 4),
        time_emb_dim: int = 256,
        num_res_blocks: int = 2,
        gn_groups: int = 8,
    ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mults = tuple(ch_mults)
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks

        # time embedding: sinusoidal -> MLP
        self.time_sinu = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # helpers
        def make_res_stack(in_c: int, out_c: int) -> nn.ModuleList:
            blocks = nn.ModuleList()
            blocks.append(ResBlock(in_c, out_c, time_emb_dim, groups=gn_groups))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_c, out_c, time_emb_dim, groups=gn_groups))
            return blocks

        # -------------------------
        # Encoder (down path)
        # -------------------------
        self.down_levels = nn.ModuleList()
        self.skip_channels: Tuple[int, ...] = tuple(base_ch * m for m in self.ch_mults)

        in_c = in_ch
        for out_c in self.skip_channels:
            level = nn.ModuleDict(
                {
                    "blocks": make_res_stack(in_c, out_c),
                    "downsample": nn.AvgPool2d(kernel_size=2),
                }
            )
            self.down_levels.append(level)
            in_c = out_c

        # -------------------------
        # Bottleneck
        # -------------------------
        self.mid = nn.ModuleList(
            [
                ResBlock(in_c, in_c, time_emb_dim, groups=gn_groups),
                ResBlock(in_c, in_c, time_emb_dim, groups=gn_groups),
            ]
        )

        # -------------------------
        # Decoder (up path)
        # -------------------------
        self.up_levels = nn.ModuleList()
        prev_c = in_c
        for skip_c in reversed(self.skip_channels):
            level = nn.ModuleDict(
                {
                    "upsample": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    "proj": nn.Conv2d(prev_c, skip_c, kernel_size=1) if prev_c != skip_c else nn.Identity(),
                    "blocks": make_res_stack(skip_c, skip_c),
                }
            )
            self.up_levels.append(level)
            prev_c = skip_c

        self.final = nn.Conv2d(prev_c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_ch, H, W) where in_ch = 1 + n_cond
        timesteps: (B,) int tensor of time indices
        returns: (B, 1, H, W)
        """
        # time embedding
        t = self.time_mlp(self.time_sinu(timesteps))

        h = x
        skips = []

        # down path
        for level in self.down_levels:
            for block in level["blocks"]:
                h = block(h, t)
            skips.append(h)
            h = level["downsample"](h)

        # bottleneck
        for block in self.mid:
            h = block(h, t)

        # up path
        for level, skip in zip(self.up_levels, reversed(skips)):
            h = level["upsample"](h)
            h = level["proj"](h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = h + skip
            for block in level["blocks"]:
                h = block(h, t)

        return self.final(h)



class EMA:
    """Exponential moving average helper class."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Load EMA weights into model (for sampling / validation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original training weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
        

class DDPM:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        img_size: Tuple[int,int] = (180, 360),
        num_timesteps: int = 1000,
        penalize_non_negative: bool = False,
        integral_loss: bool = False,
        w_integral=0,
        schedule='linear',
        prediction='eps',
    ):
        """
        Denoising Diffusion Probabilistic Model (DDPM) implementation.

        Params:
            model: U-Net to be trained to predict noise
            device: torch.device (use GPU)
            img_size: shape of maps (360x180 for 1deg grid)
            num_timesteps: number of (de)noising time steps
            penalize_non_negative: whether to penalize negative values
                (for target tracers like salinity, oxygen; but only for
                full fields, not anomalies!)
            integral_loss: whether to add a basic global integral loss term
        """
        if penalize_non_negative:# or integral_loss:
            raise NotImplementedError('Non-negative constraint currently commented out for schedule/prediction testing')
        
        self.model = model
        self.device = device
        self.img_size = img_size
        self.num_timesteps = num_timesteps
        self.penalize_non_negative = penalize_non_negative
        self.w_nonneg = 1  # weight for non-negativity loss
        self.integral_loss = integral_loss
        self.w_integral = w_integral

        # grid cell area (for area weighted integral constraint)
        da_area = xr.open_dataset('/scratch/lv2429/gridarea_r360x180.nc')
        da_area = da_area.cell_area.fillna(0.).transpose('lat', 'lon')
        self.area_tensor = torch.tensor(
            da_area.values, dtype=torch.float32).unsqueeze(0).to(self.device)

        # prediction: "eps" or "v"
        if prediction not in ("eps", "v"):
            raise ValueError(f"prediction must be 'eps' or 'v', got {prediction}")
        self.prediction = prediction

        # schedule: "linear" or "cos"
        self.schedule = schedule
        if schedule == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule == "cos":
            # Cosine schedule via alpha_bar(t)
            s = 0.008
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps, device=device) / num_timesteps
        
            alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar(0)=1
        
            # Convert alpha_bar -> betas
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError(f"Unknown schedule '{schedule}', must be 'linear' or 'cos'")
        
        self.alphas = 1.0 - self.betas                                    # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)            # alpha_bar_t
        
        # alpha_bar_{t-1} with alpha_bar_{-1}=1
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0
        )
        
        # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)
        
        # useful precomputed quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)


    def _extract(self, vec: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        """
        vec: (T,)
        t: (B,)
        returns vec[t] reshaped to (B,1,1,1) for broadcasting over x
        """
        return vec[t].view(-1, 1, 1, 1).to(x.device)
    
    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        # v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return sqrt_ab * noise - sqrt_1mab * x0
    
    def v_to_eps_x0(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """
        Given x_t and v, return (eps, x0).
        eps = sqrt(1-alpha_bar)*x_t + sqrt(alpha_bar)*v
        x0  = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v
        """
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        eps = sqrt_1mab * x_t + sqrt_ab * v
        x0 = sqrt_ab * x_t - sqrt_1mab * v
        return eps, x0

    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion q(x_t | x_0)
        x0: (B,1,H,W)
        t: (B,) long
        noise: (B,1,H,W)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1,1,1,1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1)
        return sqrt_acp * x0 + sqrt_om * noise

    def p_losses(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Computes MSE between true noise (or v) and predicted noise (or v),
        plus optional global-mean loss on reconstructed x0.
    
        Returns:
            (loss_mse, loss_integral) where loss_integral is already weighted
            by self.w_integral when enabled, else None.
        """
        B = x0.shape[0]
        device = x0.device
    
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
    
        model_in = torch.cat([xt, cond], dim=1)   # (B, 1+C, H, W)
        pred = self.model(model_in, t)            # (B, 1, H, W)
    
        # --- base DDPM loss target ---
        if self.prediction == "eps":
            target = noise
        elif self.prediction == "v":
            target = self.v_target(x0=x0, noise=noise, t=t)
        else:
            raise RuntimeError(f"Unknown prediction type: {self.prediction}")
    
        mse = (target - pred) ** 2  # (B,1,H,W)
    
        # --- build mask_b once (used by both losses) ---
        if mask is not None:
            if mask.shape[0] == 1:
                mask_b = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(device)
            else:
                mask_b = mask.to(device)
        else:
            mask_b = None
    
        # --- pixelwise MSE with optional masking ---
        if mask_b is not None:
            mse_w = mse * mask_b
            loss_mse = mse_w.sum() / mask_b.sum().clamp_min(1.0)
        else:
            loss_mse = mse.mean()
    
        # --- optional global-mean loss on reconstructed x0 ---
        if self.integral_loss:
            # 1) reconstruct predicted eps (needed to reconstruct x0_pred robustly)
            if self.prediction == "eps":
                pred_eps = pred
            elif self.prediction == "v":
                pred_eps, _x0_pred_direct = self.v_to_eps_x0(x_t=xt, v=pred, t=t)
            else:
                raise RuntimeError(f"Unknown prediction type: {self.prediction}")
    
            # 2) reconstruct x0_pred from (x_t, eps_pred)
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, xt)              # (B,1,1,1)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt)  # (B,1,1,1)
            x0_pred = (xt - sqrt_1mab * pred_eps) / sqrt_ab.clamp_min(1e-12)      # (B,1,H,W)
    
            # 3) compute (area-)weighted ocean mean error of x0
            #    weights w = mask * area  (if mask is None, treat as all-ones)
            if mask_b is None:
                # allow integral loss even without a mask
                mask_b = torch.ones_like(x0_pred, device=device)
    
            area = self.area_tensor.to(device)  # stored as (1,H,W)
            if area.ndim == 3:
                area = area.unsqueeze(1)        # -> (1,1,H,W)
            area_b = area.expand(B, 1, area.shape[-2], area.shape[-1])  # (B,1,H,W)
    
            w = (mask_b * area_b)  # (B,1,H,W)  area-weighted ocean weights
    
            # mean error per sample: μ_b = sum(w * (x0_pred-x0)) / sum(w)
            err_x0 = (x0_pred - x0)
            wsum = w.sum(dim=(2, 3), keepdim=False).clamp_min(1e-12)               # (B,1)
            mean_err = (w * err_x0).sum(dim=(2, 3), keepdim=False) / wsum          # (B,1)
    
            # 4) scale to match per-pixel MSE magnitude:
            #    Var(mean) ~ Var / N_eff  ->  mean^2 * N_eff ~ Var scale
            #    Use Kish effective sample size for weighted averages:
            #       N_eff = (sum w)^2 / sum(w^2)
            wsqsum = (w ** 2).sum(dim=(2, 3), keepdim=False).clamp_min(1e-12)      # (B,1)
            n_eff = (wsum ** 2) / wsqsum                                           # (B,1)
    
            loss_integral = (mean_err ** 2 * n_eff).mean()                         # scalar
            loss_integral = self.w_integral * loss_integral
        else:
            loss_integral = None
    
        return loss_mse, loss_integral
    

    def CLEAN_p_losses(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Computes MSE between true noise and predicted noise.

        Params:
            x0: true target sample without noise
            cond: conditioning channels
            mask: land/sea mask
            t: timestep in diffusion process
        """
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
    
        model_in = torch.cat([xt, cond], dim=1)   # (B, 1+C, H, W)
        pred = self.model(model_in, t)            # (B,1,H,W)
        
        if self.prediction == "eps":
            target = noise
        elif self.prediction == "v":
            target = self.v_target(x0=x0, noise=noise, t=t)
        else:
            raise RuntimeError(f"Unknown prediction type: {self.prediction}")
        
        mse = (target - pred) ** 2

        # Masking
        if mask is not None:
            if mask.shape[0] == 1:
                mask_b = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(x0.device)
            else:
                mask_b = mask.to(x0.device)
    
            mse = mse * mask_b
            loss_mse = mse.sum() / mask_b.sum().clamp_min(1.0)
        else:
            loss_mse = mse.mean()

        return loss_mse

    
    def p_losses_constraints(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        OLD version before implementing eps/v prediction.
        Includes mass/non-negativity constraints.
        
        Computes MSE between true noise and predicted noise.

        Params:
            x0: true target sample without noise
            cond: conditioning channels
            mask: land/sea mask
            t: timestep in diffusion process
        """
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
    
        model_in = torch.cat([xt, cond], dim=1)  # (B, 1+C, H, W)
        pred_noise = self.model(model_in, t)     # (B,1,H,W)

        if self.penalize_non_negative or self.integral_loss:
            # reconstruct predicted x0
            sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1,1,1,1)
            sqrt_om  = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1)
            x0_pred = (xt - sqrt_om * pred_noise) / sqrt_acp

        loss = 0
    
        # -----------------------------------------------------------
        # 1. Standard DDPM noise-prediction loss
        # -----------------------------------------------------------
        mse = (noise - pred_noise) ** 2
    
        # Masking
        if mask is not None:
            if mask.shape[0] == 1:
                mask_b = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(x0.device)
            else:
                mask_b = mask.to(x0.device)
    
            mse = mse * mask_b
            loss_mse = mse.sum() / mask_b.sum().clamp_min(1.0)
        else:
            loss_mse = mse.mean()

        loss += loss_mse

        # -----------------------------------------------------------
        # 2. Non-negativity loss
        # -----------------------------------------------------------
        if self.penalize_non_negative:
            loss_nonneg = (F.relu(-x0_pred) * mask).sum() / mask.sum().clamp_min(1.0)
            loss += self.w_nonneg * loss_nonneg

            # logging.info(f'{float(loss_mse)=}')
            # logging.info(f'{self.w_nonneg * float(loss_nonneg)=}')

        # -----------------------------------------------------------
        # 3. Global integral loss
        # -----------------------------------------------------------

        if self.integral_loss:
    
            # difference only over ocean
            diff = (x0 - x0_pred) * mask_b               # (B,1,H,W)
        
            # area weighting
            area = self.area_tensor        # (1,H,W)
            if area.ndim == 3:
                area = area.unsqueeze(1)                 # -> (1,1,H,W)
            area_b = area.expand(B, 1, area.shape[-2], area.shape[-1])
    
            # total ocean area (scalar)
            A_total = (area_b * mask_b).sum() + 1e-12
        
            # compute mass difference
            mass_diff = (diff * area_b).sum(dim=[1,2,3])    # (B,)
            loss_integral = ((mass_diff**2) / (A_total**2)).mean()
            
            loss += self.w_integral * loss_integral
            
            # logging.info(f'{float(loss_mse)=}')
            # logging.info(f'{self.w_integral * float(loss_integral)=}')


        return loss  # return total loss

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, mask: torch.Tensor,
        sampler: str = "ddpm", # or "ddim"
        # below: DDIM args
        steps=100,
        eta=0.1,
        timestep_spacing="uniform",  # or "quadratic"
    ):
        if sampler == 'ddpm':
            return self.sample_ddpm(cond, mask)
        elif sampler == 'ddim':
            return self.sample_ddim(
                cond, mask,
                steps=steps, eta=eta, timestep_spacing=timestep_spacing
            )
        else:
            raise ValueError(f'{sampler=}')

    @torch.no_grad()
    def sample_ddpm(self, cond: torch.Tensor, mask: torch.Tensor):
        """
        Sample x0 given cond.
        cond: (B, C, H, W)
        mask: (1,H,W) or (B,1,H,W)
        """
        B = cond.shape[0]  # number of samples
        H, W = self.img_size
        x = torch.randn((B, 1, H, W), device=self.device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            model_in = torch.cat([x, cond], dim=1)
            pred = self.model(model_in, t_batch)  # (B,1,H,W)
            
            if self.prediction == "eps":
                pred_eps = pred
                # broadcast scalars
                sqrt_ab = self.sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
                sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
                x0_pred = (x - sqrt_1mab * pred_eps) / sqrt_ab
            
            elif self.prediction == "v":
                pred_v = pred
                # directly recover x0 (more stable) and eps (needed for mean formula if you prefer)
                pred_eps, x0_pred = self.v_to_eps_x0(x_t=x, v=pred_v, t=t_batch)
            
            else:
                raise RuntimeError(f"Unknown prediction type: {self.prediction}")

            # posterior update
            if t > 0:
                beta_t = self.betas[t]
                alpha_cum_t = self.alphas_cumprod[t]
                alpha_cum_prev = self.alphas_cumprod_prev[t]
            
                coef1 = (beta_t * torch.sqrt(alpha_cum_prev)) / (1.0 - alpha_cum_t)
                coef2 = ((1.0 - alpha_cum_prev) * torch.sqrt(1.0 - beta_t)) / (1.0 - alpha_cum_t)
                mean = coef1 * x0_pred + coef2 * x
            
                var = self.posterior_variance[t]
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = x0_pred

            # enforce land mask = 0 in normalized space
            if mask is not None:
                if mask.shape[0] == 1:
                    m = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(x.device)
                else:
                    m = mask.to(x.device)
                x = x * m

        return x  # normalized-space prediction

    @torch.no_grad()
    def sample_ddim(
        self,
        cond: torch.Tensor,
        mask: torch.Tensor,
        steps: int = 100,
        eta: float = 0.0,
        timestep_spacing: str = "uniform",  # or "quadratic"
    ):
        """
        DDIM sampling (Song et al.) with optional stochasticity via `eta`.

        Usage:
        x = ddpm.sample_ddim(cond, mask, steps=50, eta=0.0)   # deterministic
        x = ddpm.sample_ddim(cond, mask, steps=50, eta=0.1)   # slightly stochastic
    
        Parameters
        ----------
        cond : torch.Tensor
            Conditioning tensor of shape (B, C, H, W).
        mask : torch.Tensor
            Land/sea mask of shape (1, H, W) or (B, 1, H, W). Applied each step to
            enforce land=0 in normalized space.
        steps : int
            Number of DDIM steps S (<< num_timesteps). Typical values: 25, 50, 100.
        eta : float
            DDIM stochasticity. eta=0 gives deterministic sampling. eta>0 adds noise.
            Typical values: 0.0, 0.1, 0.2.
        timestep_spacing : str
            "uniform" uses uniformly spaced timesteps.
            "quadratic" allocates more steps near t=0 (often slightly better).
    
        Returns
        -------
        x0 : torch.Tensor
            Generated samples in normalized space, shape (B, 1, H, W).
        """
        B = cond.shape[0]
        H, W = self.img_size
        device = self.device
    
        # initial noise
        x = torch.randn((B, 1, H, W), device=device)
    
        # build timestep schedule: a decreasing list of indices in [0, T-1]
        T = self.num_timesteps
        steps = int(steps)
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")
        if steps > T:
            raise ValueError(f"steps must be <= num_timesteps ({T}), got {steps}")
    
        if timestep_spacing == "uniform":
            t_seq = torch.linspace(T - 1, 0, steps, device=device)
        elif timestep_spacing == "quadratic":
            # more density near 0
            t_seq = (torch.linspace(0, 1, steps, device=device) ** 2) * (T - 1)
            t_seq = torch.flip(t_seq, dims=[0])
        else:
            raise ValueError("timestep_spacing must be 'uniform' or 'quadratic'")
    
        t_seq = t_seq.round().long().clamp(0, T - 1)
        # ensure strictly non-increasing and unique-ish (remove duplicates while preserving order)
        t_list = []
        last = None
        for ti in t_seq.tolist():
            if last is None or ti != last:
                t_list.append(ti)
                last = ti
        if t_list[-1] != 0:
            t_list.append(0)
    
        # prepare mask (broadcast once)
        if mask is not None:
            if mask.shape[0] == 1:
                m = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(device)
            else:
                m = mask.to(device)
        else:
            m = None
    
        for idx in range(len(t_list) - 1):
            t = t_list[idx]
            t_prev = t_list[idx + 1]  # smaller index (closer to 0)
    
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
    
            # model prediction at timestep t
            model_in = torch.cat([x, cond], dim=1)
            pred = self.model(model_in, t_batch)  # (B,1,H,W)
    
            # get x0_pred and eps_pred depending on parameterization
            if self.prediction == "eps":
                eps = pred
                sqrt_ab_t = self.sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
                sqrt_1mab_t = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
                x0_pred = (x - sqrt_1mab_t * eps) / sqrt_ab_t
    
            elif self.prediction == "v":
                eps, x0_pred = self.v_to_eps_x0(x_t=x, v=pred, t=t_batch)
    
            else:
                raise RuntimeError(f"Unknown prediction type: {self.prediction}")
    
            # DDIM update x_t -> x_{t_prev}
            a_t = self.alphas_cumprod[t].view(1, 1, 1, 1)         # alpha_bar_t
            a_prev = self.alphas_cumprod[t_prev].view(1, 1, 1, 1) # alpha_bar_{t_prev}
    
            # sigma_t (DDIM): controls added noise
            # sigma = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1 - a_t/a_prev)
            sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t)) * torch.sqrt(1.0 - (a_t / a_prev))
    
            # direction term
            # x_{t_prev} = sqrt(a_prev)*x0 + sqrt(1-a_prev-sigma^2)*eps + sigma*z
            c = torch.sqrt((1.0 - a_prev) - sigma ** 2).clamp_min(0.0)
    
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(a_prev) * x0_pred + c * eps + sigma * noise
    
            # enforce land mask after each step
            if m is not None:
                x = x * m
    
        # final x is x0 estimate at t=0
        return x


class Trainer:
    def __init__(
            self,
            ddpm: DDPM,
            dataset: Dataset,
            save_dir: str,
            save_every: int = 25,
            batch_size: int = 16,
            lr: float = 1e-4,
            epochs: int = 1000,
            num_workers: int = 4,
            grad_accum: int = 1,
            amp: bool = True,
            device=None,
            config=None
        ):
        self.ddpm = ddpm
        self.dataset = dataset
        self.save_dir = save_dir
        self.save_every = save_every
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_workers = num_workers
        self.grad_accum = grad_accum
        self.amp = amp
        self.device = device if device is not None else ddpm.device

        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True, pin_memory=True)
        self.opt = torch.optim.Adam(self.ddpm.model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp)

        self.ema = EMA(self.ddpm.model, decay=0.9999)

    def train(self, resume_from: str = None):
        model = self.ddpm.model
        device = self.device
        model.train()

        # load mask here, not in each epoch/batch
        mask = self.dataset.mask_tensor.to(device)  # (1,H,W)

        use_cuda_timing = (device.type == "cuda")
        
        logging.info('Start training')
        
        start_epoch = 0
        global_step = 0
        if resume_from is not None:
            start_epoch, global_step = self.load_checkpoint(resume_from)
            logging.info(f"Resuming from {resume_from}: {start_epoch=}, {global_step=}")

        losses_mse, losses_integral = [], []

        for epoch in range(start_epoch, self.epochs + 1):

            # Epoch wall-clock (no synchronize)
            epoch_t0 = time.perf_counter()

            # Data timing (CPU wall-clock for data loader wait + launch of H2D copies)
            data_time_total = 0.0

            # time spent waiting for the *next* batch (and doing H2D enqueue)
            t_data_start = time.perf_counter()

            measure_every = 20  # or 50
            evt_pairs = []
            comp_ms_total = 0.0
            
            for i, batch in enumerate(self.dl):
                # ---- data time (no GPU sync; measures host-side wait/enqueue) ----
                y = batch["y"].to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                cond = batch["cond"].to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                data_time_total += (time.perf_counter() - t_data_start)

                B = y.shape[0]
                t = torch.randint(0, self.ddpm.num_timesteps, (B,), device=device).long()

                do_measure = use_cuda_timing and (i % measure_every == 0)
                if do_measure:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()

                with torch.amp.autocast("cuda", enabled=self.amp):
                    loss_mse, loss_integral = self.ddpm.p_losses(y, cond, mask, t)

                    losses_mse.append(float(loss_mse.detach()))
                    if loss_integral is not None:
                        loss = loss_mse + loss_integral
                        losses_integral.append(float(loss_integral.detach()))
                    else:
                        loss = loss_mse
                    loss = loss / self.grad_accum
                        
                self.scaler.scale(loss).backward()

                if (i + 1) % self.grad_accum == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update(model)
                    global_step += 1  # update optimizer step count

                if do_measure:
                    end_evt.record()
                    evt_pairs.append((start_evt, end_evt))

                # start timing the wait for the next batch
                t_data_start = time.perf_counter()

            logging.info(f'\tMSE loss: {np.nanmean(losses_mse):.2e}, std={np.nanstd(losses_mse):.2e}')
            if loss_integral is not None:
                logging.info(f'\tIntegral loss: {np.nanmean(losses_integral):.2e}, std={np.nanstd(losses_integral):.2e}')
            
            # end epoch
            if use_cuda_timing:
                torch.cuda.synchronize(device)
                measured_ms = sum(s.elapsed_time(e) for s, e in evt_pairs)
                # scale up to approximate full epoch compute time
                comp_ms_total = measured_ms * (len(self.dl) / max(len(evt_pairs), 1))
            compute_time = comp_ms_total / 1000.0

            epoch_time = time.perf_counter() - epoch_t0
            data_time = data_time_total

            logging.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"time={epoch_time:.1f}s ({epoch_time/60:.2f}m) | "
                f"data={data_time:.1f}s ({100*data_time/max(epoch_time,1e-9):.0f}%) | "
                f"compute={compute_time:.2f}s ({100*compute_time/max(epoch_time,1e-9):.0f}%)"
            )

            # save checkpoint every N epochs, + last epoch
            if epoch % self.save_every == 0 or epoch == self.epochs:
                ckpt = {
                    "epoch": epoch + 1,
                    "global_step": global_step,            # optimizer-step count
                    "model_state": model.state_dict(),
                    "opt_state": self.opt.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                    "ema_state": {k: v.detach().cpu() for k, v in self.ema.shadow.items()},
                    "ema_decay": float(self.ema.decay),
                    "means": self.dataset.means,
                    "stds": self.dataset.stds,
                    "n_samples": len(self.dataset),
                    "loss_history_mse": np.array(losses_mse),
                    "loss_history_integral": np.array(losses_integral),
                    "timestamp": datetime.today().strftime('%Y-%m-%d %H:%M'),
                }

                # save to disk
                path_save = self.checkpoint_path(epoch)
                tmp = path_save.with_suffix(".pt.tmp")
                torch.save(ckpt, tmp)
                tmp.replace(path_save)

        logging.info("Training finished.")

    
    def checkpoint_path(self, epoch):
        """Path where checkpoint .pt file is saved for a specific epoch."""
        return Path(self.save_dir) / f"ckpt_epoch{str(epoch).zfill(3)}.pt"

    
    def load_checkpoint(self, path):
        """Load previously saved checkpoint to continue training from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # make sure the training data is unchanged
        # (only verify number of samples, full check would be more complex)
        assert ckpt['n_samples'] == len(self.dataset)
        assert ckpt['means'] == self.dataset.means
        assert ckpt['stds'] == self.dataset.stds

        # UNet and optimizer state
        self.ddpm.model.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["opt_state"])
    
        # AMP scaler
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
    
        # EMA (move back to device)
        if "ema_state" in ckpt:
            self.ema.decay = float(ckpt.get("ema_decay", self.ema.decay))
            # ensure device
            self.ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema_state"].items()}
    
        start_epoch = int(ckpt['epoch'])
        global_step = int(ckpt['global_step'])
    
        return start_epoch, global_step