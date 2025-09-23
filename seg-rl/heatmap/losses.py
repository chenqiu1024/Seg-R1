from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def ce_over_pixels(logits: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over all pixels treating each pixel as a class.

    Args:
        logits: [B,1,H,W] unnormalized scores
        target_xy: [B,2] target coordinates in pixels (x,y)
    Returns:
        scalar loss
    """
    b, _, h, w = logits.shape
    flat_logits = logits.view(b, -1)
    x = target_xy[:, 0].clamp(0, w - 1).long()
    y = target_xy[:, 1].clamp(0, h - 1).long()
    idx = y * w + x
    return F.cross_entropy(flat_logits, idx)


def gaussian_heatmap_targets(
    target_xy: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 3.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Create Gaussian target heatmaps centered at target_xy.

    Args:
        target_xy: [B,2] (x,y)
        height, width: output map size
        sigma: std of Gaussian in pixels
        normalize: make per-sample heatmap sum to 1
    Returns:
        heatmaps: [B,1,H,W]
    """
    device = target_xy.device
    b = target_xy.shape[0]
    xs = torch.arange(width, device=device).view(1, 1, 1, width).float()
    ys = torch.arange(height, device=device).view(1, 1, height, 1).float()
    x0 = target_xy[:, 0].view(b, 1, 1, 1)
    y0 = target_xy[:, 1].view(b, 1, 1, 1)
    dist2 = (xs - x0) ** 2 + (ys - y0) ** 2
    heat = torch.exp(-0.5 * dist2 / (sigma ** 2))
    if normalize:
        heat = heat / (heat.sum(dim=(2, 3), keepdim=True) + 1e-8)
    return heat


def kl_to_gaussian_targets(logits: torch.Tensor, target_xy: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """KL divergence between model distribution and Gaussian soft targets.

    Treat p as soft target (Gaussian), q as model softmax over pixels, compute KL(p || q).
    """
    b, _, h, w = logits.shape
    with torch.no_grad():
        p = gaussian_heatmap_targets(target_xy, h, w, sigma=sigma, normalize=True)  # [B,1,H,W]
        p = p.clamp_min(1e-12)
    log_q = F.log_softmax(logits.view(b, -1), dim=1).view(b, 1, h, w)
    kl = (p * (p.log() - log_q)).sum(dim=(1, 2, 3))
    return kl.mean()


