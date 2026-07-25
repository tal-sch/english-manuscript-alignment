from __future__ import annotations

import torch
from torch import Tensor

from .losses import masked_mean, ssim_map
from .model import RegistrationOutput


def registration_metrics(
    output: RegistrationOutput,
    target: Tensor,
    ground_truth_flow: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    endpoint_error = torch.linalg.vector_norm(output.flow - ground_truth_flow, dim=1, keepdim=True)
    epe = masked_mean(endpoint_error, valid_mask)
    accuracy_1px = masked_mean((endpoint_error < 1.0).to(target.dtype), valid_mask)
    accuracy_3px = masked_mean((endpoint_error < 3.0).to(target.dtype), valid_mask)
    accuracy_5px = masked_mean((endpoint_error < 5.0).to(target.dtype), valid_mask)
    mae = masked_mean((output.aligned - target).abs(), valid_mask)
    ssim = masked_mean(ssim_map(output.aligned, target), valid_mask)

    aligned_ink = output.aligned < 0.65
    target_ink = target < 0.65
    valid_boolean = valid_mask > 0.5
    intersection = (aligned_ink & target_ink & valid_boolean).sum(dtype=target.dtype)
    denominator = (
        (aligned_ink & valid_boolean).sum(dtype=target.dtype)
        + (target_ink & valid_boolean).sum(dtype=target.dtype)
    )
    ink_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return {
        "epe": epe,
        "accuracy_1px": accuracy_1px,
        "accuracy_3px": accuracy_3px,
        "accuracy_5px": accuracy_5px,
        "mae": mae,
        "ssim": ssim,
        "ink_dice": ink_dice,
    }


def identity_baseline_metrics(
    source: Tensor,
    target: Tensor,
    ground_truth_flow: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    zero_flow = torch.zeros_like(ground_truth_flow)
    endpoint_error = torch.linalg.vector_norm(zero_flow - ground_truth_flow, dim=1, keepdim=True)
    return {
        "identity_epe": masked_mean(endpoint_error, valid_mask),
        "identity_mae": masked_mean((source - target).abs(), valid_mask),
        "identity_ssim": masked_mean(ssim_map(source, target), valid_mask),
    }
