from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import RegistrationOutput


def masked_mean(values: Tensor, mask: Tensor, epsilon: float = 1e-6) -> Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(epsilon)


def ssim_map(first: Tensor, second: Tensor, window_size: int = 7) -> Tensor:
    """Differentiable single-channel SSIM map in [approximately -1, 1]."""
    padding = window_size // 2
    mu_first = F.avg_pool2d(first, window_size, stride=1, padding=padding)
    mu_second = F.avg_pool2d(second, window_size, stride=1, padding=padding)
    variance_first = F.avg_pool2d(first.square(), window_size, stride=1, padding=padding) - mu_first.square()
    variance_second = F.avg_pool2d(second.square(), window_size, stride=1, padding=padding) - mu_second.square()
    covariance = F.avg_pool2d(first * second, window_size, stride=1, padding=padding) - mu_first * mu_second
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_first * mu_second + c1) * (2 * covariance + c2)
    denominator = (mu_first.square() + mu_second.square() + c1) * (
        variance_first + variance_second + c2
    )
    return numerator / denominator.clamp_min(1e-6)


@dataclass
class RegistrationLossBreakdown:
    total: Tensor
    flow: Tensor
    coarse_flow: Tensor
    photometric: Tensor
    ssim: Tensor
    smoothness: Tensor
    monotonicity: Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in self.__dict__.items()
        }


class RegistrationLoss(nn.Module):
    def __init__(
        self,
        *,
        flow_weight: float = 1.0,
        coarse_flow_weight: float = 0.35,
        photometric_weight: float = 0.5,
        ssim_weight: float = 0.25,
        smoothness_weight: float = 0.03,
        monotonicity_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.coarse_flow_weight = coarse_flow_weight
        self.photometric_weight = photometric_weight
        self.ssim_weight = ssim_weight
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight

    def forward(
        self,
        output: RegistrationOutput,
        target: Tensor,
        ground_truth_flow: Tensor,
        valid_mask: Tensor,
        photometric_mask: Tensor | None = None,
    ) -> RegistrationLossBreakdown:
        flow_error = (output.flow - ground_truth_flow).abs().mean(dim=1, keepdim=True)
        flow_loss = masked_mean(flow_error, valid_mask)
        # The global column correlation estimates horizontal motion only. Vertical
        # and local elastic corrections are deliberately handled by the decoder.
        coarse_error = (output.coarse_flow[:, :1] - ground_truth_flow[:, :1]).abs()
        coarse_flow_loss = masked_mean(coarse_error, valid_mask)

        ink_weight = 0.15 + 0.85 * (1.0 - target)
        if photometric_mask is None:
            photometric_mask = torch.ones_like(valid_mask)
        weighted_valid = valid_mask * ink_weight * photometric_mask
        charbonnier = torch.sqrt((output.aligned - target).square() + 1e-6)
        # masked_mean safely returns zero for cross-font-only batches, whose
        # photometric mask is all zero.
        photometric_loss = masked_mean(charbonnier, weighted_valid)
        structural_loss = masked_mean(1.0 - ssim_map(output.aligned, target), weighted_valid)

        dx = output.flow[:, :, :, 1:] - output.flow[:, :, :, :-1]
        dy = output.flow[:, :, 1:, :] - output.flow[:, :, :-1, :]
        smoothness_loss = dx.abs().mean() + dy.abs().mean()

        # x + u_x must increase along the text line; otherwise the warp folds/reorders text.
        horizontal_step = 1.0 + output.flow[:, 0, :, 1:] - output.flow[:, 0, :, :-1]
        monotonicity_loss = F.relu(0.05 - horizontal_step).mean()

        total = (
            self.flow_weight * flow_loss
            + self.coarse_flow_weight * coarse_flow_loss
            + self.photometric_weight * photometric_loss
            + self.ssim_weight * structural_loss
            + self.smoothness_weight * smoothness_loss
            + self.monotonicity_weight * monotonicity_loss
        )
        return RegistrationLossBreakdown(
            total=total,
            flow=flow_loss,
            coarse_flow=coarse_flow_loss,
            photometric=photometric_loss,
            ssim=structural_loss,
            smoothness=smoothness_loss,
            monotonicity=monotonicity_loss,
        )
