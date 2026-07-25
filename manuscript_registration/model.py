from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .geometry import resize_flow, warp_image


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs) + self.skip(inputs)


class SharedPyramidEncoder(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.level1 = ConvBlock(1, base_channels, stride=2)
        self.level2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.level3 = ConvBlock(base_channels * 2, base_channels * 3, stride=2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        level1 = self.level1(image)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        return level1, level2, level3


class HorizontalPatchCorrelation(nn.Module):
    """Global target-to-source attention over vertical text-line patches."""

    def __init__(self, temperature: float = 0.08, position_weight: float = 0.15) -> None:
        super().__init__()
        self.temperature = temperature
        self.position_weight = position_weight

    def forward(self, source_features: Tensor, target_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if source_features.shape != target_features.shape:
            raise ValueError("Source and target feature maps must have the same shape")

        # Each feature-map column is one overlapping vertical patch descriptor.
        source_columns = F.normalize(source_features.mean(dim=2), dim=1)
        target_columns = F.normalize(target_features.mean(dim=2), dim=1)
        similarity = torch.einsum("bct,bcs->bts", target_columns, source_columns)

        width = source_features.shape[-1]
        coordinates = torch.linspace(-1.0, 1.0, width, device=similarity.device, dtype=similarity.dtype)
        positional_cost = (coordinates[:, None] - coordinates[None, :]).square()
        logits = similarity / self.temperature - self.position_weight * positional_cost.unsqueeze(0)
        attention = logits.softmax(dim=-1)

        source_indices = torch.arange(width, device=similarity.device, dtype=similarity.dtype)
        expected_source = torch.einsum("bts,s->bt", attention, source_indices)
        target_indices = source_indices.unsqueeze(0)
        horizontal_flow = expected_source - target_indices
        confidence = attention.amax(dim=-1)
        return horizontal_flow, confidence, similarity


@dataclass
class RegistrationOutput:
    aligned: Tensor
    flow: Tensor
    coarse_flow: Tensor
    confidence: Tensor
    similarity: Tensor


class PatchCorrelationRegistration(nn.Module):
    """Coarse-to-fine text-line registration with a differentiable spatial transformer.

    Flow uses the backward-warp convention: at every target pixel it stores the
    displacement to the source pixel that should be sampled.
    """

    def __init__(
        self,
        base_channels: int = 32,
        max_residual_pixels: float = 48.0,
        correlation_temperature: float = 0.08,
    ) -> None:
        super().__init__()
        self.encoder = SharedPyramidEncoder(base_channels)
        bottleneck_channels = base_channels * 3
        self.correlation = HorizontalPatchCorrelation(temperature=correlation_temperature)
        self.max_residual_pixels = max_residual_pixels
        # Identity is a safe initial registration. The model first learns how much
        # to trust the global patch correspondence before it can create large warps.
        self.coarse_gain = nn.Parameter(torch.tensor(0.0))

        self.bottleneck = ConvBlock(bottleneck_channels * 2 + 2, base_channels * 4)
        self.decode2 = ConvBlock(base_channels * 4 + base_channels * 4, base_channels * 3)
        self.decode1 = ConvBlock(base_channels * 3 + base_channels * 2, base_channels * 2)
        self.decode0 = ConvBlock(base_channels * 2 + 2, base_channels)
        self.flow_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, 2, 3, padding=1),
        )

        # Begin with correlation-only motion; the decoder learns residual refinement.
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def forward(self, source: Tensor, target: Tensor) -> RegistrationOutput:
        if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 1:
            raise ValueError("Expected source and target tensors with equal Bx1xHxW shapes")

        source1, source2, source3 = self.encoder(source)
        target1, target2, target3 = self.encoder(target)
        column_flow, confidence, similarity = self.correlation(source3, target3)

        coarse_flow = torch.zeros(
            source3.shape[0],
            2,
            source3.shape[2],
            source3.shape[3],
            device=source3.device,
            dtype=source3.dtype,
        )
        correlation_gain = torch.tanh(self.coarse_gain)
        coarse_flow[:, 0] = (
            column_flow.unsqueeze(1).expand(-1, source3.shape[2], -1) * correlation_gain
        )
        confidence_map = confidence.unsqueeze(1).unsqueeze(2).expand(-1, 1, source3.shape[2], -1)
        warped_source3 = warp_image(source3, coarse_flow)

        decoded3 = self.bottleneck(
            torch.cat((warped_source3, target3, coarse_flow[:, :1], confidence_map), dim=1)
        )
        decoded2 = F.interpolate(decoded3, size=source2.shape[-2:], mode="bilinear", align_corners=True)
        decoded2 = self.decode2(torch.cat((decoded2, source2, target2), dim=1))
        decoded1 = F.interpolate(decoded2, size=source1.shape[-2:], mode="bilinear", align_corners=True)
        decoded1 = self.decode1(torch.cat((decoded1, source1, target1), dim=1))
        decoded0 = F.interpolate(decoded1, size=source.shape[-2:], mode="bilinear", align_corners=True)
        decoded0 = self.decode0(torch.cat((decoded0, source, target), dim=1))

        residual_flow = torch.tanh(self.flow_head(decoded0)) * self.max_residual_pixels
        full_coarse_flow = resize_flow(coarse_flow, source.shape[-2:])
        flow = full_coarse_flow + residual_flow
        aligned = warp_image(source, flow)
        return RegistrationOutput(
            aligned=aligned,
            flow=flow,
            coarse_flow=full_coarse_flow,
            confidence=confidence,
            similarity=similarity,
        )
