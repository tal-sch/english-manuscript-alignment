from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def identity_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a normalized ``grid_sample`` identity grid with shape BxHxWx2."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)


def pixel_flow_to_normalized(flow: Tensor) -> Tensor:
    """Convert Bx2xHxW flow in pixels to normalized grid offsets."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow shaped Bx2xHxW, got {tuple(flow.shape)}")
    _, _, height, width = flow.shape
    scale_x = 2.0 / max(width - 1, 1)
    scale_y = 2.0 / max(height - 1, 1)
    normalized = torch.empty_like(flow)
    normalized[:, 0] = flow[:, 0] * scale_x
    normalized[:, 1] = flow[:, 1] * scale_y
    return normalized


def normalized_grid_to_pixel_flow(grid: Tensor) -> Tensor:
    """Convert a normalized BxHxWx2 sampling grid to Bx2xHxW pixel flow."""
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"Expected grid shaped BxHxWx2, got {tuple(grid.shape)}")
    batch, height, width, _ = grid.shape
    base = identity_grid(batch, height, width, device=grid.device, dtype=grid.dtype)
    offset = grid - base
    flow = offset.permute(0, 3, 1, 2).contiguous()
    flow[:, 0] *= max(width - 1, 1) / 2.0
    flow[:, 1] *= max(height - 1, 1) / 2.0
    return flow


def flow_to_grid(flow: Tensor) -> Tensor:
    """Create a backward sampling grid from target-to-source pixel flow."""
    batch, _, height, width = flow.shape
    base = identity_grid(batch, height, width, device=flow.device, dtype=flow.dtype)
    return base + pixel_flow_to_normalized(flow).permute(0, 2, 3, 1)


def warp_image(
    image: Tensor,
    flow: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> Tensor:
    """Backward-warp ``image`` using target-to-source pixel ``flow``."""
    if image.shape[0] != flow.shape[0] or image.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            f"Image and flow must share batch/spatial dimensions: "
            f"{tuple(image.shape)} vs {tuple(flow.shape)}"
        )
    return F.grid_sample(
        image,
        flow_to_grid(flow),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def valid_flow_mask(flow: Tensor) -> Tensor:
    """Return Bx1xHxW mask for sampling locations that remain inside the image."""
    grid = flow_to_grid(flow)
    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )
    return valid.unsqueeze(1).to(flow.dtype)


def resize_flow(flow: Tensor, size: Tuple[int, int]) -> Tensor:
    """Resize a pixel-space flow field while preserving displacement units."""
    old_height, old_width = flow.shape[-2:]
    new_height, new_width = size
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    # Treat displacement as pixels-per-feature-cell. Using the resolution ratio
    # keeps the convention stable for the fully convolutional model at dynamic
    # input widths (the sub-pixel align_corners correction is negligible here).
    resized[:, 0] *= new_width / max(old_width, 1)
    resized[:, 1] *= new_height / max(old_height, 1)
    return resized


def _sample_uniform(
    batch_size: int,
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    values = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    return values * (high - low) + low


def random_smooth_flow(
    batch_size: int,
    height: int,
    width: int,
    *,
    max_translation: float = 32.0,
    max_rotation_degrees: float = 5.0,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    max_shear_degrees: float = 3.0,
    elastic_amplitude: float = 8.0,
    elastic_grid: Tuple[int, int] = (5, 12),
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample smooth target-to-source flows for synthetic registration training.

    The transform combines affine motion with a low-resolution elastic field. The
    returned flow is expressed in full-resolution pixels and can be passed directly
    to :func:`warp_image`.
    """
    angles = _sample_uniform(
        batch_size,
        -max_rotation_degrees,
        max_rotation_degrees,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * (math.pi / 180.0)
    shears = _sample_uniform(
        batch_size,
        -max_shear_degrees,
        max_shear_degrees,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * (math.pi / 180.0)
    scales = _sample_uniform(
        batch_size,
        scale_range[0],
        scale_range[1],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    translations_x = _sample_uniform(
        batch_size,
        -max_translation,
        max_translation,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    translations_y = _sample_uniform(
        batch_size,
        -0.25 * max_translation,
        0.25 * max_translation,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales
    tan_s = torch.tan(shears)
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a + tan_s
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = translations_x * (2.0 / max(width - 1, 1))
    theta[:, 1, 2] = translations_y * (2.0 / max(height - 1, 1))

    affine_grid = F.affine_grid(
        theta,
        size=(batch_size, 1, height, width),
        align_corners=True,
    )

    elastic = torch.randn(
        batch_size,
        2,
        elastic_grid[0],
        elastic_grid[1],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    elastic = F.interpolate(elastic, size=(height, width), mode="bicubic", align_corners=True)
    elastic = F.avg_pool2d(elastic, kernel_size=9, stride=1, padding=4)
    elastic = torch.tanh(elastic) * elastic_amplitude

    affine_flow = normalized_grid_to_pixel_flow(affine_grid)
    return affine_flow + elastic
