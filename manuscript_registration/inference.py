from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import Tensor

from .model import PatchCorrelationRegistration, RegistrationOutput
from .geometry import warp_image


def load_registration_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[PatchCorrelationRegistration, dict[str, object]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PatchCorrelationRegistration(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def prepare_image_pair(
    source: Image.Image,
    target: Image.Image,
    *,
    target_height: int = 96,
    training_width: int = 1024,
    maximum_width: int = 2048,
) -> tuple[Tensor, Tensor]:
    """Normalize two full text lines to one shared, stride-compatible canvas."""
    source = ImageOps.autocontrast(source.convert("L"))
    target = ImageOps.autocontrast(target.convert("L"))
    source_tensor = _pil_to_tensor(source)
    target_tensor = _pil_to_tensor(target)

    source_width = max(1, round(source.width * target_height / max(source.height, 1)))
    target_width = max(1, round(target.width * target_height / max(target.height, 1)))
    canvas_width = max(source_width, target_width, training_width)
    common_scale = min(1.0, maximum_width / canvas_width)
    output_height = max(8, round(target_height * common_scale))
    source_width = max(1, round(source_width * common_scale))
    target_width = max(1, round(target_width * common_scale))
    canvas_width = min(maximum_width, max(source_width, target_width, training_width))
    canvas_height = target_height
    canvas_width = int((canvas_width + 7) // 8 * 8)

    def fit(image_tensor: Tensor, width: int) -> Tensor:
        resized = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(output_height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        left = (canvas_width - width) // 2
        right = canvas_width - width - left
        top = (canvas_height - output_height) // 2
        bottom = canvas_height - output_height - top
        return F.pad(resized, (left, right, top, bottom), value=1.0)

    return fit(source_tensor, source_width), fit(target_tensor, target_width)


@torch.no_grad()
def align_tensor_pair_tiled(
    model: PatchCorrelationRegistration,
    source: Tensor,
    target: Tensor,
    *,
    tile_width: int,
    overlap: int | None = None,
) -> RegistrationOutput:
    """Register a long line in overlapping training-width blocks.

    Each tile predicts target-to-source flow in its own local coordinate frame.
    Since paired tiles share the same global start coordinate, the displacement
    values remain valid on the full canvas. Hann blending suppresses tile seams.
    """
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("Expected source and target tensors with equal BxCxHxW shapes")
    if tile_width <= 0 or tile_width % 8:
        raise ValueError("tile_width must be a positive multiple of eight")
    full_width = source.shape[-1]
    if full_width <= tile_width:
        return model(source, target)

    overlap = tile_width // 2 if overlap is None else overlap
    if overlap < 0 or overlap >= tile_width:
        raise ValueError("overlap must be non-negative and smaller than tile_width")
    stride = tile_width - overlap
    starts = list(range(0, full_width - tile_width + 1, stride))
    final_start = full_width - tile_width
    if starts[-1] != final_start:
        starts.append(final_start)

    flow_total = torch.zeros(
        source.shape[0], 2, source.shape[-2], full_width, device=source.device, dtype=source.dtype
    )
    coarse_total = torch.zeros_like(flow_total)
    weight_total = torch.zeros(
        source.shape[0], 1, source.shape[-2], full_width, device=source.device, dtype=source.dtype
    )
    horizontal_weight = torch.hann_window(
        tile_width, periodic=False, device=source.device, dtype=source.dtype
    ).clamp_min(0.05).view(1, 1, 1, -1)
    tile_outputs: list[RegistrationOutput] = []
    for start in starts:
        stop = start + tile_width
        tile_output = model(source[..., start:stop], target[..., start:stop])
        tile_outputs.append(tile_output)
        flow_total[..., start:stop] += tile_output.flow * horizontal_weight
        coarse_total[..., start:stop] += tile_output.coarse_flow * horizontal_weight
        weight_total[..., start:stop] += horizontal_weight

    flow = flow_total / weight_total.clamp_min(1e-6)
    coarse_flow = coarse_total / weight_total.clamp_min(1e-6)
    confidence = torch.cat([output.confidence for output in tile_outputs], dim=1)
    similarity = torch.stack(
        [
            torch.block_diag(*[output.similarity[index] for output in tile_outputs])
            for index in range(source.shape[0])
        ]
    )
    return RegistrationOutput(
        aligned=warp_image(source, flow),
        flow=flow,
        coarse_flow=coarse_flow,
        confidence=confidence,
        similarity=similarity,
    )


def ink_bbox_affine_prealign(
    source: Tensor,
    target: Tensor,
    *,
    ink_threshold: float = 0.01,
    maximum_scale_ratio: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Globally align source/target ink extents before learned local registration."""
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 1:
        raise ValueError("Expected source and target tensors with equal Bx1xHxW shapes")
    batch_size, _, height, width = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    flows: list[Tensor] = []

    def bounds(image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        ink = (1.0 - image).clamp(0.0, 1.0)
        columns = ink.mean(dim=(0, 1))
        rows = ink.mean(dim=(0, 2))
        x_indices = torch.where(columns > ink_threshold)[0]
        y_indices = torch.where(rows > ink_threshold)[0]
        if x_indices.numel() == 0 or y_indices.numel() == 0:
            return None
        return x_indices[0], y_indices[0], x_indices[-1], y_indices[-1]

    for index in range(batch_size):
        source_bounds = bounds(source[index])
        target_bounds = bounds(target[index])
        if source_bounds is None or target_bounds is None:
            flows.append(torch.zeros(2, height, width, device=source.device, dtype=source.dtype))
            continue
        source_x1, source_y1, source_x2, source_y2 = source_bounds
        target_x1, target_y1, target_x2, target_y2 = target_bounds
        scale_x = ((source_x2 - source_x1) / (target_x2 - target_x1).clamp_min(1)).clamp(
            1.0 / maximum_scale_ratio, maximum_scale_ratio
        )
        scale_y = ((source_y2 - source_y1) / (target_y2 - target_y1).clamp_min(1)).clamp(
            1.0 / maximum_scale_ratio, maximum_scale_ratio
        )
        mapped_x = source_x1 + (xx - target_x1) * scale_x
        mapped_y = source_y1 + (yy - target_y1) * scale_y
        flows.append(torch.stack((mapped_x - xx, mapped_y - yy)))
    flow = torch.stack(flows)
    return warp_image(source, flow), flow


def compose_backward_flows(first_flow: Tensor, second_flow: Tensor) -> Tensor:
    """Compose source-to-intermediate and intermediate-to-target backward flows."""
    if first_flow.shape != second_flow.shape:
        raise ValueError("Flows must have identical Bx2xHxW shapes")
    return second_flow + warp_image(first_flow, second_flow)


@torch.no_grad()
def align_pil_images(
    model: PatchCorrelationRegistration,
    source: Image.Image,
    target: Image.Image,
    *,
    training_image_size: tuple[int, int],
    maximum_width: int = 2048,
    global_prealign: bool = True,
) -> tuple[RegistrationOutput, Tensor, Tensor]:
    source_tensor, target_tensor = prepare_image_pair(
        source,
        target,
        target_height=training_image_size[0],
        training_width=training_image_size[1],
        maximum_width=maximum_width,
    )
    device = next(model.parameters()).device
    source_batch = source_tensor.unsqueeze(0).to(device)
    target_batch = target_tensor.unsqueeze(0).to(device)
    model_source = source_batch
    prealignment_flow = torch.zeros(
        source_batch.shape[0],
        2,
        source_batch.shape[-2],
        source_batch.shape[-1],
        device=source_batch.device,
        dtype=source_batch.dtype,
    )
    if global_prealign:
        model_source, prealignment_flow = ink_bbox_affine_prealign(source_batch, target_batch)
    residual_output = align_tensor_pair_tiled(
        model,
        model_source,
        target_batch,
        tile_width=training_image_size[1],
    )
    full_flow = compose_backward_flows(prealignment_flow, residual_output.flow)
    full_coarse_flow = compose_backward_flows(prealignment_flow, residual_output.coarse_flow)
    output = RegistrationOutput(
        aligned=warp_image(source_batch, full_flow),
        flow=full_flow,
        coarse_flow=full_coarse_flow,
        confidence=residual_output.confidence,
        similarity=residual_output.similarity,
    )
    return output, source_batch, target_batch


def tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().float().cpu().squeeze()
    array = (image.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def alignment_overlay(aligned: Tensor, target: Tensor) -> Image.Image:
    aligned_np = aligned.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    target_np = target.detach().float().cpu().squeeze().clamp(0.0, 1.0).numpy()
    aligned_ink = 1.0 - aligned_np
    target_ink = 1.0 - target_np
    rgb = np.ones((*aligned_np.shape, 3), dtype=np.float32)
    rgb[..., 0] -= target_ink
    rgb[..., 1] -= 0.5 * aligned_ink + 0.5 * target_ink
    rgb[..., 2] -= aligned_ink
    return Image.fromarray((rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def flow_visualization(flow: Tensor) -> Image.Image:
    flow_np = flow.detach().float().cpu().squeeze(0).numpy()
    dx, dy = flow_np[0], flow_np[1]
    scale = max(float(np.percentile(np.abs(flow_np), 98)), 1.0)
    red = 0.5 + 0.5 * np.clip(dx / scale, -1.0, 1.0)
    blue = 0.5 - 0.5 * np.clip(dx / scale, -1.0, 1.0)
    green = 0.5 + 0.5 * np.clip(dy / scale, -1.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
