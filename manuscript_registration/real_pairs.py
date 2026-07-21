from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .data import IAMLineRecord


@dataclass(frozen=True)
class IAMWordLandmark:
    index: int
    text: str
    x: float
    y: float


def normalized_transcription(text: str) -> str:
    return " ".join(text.casefold().split())


def find_cross_writer_line_pairs(
    records: Sequence[IAMLineRecord],
    *,
    minimum_words: int = 3,
    maximum_pairs: int | None = None,
) -> list[tuple[IAMLineRecord, IAMLineRecord]]:
    """Find real IAM lines with identical transcriptions from different writers."""
    grouped: dict[str, list[IAMLineRecord]] = defaultdict(list)
    for record in records:
        key = normalized_transcription(record.transcription)
        if key and len(record.words) >= minimum_words:
            grouped[key].append(record)

    pairs: list[tuple[IAMLineRecord, IAMLineRecord]] = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda item: (item.writer_id, item.line_id))
        for source, target in combinations(candidates, 2):
            if source.writer_id == target.writer_id or len(source.words) != len(target.words):
                continue
            pairs.append((source, target))
            if maximum_pairs is not None and len(pairs) >= maximum_pairs:
                return pairs
    return pairs


def load_normalized_iam_line(
    record: IAMLineRecord,
    *,
    image_size: tuple[int, int],
    crop_padding: int = 20,
) -> tuple[Tensor, list[IAMWordLandmark]]:
    """Load an IAM line and map annotated word centers onto the model canvas."""
    with Image.open(record.form_path) as form_image:
        form = form_image.convert("L")
        x1, y1, x2, y2 = record.bbox
        crop_left = max(0, x1 - crop_padding)
        crop_top = max(0, y1 - crop_padding)
        crop_right = min(form.width, x2 + crop_padding)
        crop_bottom = min(form.height, y2 + crop_padding)
        crop = form.crop((crop_left, crop_top, crop_right, crop_bottom))
        array = np.asarray(crop, dtype=np.float32).copy() / 255.0

    line = torch.from_numpy(array).unsqueeze(0)
    target_height, target_width = image_size
    source_height, source_width = line.shape[-2:]
    scale = target_height / max(source_height, 1)
    resized_width = max(1, int(round(source_width * scale)))
    line = F.interpolate(
        line.unsqueeze(0),
        size=(target_height, resized_width),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)

    x_scale = (resized_width - 1) / max(source_width - 1, 1)
    y_scale = (target_height - 1) / max(source_height - 1, 1)
    if resized_width > target_width:
        horizontal_offset = -((resized_width - target_width) // 2)
        line = line[:, :, -horizontal_offset : -horizontal_offset + target_width]
    else:
        horizontal_offset = (target_width - resized_width) // 2
        line = F.pad(
            line,
            (horizontal_offset, target_width - resized_width - horizontal_offset),
            value=1.0,
        )

    landmarks: list[IAMWordLandmark] = []
    for index, word in enumerate(record.words):
        word_x1, word_y1, word_x2, word_y2 = word.bbox
        center_x = ((word_x1 + word_x2) / 2 - crop_left) * x_scale + horizontal_offset
        center_y = ((word_y1 + word_y2) / 2 - crop_top) * y_scale
        if 0 <= center_x <= target_width - 1 and 0 <= center_y <= target_height - 1:
            landmarks.append(IAMWordLandmark(index, word.text, center_x, center_y))
    return line.clamp(0.0, 1.0), landmarks
