from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import Dataset

from .geometry import random_smooth_flow, valid_flow_mask, warp_image


@dataclass(frozen=True)
class IAMWordRecord:
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class IAMLineRecord:
    line_id: str
    writer_id: str
    transcription: str
    form_path: Path
    bbox: tuple[int, int, int, int]
    words: tuple[IAMWordRecord, ...] = ()


def build_iam_line_records(iam_root: str | Path, *, only_valid: bool = True) -> list[IAMLineRecord]:
    """Build a line index directly from IAM form images and XML annotations."""
    iam_root = Path(iam_root)
    xml_dir = iam_root / "xml"
    forms_dir = iam_root / "forms"
    if not xml_dir.is_dir() or not forms_dir.is_dir():
        raise FileNotFoundError(f"Expected IAM xml/ and forms/ directories under {iam_root}")

    records: list[IAMLineRecord] = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        form_id = root.attrib.get("id", xml_path.stem)
        writer_id = root.attrib.get("writer-id", "unknown")
        form_path = forms_dir / f"{form_id}.png"
        if not form_path.exists():
            jpg_path = forms_dir / f"{form_id}.jpg"
            if not jpg_path.exists():
                continue
            form_path = jpg_path

        for line in root.findall("./handwritten-part/line"):
            if only_valid and line.attrib.get("segmentation", "ok") != "ok":
                continue
            components = line.findall(".//cmp")
            if not components:
                continue
            xs = [int(component.attrib["x"]) for component in components]
            ys = [int(component.attrib["y"]) for component in components]
            x2s = [
                int(component.attrib["x"]) + int(component.attrib["width"])
                for component in components
            ]
            y2s = [
                int(component.attrib["y"]) + int(component.attrib["height"])
                for component in components
            ]
            words: list[IAMWordRecord] = []
            for word in line.findall("./word"):
                word_components = word.findall(".//cmp")
                if not word_components:
                    continue
                word_xs = [int(component.attrib["x"]) for component in word_components]
                word_ys = [int(component.attrib["y"]) for component in word_components]
                word_x2s = [
                    int(component.attrib["x"]) + int(component.attrib["width"])
                    for component in word_components
                ]
                word_y2s = [
                    int(component.attrib["y"]) + int(component.attrib["height"])
                    for component in word_components
                ]
                words.append(
                    IAMWordRecord(
                        text=word.attrib.get("text", ""),
                        bbox=(min(word_xs), min(word_ys), max(word_x2s), max(word_y2s)),
                    )
                )
            records.append(
                IAMLineRecord(
                    line_id=line.attrib["id"],
                    writer_id=writer_id,
                    transcription=line.attrib.get("text", ""),
                    form_path=form_path,
                    bbox=(min(xs), min(ys), max(x2s), max(y2s)),
                    words=tuple(words),
                )
            )
    if not records:
        raise RuntimeError(f"No valid IAM line records found under {iam_root}")
    return records


def split_records_by_writer(
    records: Sequence[IAMLineRecord],
    *,
    seed: int = 17,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, list[IAMLineRecord]]:
    """Create writer-disjoint train/validation/test splits."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a non-empty test fraction")
    writers = sorted({record.writer_id for record in records})
    random.Random(seed).shuffle(writers)
    train_end = int(len(writers) * train_fraction)
    val_end = train_end + int(len(writers) * val_fraction)
    writer_split = {
        **{writer: "train" for writer in writers[:train_end]},
        **{writer: "val" for writer in writers[train_end:val_end]},
        **{writer: "test" for writer in writers[val_end:]},
    }
    splits: dict[str, list[IAMLineRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        splits[writer_split[record.writer_id]].append(record)
    return splits


def split_statistics(splits: dict[str, Sequence[IAMLineRecord]]) -> dict[str, dict[str, float]]:
    """Return line/writer counts and word-level OOV rates for each split."""
    train_words = {
        token.casefold()
        for record in splits["train"]
        for token in record.transcription.split()
        if token
    }
    statistics: dict[str, dict[str, float]] = {}
    for name, records in splits.items():
        tokens = [
            token.casefold()
            for record in records
            for token in record.transcription.split()
            if token
        ]
        oov = sum(token not in train_words for token in tokens) if name != "train" else 0
        statistics[name] = {
            "lines": float(len(records)),
            "writers": float(len({record.writer_id for record in records})),
            "tokens": float(len(tokens)),
            "oov_rate": float(oov / max(len(tokens), 1)),
        }
    return statistics


def _random_scalar(
    low: float,
    high: float,
    *,
    generator: torch.Generator | None,
) -> float:
    return float(torch.rand((), generator=generator).item() * (high - low) + low)


def _photometric_augmentation(image: Tensor, generator: torch.Generator | None) -> Tensor:
    contrast = _random_scalar(0.75, 1.25, generator=generator)
    brightness = _random_scalar(-0.06, 0.06, generator=generator)
    gamma = _random_scalar(0.75, 1.35, generator=generator)
    augmented = ((image - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)
    augmented = augmented.clamp_min(1e-4).pow(gamma)

    if _random_scalar(0.0, 1.0, generator=generator) < 0.35:
        ink = 1.0 - augmented
        if _random_scalar(0.0, 1.0, generator=generator) < 0.5:
            ink = F.max_pool2d(ink.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        else:
            ink = -F.max_pool2d(-ink.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        augmented = 1.0 - ink
    if _random_scalar(0.0, 1.0, generator=generator) < 0.3:
        augmented = F.avg_pool2d(augmented.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)

    noise_sigma = _random_scalar(0.0, 0.025, generator=generator)
    if noise_sigma > 0:
        noise = torch.randn(
            augmented.shape,
            dtype=augmented.dtype,
            device=augmented.device,
            generator=generator,
        )
        augmented = augmented + noise * noise_sigma
    return augmented.clamp(0.0, 1.0)


class IAMRegistrationDataset(Dataset[dict[str, Tensor | str]]):
    """Generate paired line images and exact synthetic registration ground truth."""

    def __init__(
        self,
        records: Sequence[IAMLineRecord],
        *,
        image_size: tuple[int, int] = (96, 1024),
        training: bool,
        seed: int = 17,
        crop_padding: int = 20,
        max_translation: float = 48.0,
        elastic_amplitude: float = 10.0,
        samples_per_record: int = 1,
        identity_probability: float = 0.0,
    ) -> None:
        if not records:
            raise ValueError("IAMRegistrationDataset requires at least one line record")
        self.records = list(records)
        self.image_size = image_size
        self.training = training
        self.seed = seed
        self.crop_padding = crop_padding
        self.max_translation = max_translation
        self.elastic_amplitude = elastic_amplitude
        self.samples_per_record = samples_per_record
        if not 0.0 <= identity_probability <= 1.0:
            raise ValueError("identity_probability must be between zero and one")
        self.identity_probability = identity_probability

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_record

    def _load_line(self, record: IAMLineRecord, generator: torch.Generator | None) -> Tensor:
        with Image.open(record.form_path) as form:
            form = form.convert("L")
            x1, y1, x2, y2 = record.bbox
            crop = form.crop(
                (
                    max(0, x1 - self.crop_padding),
                    max(0, y1 - self.crop_padding),
                    min(form.width, x2 + self.crop_padding),
                    min(form.height, y2 + self.crop_padding),
                )
            )
            array = np.asarray(crop, dtype=np.float32).copy() / 255.0
        line = torch.from_numpy(array).unsqueeze(0)
        return self._fit_canvas(line, generator)

    def _fit_canvas(self, line: Tensor, generator: torch.Generator | None) -> Tensor:
        target_height, target_width = self.image_size
        source_height, source_width = line.shape[-2:]
        scale = target_height / max(source_height, 1)
        resized_width = max(1, int(round(source_width * scale)))
        line = F.interpolate(
            line.unsqueeze(0),
            size=(target_height, resized_width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        if resized_width > target_width:
            max_start = resized_width - target_width
            if self.training:
                start = int(torch.randint(max_start + 1, (), generator=generator).item())
            else:
                start = max_start // 2
            line = line[:, :, start : start + target_width]
        elif resized_width < target_width:
            missing = target_width - resized_width
            if self.training:
                left = int(torch.randint(missing + 1, (), generator=generator).item())
            else:
                left = missing // 2
            line = F.pad(line, (left, missing - left), value=1.0)
        return line.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index % len(self.records)]
        generator = None
        if not self.training:
            generator = torch.Generator().manual_seed(self.seed + index)

        base = self._load_line(record, generator)
        is_identity = (
            self.identity_probability > 0
            and _random_scalar(0.0, 1.0, generator=generator) < self.identity_probability
        )
        if is_identity:
            ground_truth_flow = torch.zeros(
                2, self.image_size[0], self.image_size[1], dtype=base.dtype, device=base.device
            )
        else:
            ground_truth_flow = random_smooth_flow(
                1,
                self.image_size[0],
                self.image_size[1],
                max_translation=self.max_translation,
                elastic_amplitude=self.elastic_amplitude,
                device=base.device,
                dtype=base.dtype,
                generator=generator,
            ).squeeze(0)
        source = _photometric_augmentation(base, generator)
        target_geometry = warp_image(base.unsqueeze(0), ground_truth_flow.unsqueeze(0)).squeeze(0)
        target = _photometric_augmentation(target_geometry, generator)
        valid_mask = valid_flow_mask(ground_truth_flow.unsqueeze(0)).squeeze(0)
        return {
            "source": source,
            "target": target,
            "flow": ground_truth_flow,
            "valid_mask": valid_mask,
            "photometric_mask": torch.ones_like(valid_mask),
            "line_id": record.line_id,
            "writer_id": record.writer_id,
            "transcription": record.transcription,
            "is_identity": torch.tensor(is_identity),
        }


def split_vocabulary(
    words: Sequence[str],
    *,
    seed: int = 17,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, list[str]]:
    """Split word identities before rendering so test words are truly unseen."""
    unique_words = sorted({word.strip() for word in words if word.strip()})
    random.Random(seed).shuffle(unique_words)
    train_end = int(len(unique_words) * train_fraction)
    val_end = train_end + int(len(unique_words) * val_fraction)
    return {
        "train": unique_words[:train_end],
        "val": unique_words[train_end:val_end],
        "test": unique_words[val_end:],
    }


def load_vocabulary_splits(words_file: str | Path, *, seed: int = 17) -> dict[str, list[str]]:
    words = Path(words_file).read_text(encoding="utf-8").splitlines()
    return split_vocabulary(words, seed=seed)


class SyntheticCrossFontRegistrationDataset(Dataset[dict[str, Tensor | str]]):
    """Pairs identical unseen text rendered in different handwriting-like fonts.

    Words occupy shared semantic cells, so the known synthetic warp remains a valid
    spatial target even though source and target pixels have different glyph styles.
    Photometric loss is disabled for these pairs; flow supervision remains exact.
    """

    def __init__(
        self,
        words: Sequence[str],
        font_paths: Sequence[str | Path],
        *,
        length: int,
        image_size: tuple[int, int] = (96, 512),
        training: bool,
        seed: int = 17,
        min_words: int = 3,
        max_words: int = 5,
        max_translation: float = 48.0,
        elastic_amplitude: float = 10.0,
    ) -> None:
        self.words = list(words)
        self.font_paths = [Path(path) for path in font_paths]
        self.length = length
        self.image_size = image_size
        self.training = training
        self.seed = seed
        self.min_words = min_words
        self.max_words = max_words
        self.max_translation = max_translation
        self.elastic_amplitude = elastic_amplitude
        if len(self.words) < max_words:
            raise ValueError("Not enough vocabulary entries for synthetic line generation")
        if len(self.font_paths) < 2 or any(not path.is_file() for path in self.font_paths):
            raise FileNotFoundError("At least two valid font files are required")

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _randint(high: int, generator: torch.Generator | None) -> int:
        return int(torch.randint(high, (), generator=generator).item())

    def _render(self, words: Sequence[str], font_path: Path) -> Tensor:
        height, width = self.image_size
        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)
        cell_width = width / len(words)
        for index, word in enumerate(words):
            max_text_width = cell_width * 0.88
            font_size = max(12, int(height * 0.62))
            while font_size > 12:
                font = ImageFont.truetype(str(font_path), font_size)
                bbox = draw.textbbox((0, 0), word, font=font)
                if bbox[2] - bbox[0] <= max_text_width:
                    break
                font_size -= 2
            font = ImageFont.truetype(str(font_path), font_size)
            bbox = draw.textbbox((0, 0), word, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            center_x = (index + 0.5) * cell_width
            x = center_x - text_width / 2 - bbox[0]
            y = height / 2 - text_height / 2 - bbox[1]
            draw.text((x, y), word, font=font, fill=0)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        generator = None
        if not self.training:
            generator = torch.Generator().manual_seed(self.seed + index)
        word_count = self.min_words + self._randint(
            self.max_words - self.min_words + 1, generator
        )
        indices = torch.randperm(len(self.words), generator=generator)[:word_count].tolist()
        selected_words = [self.words[word_index] for word_index in indices]
        source_font_index = self._randint(len(self.font_paths), generator)
        target_font_index = self._randint(len(self.font_paths) - 1, generator)
        if target_font_index >= source_font_index:
            target_font_index += 1
        source_base = self._render(selected_words, self.font_paths[source_font_index])
        target_base = self._render(selected_words, self.font_paths[target_font_index])
        flow = random_smooth_flow(
            1,
            self.image_size[0],
            self.image_size[1],
            max_translation=self.max_translation,
            elastic_amplitude=self.elastic_amplitude,
            device=source_base.device,
            dtype=source_base.dtype,
            generator=generator,
        ).squeeze(0)
        source = _photometric_augmentation(source_base, generator)
        target = warp_image(target_base.unsqueeze(0), flow.unsqueeze(0)).squeeze(0)
        target = _photometric_augmentation(target, generator)
        valid_mask = valid_flow_mask(flow.unsqueeze(0)).squeeze(0)
        return {
            "source": source,
            "target": target,
            "flow": flow,
            "valid_mask": valid_mask,
            "photometric_mask": torch.zeros_like(valid_mask),
            "line_id": f"synthetic-{index}",
            "writer_id": f"{self.font_paths[source_font_index].stem}->{self.font_paths[target_font_index].stem}",
            "transcription": " ".join(selected_words),
            "is_identity": torch.tensor(False),
        }


def records_to_jsonable(records: Iterable[IAMLineRecord]) -> list[dict[str, object]]:
    return [
        {
            "line_id": record.line_id,
            "writer_id": record.writer_id,
            "transcription": record.transcription,
            "form_path": str(record.form_path),
            "bbox": list(record.bbox),
            "words": [
                {"text": word.text, "bbox": list(word.bbox)} for word in record.words
            ],
        }
        for record in records
    ]
