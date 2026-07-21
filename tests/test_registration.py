from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from manuscript_registration.data import (
    IAMLineRecord,
    IAMRegistrationDataset,
    SyntheticCrossFontRegistrationDataset,
    build_iam_line_records,
    split_records_by_writer,
)
from manuscript_registration.geometry import resize_flow, warp_image
from manuscript_registration.model import PatchCorrelationRegistration
from manuscript_registration.inference import (
    align_tensor_pair_tiled,
    compose_backward_flows,
    ink_bbox_affine_prealign,
)
from manuscript_registration.real_pairs import (
    find_cross_writer_line_pairs,
    load_normalized_iam_line,
)


def test_identity_warp_is_exact() -> None:
    image = torch.rand(2, 1, 24, 40)
    flow = torch.zeros(2, 2, 24, 40)
    warped = warp_image(image, flow)
    torch.testing.assert_close(warped, image, atol=1e-5, rtol=1e-5)


def test_horizontal_pixel_flow_uses_backward_sampling_convention() -> None:
    image = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(-1, -1, 4, -1)
    flow = torch.zeros(1, 2, 4, 8)
    flow[:, 0] = 1.0
    warped = warp_image(image, flow)
    torch.testing.assert_close(warped[..., :-1], image[..., 1:], atol=1e-5, rtol=1e-5)


def test_resize_flow_preserves_pixel_scale() -> None:
    flow = torch.ones(1, 2, 8, 16)
    resized = resize_flow(flow, (16, 32))
    assert resized.shape == (1, 2, 16, 32)
    torch.testing.assert_close(resized, torch.full_like(resized, 2.0))


def test_model_outputs_aligned_image_and_dense_flow() -> None:
    model = PatchCorrelationRegistration(base_channels=8, max_residual_pixels=4.0)
    source = torch.rand(2, 1, 64, 128, requires_grad=True)
    target = torch.rand(2, 1, 64, 128)
    output = model(source, target)
    assert output.aligned.shape == source.shape
    assert output.flow.shape == (2, 2, 64, 128)
    assert output.similarity.shape == (2, 16, 16)
    output.aligned.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tiled_inference_preserves_long_line_shape_and_identity_initialization() -> None:
    model = PatchCorrelationRegistration(base_channels=8, max_residual_pixels=4.0)
    source = torch.rand(1, 1, 64, 320)
    output = align_tensor_pair_tiled(model, source, source, tile_width=128, overlap=64)
    assert output.aligned.shape == source.shape
    assert output.flow.shape == (1, 2, 64, 320)
    assert torch.isfinite(output.flow).all()
    torch.testing.assert_close(output.flow, torch.zeros_like(output.flow), atol=1e-6, rtol=0)


def test_ink_bbox_affine_prealignment_matches_global_extents() -> None:
    source = torch.ones(1, 1, 32, 96)
    target = torch.ones_like(source)
    source[:, :, 8:24, 8:88] = 0
    target[:, :, 10:22, 20:76] = 0
    prealigned, flow = ink_bbox_affine_prealign(source, target)
    assert prealigned.shape == source.shape
    assert flow.shape == (1, 2, 32, 96)
    source_ink_columns = torch.where((1 - prealigned).mean(dim=(0, 1, 2)) > 0.01)[0]
    target_ink_columns = torch.where((1 - target).mean(dim=(0, 1, 2)) > 0.01)[0]
    assert abs(int(source_ink_columns[0]) - int(target_ink_columns[0])) <= 1
    assert abs(int(source_ink_columns[-1]) - int(target_ink_columns[-1])) <= 1


def test_backward_flow_composition_preserves_constant_translations() -> None:
    first = torch.zeros(1, 2, 16, 24)
    second = torch.zeros_like(first)
    first[:, 0] = 3
    second[:, 1] = -2
    composed = compose_backward_flows(first, second)
    torch.testing.assert_close(composed[:, 0], torch.full_like(composed[:, 0], 3))
    torch.testing.assert_close(composed[:, 1], torch.full_like(composed[:, 1], -2))


def _write_tiny_iam(root: Path) -> None:
    (root / "forms").mkdir(parents=True)
    (root / "xml").mkdir(parents=True)
    image = np.full((48, 160), 255, dtype=np.uint8)
    image[18:28, 20:140] = 20
    Image.fromarray(image).save(root / "forms" / "a01-000u.png")
    (root / "xml" / "a01-000u.xml").write_text(
        """<?xml version="1.0"?>
<form id="a01-000u" writer-id="007">
  <handwritten-part>
    <line id="a01-000u-00" segmentation="ok" text="unseen sentence">
      <word id="w0" text="unseen"><cmp x="20" y="18" width="50" height="10" /></word>
      <word id="w1" text="sentence"><cmp x="80" y="18" width="60" height="10" /></word>
    </line>
  </handwritten-part>
</form>
""",
        encoding="utf-8",
    )


def test_iam_dataset_crops_xml_lines_and_generates_ground_truth(tmp_path: Path) -> None:
    _write_tiny_iam(tmp_path)
    records = build_iam_line_records(tmp_path)
    assert len(records) == 1
    assert records[0].writer_id == "007"
    assert [word.text for word in records[0].words] == ["unseen", "sentence"]
    dataset = IAMRegistrationDataset(
        records,
        image_size=(32, 64),
        training=False,
        max_translation=4.0,
        elastic_amplitude=2.0,
    )
    sample = dataset[0]
    assert sample["source"].shape == (1, 32, 64)
    assert sample["target"].shape == (1, 32, 64)
    assert sample["flow"].shape == (2, 32, 64)
    assert sample["valid_mask"].shape == (1, 32, 64)


def test_identity_pair_has_exact_zero_ground_truth_flow(tmp_path: Path) -> None:
    _write_tiny_iam(tmp_path)
    dataset = IAMRegistrationDataset(
        build_iam_line_records(tmp_path),
        image_size=(32, 64),
        training=False,
        identity_probability=1.0,
    )
    sample = dataset[0]
    assert sample["is_identity"].item() is True
    assert torch.count_nonzero(sample["flow"]).item() == 0


def test_real_pair_landmarks_use_same_normalized_canvas(tmp_path: Path) -> None:
    _write_tiny_iam(tmp_path)
    first = build_iam_line_records(tmp_path)[0]
    second = IAMLineRecord(
        line_id="second-line",
        writer_id="008",
        transcription=first.transcription,
        form_path=first.form_path,
        bbox=first.bbox,
        words=first.words,
    )
    pairs = find_cross_writer_line_pairs([first, second], minimum_words=2)
    assert pairs == [(first, second)]
    image, landmarks = load_normalized_iam_line(first, image_size=(32, 64))
    assert image.shape == (1, 32, 64)
    assert len(landmarks) == 2
    assert all(0 <= landmark.x < 64 and 0 <= landmark.y < 32 for landmark in landmarks)


def test_writer_splits_are_disjoint() -> None:
    records = [
        IAMLineRecord(
            line_id=f"line-{writer}",
            writer_id=str(writer),
            transcription=f"text {writer}",
            form_path=Path("unused.png"),
            bbox=(0, 0, 1, 1),
        )
        for writer in range(20)
    ]
    splits = split_records_by_writer(records, seed=3)
    writer_sets = [{record.writer_id for record in split} for split in splits.values()]
    assert writer_sets[0].isdisjoint(writer_sets[1])
    assert writer_sets[0].isdisjoint(writer_sets[2])
    assert writer_sets[1].isdisjoint(writer_sets[2])


def test_cross_font_dataset_uses_flow_supervision_without_pixel_loss() -> None:
    font_dir = Path("fonts")
    fonts = sorted(font_dir.glob("*.ttf"))[:2]
    dataset = SyntheticCrossFontRegistrationDataset(
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        fonts,
        length=1,
        image_size=(32, 64),
        training=False,
        min_words=3,
        max_words=3,
        max_translation=4.0,
        elastic_amplitude=2.0,
    )
    sample = dataset[0]
    assert sample["source"].shape == (1, 32, 64)
    assert sample["flow"].shape == (2, 32, 64)
    assert sample["photometric_mask"].sum().item() == 0
    assert sample["is_identity"].item() is False
