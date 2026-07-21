from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from manuscript_registration.data import build_iam_line_records, split_records_by_writer
from manuscript_registration.inference import (
    alignment_overlay,
    flow_visualization,
    load_registration_model,
    tensor_to_pil,
)
from manuscript_registration.real_pairs import (
    IAMWordLandmark,
    find_cross_writer_line_pairs,
    load_normalized_iam_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate word-landmark alignment on real IAM cross-writer line pairs"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--iam-root", type=Path, default=Path("IAM_Data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--minimum-words", type=int, default=3)
    return parser.parse_args()


def sample_flow(flow: torch.Tensor, landmarks: list[IAMWordLandmark]) -> torch.Tensor:
    height, width = flow.shape[-2:]
    coordinates = torch.tensor(
        [
            [
                2.0 * landmark.x / max(width - 1, 1) - 1.0,
                2.0 * landmark.y / max(height - 1, 1) - 1.0,
            ]
            for landmark in landmarks
        ],
        device=flow.device,
        dtype=flow.dtype,
    ).view(1, 1, -1, 2)
    return F.grid_sample(flow, coordinates, align_corners=True).view(2, -1).transpose(0, 1)


def mark_landmarks(image: Image.Image, landmarks: list[IAMWordLandmark]) -> Image.Image:
    marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    for landmark in landmarks:
        x, y = landmark.x, landmark.y
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(220, 30, 30), width=2)
    return marked


def labeled_panel(images: list[tuple[str, Image.Image]]) -> Image.Image:
    label_height = 24
    panel = Image.new(
        "RGB", (sum(image.width for _, image in images), max(i.height for _, i in images) + 24), "white"
    )
    draw = ImageDraw.Draw(panel)
    offset = 0
    for label, image in images:
        panel.paste(image.convert("RGB"), (offset, label_height))
        draw.text((offset + 5, 5), label, fill="black")
        offset += image.width
    return panel


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_registration_model(args.checkpoint, device=device)
    data_config = checkpoint["data_config"]
    image_size = tuple(data_config["image_size"])
    splits = split_records_by_writer(
        build_iam_line_records(args.iam_root), seed=int(data_config["split_seed"])
    )
    pairs = find_cross_writer_line_pairs(
        splits["test"], minimum_words=args.minimum_words, maximum_pairs=args.max_pairs
    )
    if not pairs:
        raise RuntimeError("No repeated test transcription was found across IAM writers")

    output_dir = args.output_dir or args.checkpoint.with_name("real_pair_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_results: list[dict[str, object]] = []
    all_baseline_errors: list[float] = []
    all_model_errors: list[float] = []

    for pair_index, (source_record, target_record) in enumerate(pairs):
        source, source_landmarks = load_normalized_iam_line(source_record, image_size=image_size)
        target, target_landmarks = load_normalized_iam_line(target_record, image_size=image_size)
        source_by_index = {landmark.index: landmark for landmark in source_landmarks}
        target_by_index = {landmark.index: landmark for landmark in target_landmarks}
        common_indices = sorted(source_by_index.keys() & target_by_index.keys())
        if not common_indices:
            continue
        source_points = torch.tensor(
            [[source_by_index[i].x, source_by_index[i].y] for i in common_indices],
            device=device,
        )
        selected_targets = [target_by_index[i] for i in common_indices]
        target_points = torch.tensor(
            [[landmark.x, landmark.y] for landmark in selected_targets], device=device
        )
        output = model(source.unsqueeze(0).to(device), target.unsqueeze(0).to(device))
        mapped_source_points = target_points + sample_flow(output.flow, selected_targets)
        baseline_errors = torch.linalg.vector_norm(target_points - source_points, dim=1)
        model_errors = torch.linalg.vector_norm(mapped_source_points - source_points, dim=1)
        baseline_values = baseline_errors.cpu().tolist()
        model_values = model_errors.cpu().tolist()
        all_baseline_errors.extend(baseline_values)
        all_model_errors.extend(model_values)

        panel = labeled_panel(
            [
                ("Source landmarks", mark_landmarks(tensor_to_pil(source), source_landmarks)),
                ("Target landmarks", mark_landmarks(tensor_to_pil(target), target_landmarks)),
                ("Aligned", tensor_to_pil(output.aligned)),
                ("Overlay", alignment_overlay(output.aligned, target.unsqueeze(0).to(device))),
                ("Predicted flow", flow_visualization(output.flow)),
            ]
        )
        filename = f"{pair_index:02d}_{source_record.line_id}_{target_record.line_id}.png"
        panel.save(output_dir / filename)
        pair_results.append(
            {
                "source_line_id": source_record.line_id,
                "target_line_id": target_record.line_id,
                "source_writer": source_record.writer_id,
                "target_writer": target_record.writer_id,
                "transcription": source_record.transcription,
                "landmarks": len(common_indices),
                "identity_landmark_error": sum(baseline_values) / len(baseline_values),
                "model_landmark_error": sum(model_values) / len(model_values),
                "image": filename,
            }
        )

    baseline_mean = sum(all_baseline_errors) / len(all_baseline_errors)
    model_mean = sum(all_model_errors) / len(all_model_errors)
    results = {
        "pairs": len(pair_results),
        "landmarks": len(all_model_errors),
        "identity_landmark_error": baseline_mean,
        "model_landmark_error": model_mean,
        "landmark_error_reduction_percent": 100.0 * (baseline_mean - model_mean) / baseline_mean,
        "model_landmarks_within_5px": sum(error <= 5 for error in all_model_errors)
        / len(all_model_errors),
        "model_landmarks_within_10px": sum(error <= 10 for error in all_model_errors)
        / len(all_model_errors),
        "metric_note": (
            "Predicted backward flow is sampled at each target word center and compared with "
            "the matching source word center. These are real IAM images from different writers."
        ),
        "pair_results": pair_results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "pair_results"}, indent=2))


if __name__ == "__main__":
    main()
