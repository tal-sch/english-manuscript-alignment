from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from manuscript_registration.data import build_iam_line_records, split_records_by_writer
from manuscript_registration.inference import load_registration_model
from manuscript_registration.losses import masked_mean, ssim_map
from manuscript_registration.real_pairs import load_normalized_iam_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the no-warp identity case")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--iam-root", type=Path, default=Path("IAM_Data"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


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
    records = splits["test"]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    weighted_totals = {
        "mean_flow_magnitude": 0.0,
        "mean_absolute_horizontal_flow": 0.0,
        "mean_absolute_vertical_flow": 0.0,
        "ink_flow_magnitude": 0.0,
        "aligned_mae": 0.0,
        "aligned_ssim": 0.0,
    }
    processed = 0
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        images = torch.stack(
            [load_normalized_iam_line(record, image_size=image_size)[0] for record in batch_records]
        ).to(device)
        output = model(images, images)
        magnitude = torch.linalg.vector_norm(output.flow, dim=1, keepdim=True)
        ink_mask = (1.0 - images).clamp(0.0, 1.0)
        batch_values = {
            "mean_flow_magnitude": magnitude.mean(),
            "mean_absolute_horizontal_flow": output.flow[:, :1].abs().mean(),
            "mean_absolute_vertical_flow": output.flow[:, 1:].abs().mean(),
            "ink_flow_magnitude": masked_mean(magnitude, ink_mask),
            "aligned_mae": (output.aligned - images).abs().mean(),
            "aligned_ssim": ssim_map(output.aligned, images).mean(),
        }
        batch_count = len(batch_records)
        for name, value in batch_values.items():
            weighted_totals[name] += float(value.cpu()) * batch_count
        processed += batch_count

    results = {name: total / processed for name, total in weighted_totals.items()}
    results.update(
        {
            "samples": processed,
            "identity_ground_truth_flow": 0.0,
            "note": "Source and target are the exact same normalized writer-disjoint IAM image.",
        }
    )
    print(json.dumps(results, indent=2))
    output_path = args.output or args.checkpoint.with_name("identity_metrics.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
