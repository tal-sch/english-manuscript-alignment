from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from manuscript_registration.data import (
    IAMRegistrationDataset,
    build_iam_line_records,
    split_records_by_writer,
)
from manuscript_registration.inference import (
    alignment_overlay,
    flow_visualization,
    load_registration_model,
    tensor_to_pil,
)
from manuscript_registration.metrics import identity_baseline_metrics, registration_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qualitative registration examples")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--iam-root", type=Path, default=Path("IAM_Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("alignment_output/examples"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def labeled_panel(images: list[tuple[str, Image.Image]]) -> Image.Image:
    label_height = 24
    widths = [image.width for _, image in images]
    height = max(image.height for _, image in images)
    panel = Image.new("RGB", (sum(widths), height + label_height), "white")
    draw = ImageDraw.Draw(panel)
    offset = 0
    for label, image in images:
        panel.paste(image.convert("RGB"), (offset, label_height))
        draw.text((offset + 6, 5), label, fill="black")
        offset += image.width
    return panel


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_registration_model(args.checkpoint, device=device)
    data_config = checkpoint["data_config"]
    splits = split_records_by_writer(
        build_iam_line_records(args.iam_root), seed=data_config["split_seed"]
    )
    dataset = IAMRegistrationDataset(
        splits["test"],
        image_size=tuple(data_config["image_size"]),
        training=False,
        seed=data_config["split_seed"] + 20_000,
        max_translation=data_config["max_translation"],
        elastic_amplitude=data_config["elastic_amplitude"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for index in range(args.start_index, min(args.start_index + args.count, len(dataset))):
        sample = dataset[index]
        source = sample["source"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        flow = sample["flow"].unsqueeze(0).to(device)
        valid = sample["valid_mask"].unsqueeze(0).to(device)
        output = model(source, target)
        metrics = registration_metrics(output, target, flow, valid)
        baseline = identity_baseline_metrics(source, target, flow, valid)
        panel = labeled_panel(
            [
                ("Source", tensor_to_pil(source)),
                ("Target", tensor_to_pil(target)),
                ("Aligned", tensor_to_pil(output.aligned)),
                ("Overlay", alignment_overlay(output.aligned, target)),
                ("Flow", flow_visualization(output.flow)),
            ]
        )
        filename = f"{index:04d}_{sample['line_id']}.png"
        panel.save(args.output_dir / filename)
        summaries.append(
            {
                "index": index,
                "line_id": sample["line_id"],
                "writer_id": sample["writer_id"],
                "transcription": sample["transcription"],
                **{name: float(value.cpu()) for name, value in metrics.items()},
                **{name: float(value.cpu()) for name, value in baseline.items()},
            }
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(summaries)} examples to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
