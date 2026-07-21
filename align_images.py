from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from manuscript_registration.inference import (
    align_pil_images,
    alignment_overlay,
    flow_visualization,
    load_registration_model,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a source manuscript line to a target line")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("alignment_output"))
    parser.add_argument("--maximum-width", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_registration_model(args.checkpoint, device=device)
    with Image.open(args.source) as source, Image.open(args.target) as target:
        output, source_tensor, target_tensor = align_pil_images(
            model,
            source,
            target,
            training_image_size=tuple(checkpoint["data_config"]["image_size"]),
            maximum_width=args.maximum_width,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(source_tensor).save(args.output_dir / "source_normalized.png")
    tensor_to_pil(target_tensor).save(args.output_dir / "target_normalized.png")
    tensor_to_pil(output.aligned).save(args.output_dir / "aligned.png")
    alignment_overlay(output.aligned, target_tensor).save(args.output_dir / "overlay.png")
    flow_visualization(output.flow).save(args.output_dir / "flow.png")
    mean_displacement = torch.linalg.vector_norm(output.flow, dim=1).mean().item()
    print(f"Saved alignment to {args.output_dir.resolve()}")
    print(f"Mean predicted displacement: {mean_displacement:.2f}px")


if __name__ == "__main__":
    main()
