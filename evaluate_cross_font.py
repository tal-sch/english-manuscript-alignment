from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from manuscript_registration.data import (
    SyntheticCrossFontRegistrationDataset,
    load_vocabulary_splits,
)
from manuscript_registration.inference import load_registration_model
from manuscript_registration.metrics import identity_baseline_metrics, registration_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate registration on held-out words and a held-out font"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--words-file", type=Path, default=Path("common_words.txt"))
    parser.add_argument("--fonts-dir", type=Path, default=Path("fonts"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_registration_model(args.checkpoint, device=device)
    data_config = checkpoint["data_config"]
    seed = int(data_config["split_seed"])
    vocabulary = load_vocabulary_splits(args.words_file, seed=seed)
    font_paths = sorted(args.fonts_dir.glob("*.ttf"))
    if len(font_paths) < 3:
        raise RuntimeError("Cross-font held-out evaluation requires at least three fonts")
    # The final font is excluded from training_registration.py; pair it with a seen font.
    evaluation_fonts = [font_paths[0], font_paths[-1]]
    dataset = SyntheticCrossFontRegistrationDataset(
        vocabulary["test"],
        evaluation_fonts,
        length=args.samples,
        image_size=tuple(data_config["image_size"]),
        training=False,
        seed=seed + 30_000,
        max_translation=float(data_config["max_translation"]),
        elastic_amplitude=float(data_config["elastic_amplitude"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        flow = batch["flow"].to(device)
        valid = batch["valid_mask"].to(device)
        output = model(source, target)
        values = {
            **registration_metrics(output, target, flow, valid),
            **identity_baseline_metrics(source, target, flow, valid),
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.cpu())
        batches += 1
    results = {name: value / max(batches, 1) for name, value in totals.items()}
    results.update(
        {
            "samples": len(dataset),
            "unseen_vocabulary_size": len(vocabulary["test"]),
            "fonts": [path.name for path in evaluation_fonts],
            "note": "Pixel similarity is not a semantic metric here because glyph styles differ; EPE is primary.",
        }
    )
    print(json.dumps(results, indent=2))
    output_path = args.output or args.checkpoint.with_name("cross_font_metrics.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
