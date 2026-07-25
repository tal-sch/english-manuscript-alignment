from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from manuscript_registration.data import (
    IAMRegistrationDataset,
    build_iam_line_records,
    split_records_by_writer,
    split_statistics,
)
from manuscript_registration.metrics import identity_baseline_metrics, registration_metrics
from manuscript_registration.model import PatchCorrelationRegistration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate registration on writer-disjoint IAM test data")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--iam-root", type=Path, default=Path("IAM_Data"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PatchCorrelationRegistration(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_config = checkpoint["data_config"]
    records = build_iam_line_records(args.iam_root)
    splits = split_records_by_writer(records, seed=data_config["split_seed"])
    dataset = IAMRegistrationDataset(
        splits["test"],
        image_size=tuple(data_config["image_size"]),
        training=False,
        seed=data_config["split_seed"] + 20_000,
        max_translation=data_config["max_translation"],
        elastic_amplitude=data_config["elastic_amplitude"],
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
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
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
    results["evaluated_batches"] = batches
    results["split_statistics"] = split_statistics(splits)["test"]
    print(json.dumps(results, indent=2))
    output_path = args.output or args.checkpoint.with_name("test_metrics.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
