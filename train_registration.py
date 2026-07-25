from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from manuscript_registration.data import (
    IAMRegistrationDataset,
    SyntheticCrossFontRegistrationDataset,
    build_iam_line_records,
    records_to_jsonable,
    load_vocabulary_splits,
    split_records_by_writer,
    split_statistics,
)
from manuscript_registration.losses import RegistrationLoss
from manuscript_registration.metrics import identity_baseline_metrics, registration_metrics
from manuscript_registration.model import PatchCorrelationRegistration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manuscript line image registration")
    parser.add_argument("--iam-root", type=Path, default=Path("IAM_Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("registration_runs/default"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-residual-pixels", type=float, default=48.0)
    parser.add_argument("--max-translation", type=float, default=48.0)
    parser.add_argument("--elastic-amplitude", type=float, default=10.0)
    parser.add_argument("--samples-per-record", type=int, default=1)
    parser.add_argument(
        "--identity-probability",
        type=float,
        default=0.0,
        help="Fraction of IAM training pairs with exact zero ground-truth flow",
    )
    parser.add_argument("--synthetic-samples", type=int, default=4000)
    parser.add_argument("--words-file", type=Path, default=Path("common_words.txt"))
    parser.add_argument("--fonts-dir", type=Path, default=Path("fonts"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights only and start a fresh fine-tuning schedule",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def average_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_epoch(
    model: PatchCorrelationRegistration,
    loader: DataLoader,
    criterion: RegistrationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(batch["source"], batch["target"])
            losses = criterion(
                output,
                batch["target"],
                batch["flow"],
                batch["valid_mask"],
                batch.get("photometric_mask"),
            )
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for name, value in losses.detached().items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    return average_totals(totals, count)


@torch.no_grad()
def validate_epoch(
    model: PatchCorrelationRegistration,
    loader: DataLoader,
    criterion: RegistrationLoss,
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(batch["source"], batch["target"])
            losses = criterion(
                output,
                batch["target"],
                batch["flow"],
                batch["valid_mask"],
                batch.get("photometric_mask"),
            )
        metrics = registration_metrics(output, batch["target"], batch["flow"], batch["valid_mask"])
        baseline = identity_baseline_metrics(
            batch["source"], batch["target"], batch["flow"], batch["valid_mask"]
        )
        values = {
            **{f"loss_{name}": value for name, value in losses.detached().items()},
            **{name: float(value.detach().cpu()) for name, value in metrics.items()},
            **{name: float(value.detach().cpu()) for name, value in baseline.items()},
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    return average_totals(totals, count)


def save_checkpoint(
    path: Path,
    model: PatchCorrelationRegistration,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "model_config": {
            "base_channels": args.base_channels,
            "max_residual_pixels": args.max_residual_pixels,
        },
        "data_config": {
            "image_size": [args.height, args.width],
            "split_seed": args.seed,
            "max_translation": args.max_translation,
            "elastic_amplitude": args.elastic_amplitude,
            "identity_probability": args.identity_probability,
            "synthetic_samples": args.synthetic_samples,
            "words_file": str(args.words_file),
            "fonts_dir": str(args.fonts_dir),
        },
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    if args.height % 8 or args.width % 8:
        raise ValueError("Height and width must be divisible by 8")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device} | AMP: {use_amp} ({amp_dtype})")

    records = build_iam_line_records(args.iam_root)
    splits = split_records_by_writer(records, seed=args.seed)
    statistics = split_statistics(splits)
    print(json.dumps(statistics, indent=2))
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as manifest:
        json.dump(
            {
                "seed": args.seed,
                "statistics": statistics,
                "splits": {name: records_to_jsonable(values) for name, values in splits.items()},
            },
            manifest,
            indent=2,
        )

    image_size = (args.height, args.width)
    iam_train_dataset = IAMRegistrationDataset(
        splits["train"],
        image_size=image_size,
        training=True,
        seed=args.seed,
        max_translation=args.max_translation,
        elastic_amplitude=args.elastic_amplitude,
        samples_per_record=args.samples_per_record,
        identity_probability=args.identity_probability,
    )
    train_parts: list[Dataset] = [iam_train_dataset]
    if args.synthetic_samples > 0:
        vocabulary = load_vocabulary_splits(args.words_file, seed=args.seed)
        font_paths = sorted(args.fonts_dir.glob("*.ttf"))
        training_fonts = font_paths[:-1] if len(font_paths) > 2 else font_paths
        synthetic_dataset = SyntheticCrossFontRegistrationDataset(
            vocabulary["train"],
            training_fonts,
            length=args.synthetic_samples,
            image_size=image_size,
            training=True,
            seed=args.seed,
            max_translation=args.max_translation,
            elastic_amplitude=args.elastic_amplitude,
        )
        train_parts.append(synthetic_dataset)
        print(
            f"Added {len(synthetic_dataset)} cross-font pairs using {len(training_fonts)} fonts; "
            f"held out font: {font_paths[-1].name if len(font_paths) > 2 else 'none'}"
        )
    train_dataset: Dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_dataset = IAMRegistrationDataset(
        splits["val"],
        image_size=image_size,
        training=False,
        seed=args.seed + 10_000,
        max_translation=args.max_translation,
        elastic_amplitude=args.elastic_amplitude,
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = PatchCorrelationRegistration(
        base_channels=args.base_channels,
        max_residual_pixels=args.max_residual_pixels,
    ).to(device)
    criterion = RegistrationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_epoch = 1
    best_epe = float("inf")
    if args.init_checkpoint is not None:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(initial_checkpoint["model_state"])
        print(f"Initialized model weights from {args.init_checkpoint}")
    elif args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_epe = float(resume_checkpoint.get("metrics", {}).get("epe", float("inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")
    stale_epochs = 0
    history_path = args.output_dir / "history.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_values = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )
        val_values = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_values.items()},
            **{f"val_{key}": value for key, value in val_values.items()},
        }
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_values['total']:.4f} | val_epe={val_values['epe']:.3f}px | "
            f"val_ssim={val_values['ssim']:.4f} | {row['seconds']:.1f}s"
        )
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, scheduler, epoch, val_values, args
        )
        if val_values["epe"] < best_epe:
            best_epe = val_values["epe"]
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, scheduler, epoch, val_values, args
            )
            print(f"Saved new best checkpoint (EPE={best_epe:.3f}px)")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epochs without EPE improvement")
                break


if __name__ == "__main__":
    main()
