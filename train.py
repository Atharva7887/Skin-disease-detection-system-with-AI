"""Train a skin-lesion classifier on HAM10000.

Example:
    python train.py --data-dir D:/datasets/HAM10000 --model efficientnet_b0 --epochs 15
    python train.py --data-dir D:/datasets/HAM10000 --model timm/convnextv2_tiny.fcmae_ft_in22k_in1k
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from skinlesion.config import CLASS_CODES, IMG_SIZE
from skinlesion.data import (LesionDataset, build_transforms, class_weights,
                             load_metadata, split_by_lesion)
from skinlesion.evaluate import compute_metrics, predict_loader
from skinlesion.model import (build_model, save_checkpoint,
                              set_backbone_trainable)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Folder containing HAM10000_metadata.csv and the images")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=2, help="Epochs with the backbone frozen")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5, help="Early-stop patience on val macro-F1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    return ap.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
    return total / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_metadata(args.data_dir)
    splits = split_by_lesion(df, args.seed)
    for name, part in splits.items():
        print(f"{name:5s}: {len(part):5d} images, {part['lesion_id'].nunique():5d} lesions")
    assert not set(splits["train"]["lesion_id"]) & set(splits["test"]["lesion_id"]), "lesion leakage"

    loaders = {
        name: DataLoader(
            LesionDataset(part, build_transforms(name == "train", args.img_size)),
            batch_size=args.batch_size, shuffle=name == "train", num_workers=args.workers,
            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        )
        for name, part in splits.items()
    }

    model = build_model(args.model, pretrained=True).to(device)
    weights = class_weights(splits["train"]["label"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, stale, ckpt_path = -1.0, 0, out_dir / "best.pt"
    optimizer, scheduler = None, None

    for epoch in range(args.epochs):
        if epoch == 0 or epoch == args.warmup_epochs:
            frozen = epoch < args.warmup_epochs
            set_backbone_trainable(model, args.model, not frozen)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr if frozen else args.lr / 3, weight_decay=1e-2,
            )
            remaining = args.warmup_epochs if frozen else args.epochs - args.warmup_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(remaining, 1))

        t0 = time.time()
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        scheduler.step()
        probs, targets = predict_loader(model, loaders["val"], device)
        val = compute_metrics(probs, targets)
        print(f"epoch {epoch + 1:2d}/{args.epochs} loss {train_loss:.4f} "
              f"val acc {val['accuracy']:.3f} bal-acc {val['balanced_accuracy']:.3f} "
              f"macro-F1 {val['macro_f1']:.3f} mel-recall {val['melanoma_recall']:.3f} "
              f"({time.time() - t0:.0f}s)")

        if val["macro_f1"] > best_f1:
            best_f1, stale = val["macro_f1"], 0
            save_checkpoint(ckpt_path, model, args.model, args.img_size, {"val": val})
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    # Final, one-shot evaluation of the best checkpoint on the untouched test split.
    best = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    probs, targets = predict_loader(model, loaders["test"], device)
    test = compute_metrics(probs, targets)
    save_checkpoint(ckpt_path, model, args.model, args.img_size, {"val": best["metrics"]["val"], "test": test})
    (out_dir / "metrics.json").write_text(json.dumps({"classes": CLASS_CODES, "test": test}, indent=2))
    print(f"\nTEST  acc {test['accuracy']:.3f}  balanced-acc {test['balanced_accuracy']:.3f}  "
          f"macro-F1 {test['macro_f1']:.3f}  melanoma-recall {test['melanoma_recall']:.3f}")
    print(f"Saved {ckpt_path} and {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
