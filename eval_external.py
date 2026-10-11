"""Evaluate a trained checkpoint on an independent labelled set in HAM10000 format.

Example (ISIC 2018 Task 3 test set, never seen in training):
    python eval_external.py --checkpoint models/effb0/best.pt --data-dir data/ISIC2018 \
        --csv ISIC2018_Task3_Test_GroundTruth.csv
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from skinlesion.config import CLASS_CODES
from skinlesion.data import LesionDataset, eval_transform_for, load_metadata, split_by_lesion
from skinlesion.evaluate import compute_metrics, predict_loader
from skinlesion.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, nargs="+",
                    help="One or more checkpoints; several are ensembled by averaging probabilities")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="HAM10000_metadata.csv")
    ap.add_argument("--holdout", action="store_true",
                    help="Evaluate on the lesion-level test split of --data-dir (seed 42, as in train.py)")
    ap.add_argument("--out", help="Optional path to write metrics JSON")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(args.data_dir, args.csv)
    if args.holdout:
        df = split_by_lesion(df, seed=42)["test"]
    all_probs = []
    for path in args.checkpoint:
        model, ckpt = load_checkpoint(path, device)
        loader = DataLoader(LesionDataset(df, eval_transform_for(ckpt)),
                            batch_size=args.batch_size, num_workers=args.workers)
        probs, targets = predict_loader(model, loader, device)
        all_probs.append(probs)
        if len(args.checkpoint) > 1:
            single = compute_metrics(probs, targets)
            print(f"  {path}: acc {single['accuracy']:.3f} bal-acc {single['balanced_accuracy']:.3f} "
                  f"macro-F1 {single['macro_f1']:.3f} mel-recall {single['melanoma_recall']:.3f}")
    probs = np.mean(all_probs, axis=0)
    m = compute_metrics(probs, targets)
    print(f"{len(df)} images | acc {m['accuracy']:.3f} | balanced-acc {m['balanced_accuracy']:.3f} | "
          f"macro-F1 {m['macro_f1']:.3f} | melanoma-recall {m['melanoma_recall']:.3f}")
    for c in CLASS_CODES:
        r = m["report"][c]
        print(f"  {c:6s} precision {r['precision']:.2f} recall {r['recall']:.2f} n={int(r['support'])}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
