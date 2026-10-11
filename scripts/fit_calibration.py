"""Fit the ensemble's temperature on the HAM10000 VALIDATION split only and write
webapp/calibration.json. No ISIC 2018 label is used.

    python scripts/fit_calibration.py CKPT [CKPT ...] [--no-tta]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinlesion.data import LesionDataset, eval_transform_for, load_metadata, split_by_lesion
from skinlesion.model import load_checkpoint
from skinlesion.predict import model_probs


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > a) & (conf <= b)).mean() * abs((pred[(conf > a) & (conf <= b)] == y[(conf > a) & (conf <= b)]).mean() - conf[(conf > a) & (conf <= b)].mean())
                     for a, b in zip(e[:-1], e[1:]) if ((conf > a) & (conf <= b)).any()))


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean())


def temper(p, t):
    q = np.power(np.clip(p, 1e-8, 1), 1.0 / t)
    return q / q.sum(1, keepdims=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="+")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "webapp/calibration.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = split_by_lesion(load_metadata(ROOT / "data/HAM10000"))["val"]
    y = val["label"].to_numpy()
    per_model = []
    for path in a.checkpoint:
        m, ck = load_checkpoint(ROOT / path, dev)
        dl = DataLoader(LesionDataset(val, eval_transform_for(ck)), batch_size=50, num_workers=2)
        per_model.append(torch.cat([model_probs(m, x.to(dev), not a.no_tta).cpu() for x, _ in dl]).numpy())
    p = np.mean(per_model, axis=0)
    grid = np.round(np.arange(0.15, 1.51, 0.01), 2)
    scores = [nll(temper(p, t), y) for t in grid]
    T = float(grid[int(np.argmin(scores))])
    assert grid[0] < T < grid[-1], f"optimum T={T} is at the edge of the search grid; widen it"
    out = {
        "temperature": T,
        "members": a.checkpoint,
        "tta": not a.no_tta,
        "fitted_on": f"HAM10000 validation split ({len(y)} images)",
        "fitted": datetime.date.today().isoformat(),
        "val_before": {"ece": round(ece(p, y), 4), "nll": round(nll(p, y), 4), "mean_confidence": round(float(p.max(1).mean()), 4)},
        "val_after": {"ece": round(ece(temper(p, T), y), 4), "nll": round(nll(temper(p, T), y), 4), "mean_confidence": round(float(temper(p, T).max(1).mean()), 4)},
        "val_accuracy": round(float((p.argmax(1) == y).mean()), 4),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
