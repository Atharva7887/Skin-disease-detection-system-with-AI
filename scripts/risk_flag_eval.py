"""How the app's risk flag behaves on a labelled set: sensitivity for melanoma / any malignant
class vs. false-alarm rate on benign nevi, for one or more (ensembled) checkpoints.

    python scripts/risk_flag_eval.py CKPT [CKPT ...] --data-dir data/ISIC2018 --csv ISIC2018_Task3_Test_GroundTruth.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skinlesion.config import CLASS_CODES
from skinlesion.data import LesionDataset, eval_transform_for, load_metadata
from skinlesion.evaluate import predict_loader
from skinlesion.model import load_checkpoint
from skinlesion.triage import HIGH_RISK_MASS, MALIGNANT_OR_PRECANCEROUS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="+")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", default="HAM10000_metadata.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_metadata(a.data_dir, a.csv)
    P = []
    for path in a.checkpoint:
        m, ck = load_checkpoint(path, dev)
        dl = DataLoader(LesionDataset(df, eval_transform_for(ck)), batch_size=64, num_workers=2)
        probs, y = predict_loader(m, dl, dev); P.append(probs)
    probs = np.mean(P, axis=0)
    idx = {c: i for i, c in enumerate(CLASS_CODES)}
    conc = probs[:, [idx[c] for c in MALIGNANT_OR_PRECANCEROUS]].sum(1)
    top_bad = np.isin(probs.argmax(1), [idx[c] for c in MALIGNANT_OR_PRECANCEROUS])
    mel, nv = y == idx["mel"], y == idx["nv"]
    bad = np.isin(y, [idx[c] for c in MALIGNANT_OR_PRECANCEROUS])
    print(f"{len(y)} images | melanoma n={mel.sum()}, nevus n={nv.sum()}, mel/bcc/akiec n={bad.sum()}")
    print(f"{'rule (flag HIGH when...)':46s} {'mel sens':>9s} {'mel/bcc/akiec sens':>19s} {'nevus false alarms':>19s}")
    for name, flag in [("app rule: top class malignant OR mass>=%.2f" % HIGH_RISK_MASS, top_bad | (conc >= HIGH_RISK_MASS))] + \
                      [(f"malignant mass >= {t:.2f}", conc >= t) for t in (0.10, 0.15, 0.30, 0.50)]:
        print(f"{name:46s} {flag[mel].mean():9.1%} {flag[bad].mean():19.1%} {flag[nv].mean():19.1%}")
