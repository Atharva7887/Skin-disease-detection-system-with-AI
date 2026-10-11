"""Does test-time image processing, TTA or calibration help?  (No retraining.)

Methods are compared on the HAM10000 *validation* split; the independent ISIC 2018 set is reported
alongside so selection is not made on it. All variants share one resize path
(resize 448 -> preprocess -> resize 224) so differences come from the processing only.

    python scripts/inference_study.py CKPT [CKPT ...]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

from skinlesion.config import CLASS_CODES
from skinlesion.data import load_metadata, split_by_lesion
from skinlesion.imageproc import PREPROCESSORS, WORK_SIZE
from skinlesion.model import load_checkpoint


def prep_one(args):
    path, name = args
    im = Image.open(path).convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR)
    fn = PREPROCESSORS[name]
    if fn is not None:
        im = fn(im)
    return np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.uint8)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-8, 1)).mean())


def temper(probs, T):
    p = np.power(np.clip(probs, 1e-8, 1), 1.0 / T)
    return p / p.sum(1, keepdims=True)


@torch.no_grad()
def model_probs(model, ck, arr, tta, dev, bs=100):
    mean = torch.tensor(ck.get("mean", (0.485, 0.456, 0.406)), device=dev).view(1, 3, 1, 1)
    std = torch.tensor(ck.get("std", (0.229, 0.224, 0.225)), device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        views = [x] if not tta else [torch.rot90(x, k, (2, 3)) for k in range(4)] + \
                                    [torch.rot90(x.flip(3), k, (2, 3)) for k in range(4)]
        out.append(torch.stack([torch.softmax(model(v).float(), 1) for v in views]).mean(0).cpu())
    return torch.cat(out).numpy()


def summarize(name, probs, y):
    pred, conf = probs.argmax(1), probs.max(1)
    ok = pred == y
    mel = y == CLASS_CODES.index("mel")
    from sklearn.metrics import f1_score
    return (f"{name:34s} acc {ok.mean():.3f} F1 {f1_score(y, pred, average='macro'):.3f} "
            f"mel-rec {(pred[mel] == CLASS_CODES.index('mel')).mean():.3f} | conf {conf.mean():.3f} "
            f"(right {conf[ok].mean():.3f} / wrong {conf[~ok].mean():.3f}) ECE {ece(probs, y):.3f} NLL {nll(probs, y):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="+")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham = load_metadata("data/HAM10000")
    val = split_by_lesion(ham)["val"]
    isic = load_metadata("data/ISIC2018", "ISIC2018_Task3_Test_GroundTruth.csv")
    sets = {"VAL (HAM10000 val split)": val, "TEST (ISIC 2018, independent)": isic}
    models = [load_checkpoint(p, dev) for p in a.checkpoint]

    results = {}
    with Pool(6) as pool:
        for sname, df in sets.items():
            y = df["label"].to_numpy()
            print(f"\n=== {sname}: {len(df)} images ===", flush=True)
            for name in PREPROCESSORS:
                arr = np.stack(pool.map(prep_one, [(p, name) for p in df["path"]], chunksize=16))
                for tta in (False, True) if name in ("none", "hair_removal") else (False,):
                    probs = np.mean([model_probs(m, ck, arr, tta, dev) for m, ck in models], axis=0)
                    label = f"{name}{' + TTA(8)' if tta else ''}"
                    results[(sname, label)] = (probs, y)
                    print(summarize(label, probs, y), flush=True)

    # Temperature scaling: fit on VAL (no ISIC 2018 labels used), apply to both.
    vp, vy = results[("VAL (HAM10000 val split)", "none")]
    Ts = np.arange(0.5, 3.01, 0.05)
    T = float(Ts[int(np.argmin([nll(temper(vp, t), vy) for t in Ts]))])
    print(f"\n=== Temperature scaling (fitted on VAL): T = {T:.2f}  (T<1 sharpens, T>1 softens) ===")
    for sname in sets:
        p, y = results[(sname, "none")]
        print(summarize(f"{sname.split(' ')[0]} raw", p, y))
        print(summarize(f"{sname.split(' ')[0]} calibrated (T={T:.2f})", temper(p, T), y))
