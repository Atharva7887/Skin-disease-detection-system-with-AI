"""Generate webapp/model_card.json: every number on the "Technical details" page.

Re-evaluates the shipped checkpoints (nothing is typed in by hand) and records hardware/software
and training-time facts. Re-run after retraining or changing the ensemble.

    python scripts/make_model_card.py CKPT [CKPT ...]      # same order as DEFAULT_MODEL_PATH
"""
import argparse
import datetime
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinlesion.config import CLASS_CODES, CLASS_NAMES
from skinlesion.data import LesionDataset, eval_transform_for, load_metadata, split_by_lesion
from skinlesion.evaluate import compute_metrics, predict_loader
from skinlesion.model import load_checkpoint
from skinlesion.predict import apply_temperature, model_probs
from skinlesion.triage import HIGH_RISK_MASS, MALIGNANT_OR_PRECANCEROUS

KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, shell=isinstance(cmd, str)).stdout.strip()
    except Exception:
        return ""


def hardware():
    gpu = run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader").split("\n")[0].split(", ")
    cpu = run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Select -First 1).Name"') or platform.processor()
    ram = run('powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"')
    return {
        "gpu": gpu[0] if gpu and gpu[0] else "none (CPU)",
        "gpu_memory": gpu[1] if len(gpu) > 1 else "",
        "driver": gpu[2] if len(gpu) > 2 else "",
        "cpu": cpu.strip(),
        "logical_cpus": __import__("os").cpu_count(),
        "ram_gb": ram,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    }


def software():
    import timm
    import torchvision
    return {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "torchvision": torchvision.__version__, "timm": timm.__version__}


def epoch_minutes(log: Path, marker: str | None = None):
    """(epochs, minutes) summed from 'epoch ... (NNs)' lines, optionally only after a marker line."""
    if not log.exists():
        return None
    active, n, secs = marker is None, 0, 0
    for line in log.read_text(errors="ignore").splitlines():
        if marker and marker in line:
            active, n, secs = True, 0, 0
        if active:
            m = re.match(r"epoch\s+\d+/\d+ .*\((\d+)s\)", line)
            if m:
                n, secs = n + 1, secs + int(m.group(1))
    return {"epochs_run": n, "minutes": round(secs / 60)} if n else None


def probs_on(model, ck, df, dev, tta=True):
    """Probabilities with the shipped inference mode (8-view TTA); returns (probs, labels)."""
    dl = DataLoader(LesionDataset(df, eval_transform_for(ck)), batch_size=50, num_workers=2)
    probs, ys = [], []
    for x, y in dl:
        probs.append(model_probs(model, x.to(dev), tta).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def calib_stats(probs, y, T, bins=15):
    def ece(p):
        conf, pred = p.max(1), p.argmax(1)
        e, tot = np.linspace(0, 1, bins + 1), 0.0
        for lo, hi in zip(e[:-1], e[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
        return float(tot)
    cal = apply_temperature(torch.tensor(probs), T).numpy()
    acc = float((probs.argmax(1) == y).mean())
    nll = lambda p: float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean())  # noqa: E731
    return {"accuracy": round(acc, 4),
            "raw": {"mean_confidence": round(float(probs.max(1).mean()), 4), "ece": round(ece(probs), 4), "nll": round(nll(probs), 4)},
            "calibrated": {"mean_confidence": round(float(cal.max(1).mean()), 4), "ece": round(ece(cal), 4), "nll": round(nll(cal), 4)}}


def pick(m):
    return {k: round(m[k], 4) for k in KEYS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "webapp/model_card.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ham = load_metadata(ROOT / "data/HAM10000")
    sets = {
        "isic2018": load_metadata(ROOT / "data/ISIC2018", "ISIC2018_Task3_Test_GroundTruth.csv"),
        "holdout": split_by_lesion(ham)["test"],
    }
    models = []
    for p in a.checkpoint:
        m, ck = load_checkpoint(ROOT / p, dev)
        models.append((p, m, ck, sum(x.numel() for x in m.parameters()) / 1e6))

    cal_path = ROOT / "webapp/calibration.json"
    cal_cfg = json.loads(cal_path.read_text()) if cal_path.exists() else None
    if cal_cfg and set(cal_cfg["members"]) != set(a.checkpoint):
        sys.exit("calibration.json was fitted for a different set of checkpoints: re-run scripts/fit_calibration.py")
    probs = {s: [] for s in sets}
    out_models = []
    for path, m, ck, params in models:
        row = {"arch": ck["arch"].replace("timm/", ""), "params_millions": round(params, 1), "checkpoint": path}
        for s, df in sets.items():
            pr, y = probs_on(m, ck, df, dev)
            probs[s].append(pr)
            row[s] = pick(compute_metrics(pr, y))
        out_models.append(row)

    ens = {}
    for s, df in sets.items():
        y = df["label"].to_numpy()
        pr = np.mean(probs[s], axis=0)
        full = compute_metrics(pr, y)
        if cal_cfg:
            ens.setdefault("calibration", {})[s] = calib_stats(pr, y, cal_cfg["temperature"])
        ens[s] = {**pick(full), "n_images": int(len(y)),
                  "per_class": {c: {"name": CLASS_NAMES[c], **{k: round(full["report"][c][k], 3) for k in ("precision", "recall", "f1-score")},
                                    "support": int(full["report"][c]["support"])} for c in CLASS_CODES},
                  "confusion_matrix": full["confusion_matrix"]}
        if s == "isic2018" and cal_cfg:
            shown = apply_temperature(torch.tensor(pr), cal_cfg["temperature"]).numpy()
            conf, pred = shown.max(1), shown.argmax(1)
            edges = [0.0, 0.5, 0.7, 0.9, 0.97, 1.0001]
            rel = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf >= lo) & (conf < hi)
                if m.any():
                    rel.append({"low": lo, "high": min(hi, 1.0), "n": int(m.sum()), "accuracy": round(float((pred[m] == y[m]).mean()), 3)})
            ens["reliability"] = rel
            nv_i, mel_i = CLASS_CODES.index("nv"), CLASS_CODES.index("mel")
            m = (pred == nv_i) & (conf >= 0.9)
            ens["confident_nevus"] = {"n": int(m.sum()), "n_melanoma": int((y[m] == mel_i).sum()),
                                      "share_melanoma": round(float((y[m] == mel_i).mean()), 3),
                                      "n_melanoma_total": int((y == mel_i).sum()),
                                      "melanomas_in_group": round(float(((y == mel_i) & m).sum() / max((y == mel_i).sum(), 1)), 3)}
        if s == "isic2018":
            idx = {c: i for i, c in enumerate(CLASS_CODES)}
            bad = [idx[c] for c in MALIGNANT_OR_PRECANCEROUS]
            flag = np.isin(pr.argmax(1), bad) | (pr[:, bad].sum(1) >= HIGH_RISK_MASS)
            ens["risk_flag"] = {
                "rule": f"HIGH if the top class is melanoma / basal cell carcinoma / actinic keratosis, or their combined probability is at least {HIGH_RISK_MASS:.0%}",
                "melanoma_flagged": round(float(flag[y == idx["mel"]].mean()), 3),
                "any_malignant_flagged": round(float(flag[np.isin(y, bad)].mean()), 3),
                "nevus_flagged": round(float(flag[y == idx["nv"]].mean()), 3),
                "n_melanoma": int((y == idx["mel"]).sum()), "n_nevus": int((y == idx["nv"]).sum()),
            }

    logs = ROOT / "logs"
    times = [epoch_minutes(logs / "combined.log"), epoch_minutes(logs / "combined2.log")]
    conv = {"epochs_run": sum(t["epochs_run"] for t in times if t), "minutes": sum(t["minutes"] for t in times if t)} if any(times) else None
    q = logs / "queue3.log"
    effv2 = epoch_minutes(q, "training timm/tf_efficientnetv2") if q.exists() else None
    if effv2 and q.exists():  # the effnet section runs until the maxvit marker
        txt = q.read_text(errors="ignore").split("training timm/maxvit_tiny")[0]
        tmp = ROOT / "logs/_tmp_eff.log"; tmp.write_text(txt)
        effv2 = epoch_minutes(tmp, "training timm/tf_efficientnetv2"); tmp.unlink()
    maxv = epoch_minutes(q, "training timm/maxvit_tiny") if q.exists() else None
    for row, t in zip(out_models, (conv, effv2, maxv)):
        row["training"] = t

    card = {
        "generated": datetime.date.today().isoformat(),
        "eval_sets": {
            "isic2018": "ISIC 2018 Task 3 test set: 1,511 independent dermoscopy images, never used for training or model selection.",
            "holdout": "HAM10000 test split: 1,002 images from lesions that are not in the training data.",
        },
        "inference": {
            "tta": "average over 8 flips/rotations of the image",
            "ensemble": "mean of the three models' probabilities",
            "calibration": (f"temperature {cal_cfg['temperature']} fitted on the {cal_cfg['fitted_on']}; changes the displayed percentage only, "
                            "never the predicted class, and never the risk flag (defined on raw probabilities)") if cal_cfg else "none",
        },
        "models": out_models,
        "ensemble": ens,
        "hardware": hardware(),
        "software": software(),
        "training": {
            "data": "HAM10000 (8,011 train images after a lesion-level split) + 14,885 ISIC 2019 images (train only); all HAM10000 and ISIC 2018 test images excluded from the extra data",
            "input_size": "224 x 224",
            "optimiser": "AdamW, weight decay 0.01, batch 32, mixed precision",
            "schedule": "2 epochs head-only warm-up, then full fine-tuning with cosine decay; learning rate 2e-4 to 3e-4 (head), one third of that for the backbone",
            "loss": "cross-entropy, label smoothing 0.1, square-root inverse-frequency class weights",
            "augmentation": "random resized crop, horizontal and vertical flips, rotation up to 30 degrees, colour jitter",
            "model_selection": "best validation macro-F1, early stopping (patience 5 to 6)",
            "pretraining": "ImageNet-pretrained backbones (ConvNeXt-V2 and EfficientNetV2-S: ImageNet-22k then 1k / 21k then 1k; MaxViT-tiny: ImageNet-1k)",
        },
        "classes": {c: CLASS_NAMES[c] for c in CLASS_CODES},
    }
    Path(a.out).write_text(json.dumps(card, indent=2))
    print("wrote", a.out)
