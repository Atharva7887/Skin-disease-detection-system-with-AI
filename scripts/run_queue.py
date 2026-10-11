"""Train additional architectures on the combined data, one after another, once the first
combined run (ConvNeXt-V2) has finished. Re-runnable: finished models are skipped, interrupted
ones resume from last.pt.

    python scripts/run_queue.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
COMMON = ["--data-dir", "data/HAM10000", "--extra-isic2019", "data/ISIC2019",
          "--exclude-csv", "data/ISIC2018/ISIC2018_Task3_Test_GroundTruth.csv",
          "--epochs", "15", "--patience", "5", "--weight-power", "0.5", "--workers", "6", "--resume"]
QUEUE = [  # (model, out dir, lr)
    ("timm/tf_efficientnetv2_s.in21k_ft_in1k", "models/effnetv2s_combined", "3e-4"),
    ("timm/maxvit_tiny_tf_224.in1k", "models/maxvit_combined", "2e-4"),
]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    first = ROOT / "models/convnextv2_combined/metrics.json"
    while not first.exists():
        time.sleep(60)
    log("first combined run finished; starting queue")
    for model, out, lr in QUEUE:
        if (ROOT / out / "metrics.json").exists():
            log(f"skip {model} (done)")
            continue
        log(f"training {model}")
        rc = subprocess.call([PY, "-u", "train.py", "--model", model, "--lr", lr, "--out-dir", out, *COMMON], cwd=ROOT)
        log(f"{model} exited with {rc}")
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
