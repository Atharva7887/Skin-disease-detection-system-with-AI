"""Wait for the ISIC 2019 download, extract it, then train on HAM10000 + ISIC 2019.

Safe to re-run: extraction is skipped when images exist and training uses --resume.
    python scripts/run_combined.py
"""
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/ISIC2019/ISIC_2019_Training_Input.zip"
IMAGES = ROOT / "data/ISIC2019/images"
FETCH_LOG = ROOT / "logs/fetch_isic2019.log"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def download_finished():
    return ZIP.exists() and FETCH_LOG.exists() and FETCH_LOG.read_text().strip().endswith("done")


def main():
    while not download_finished():
        time.sleep(30)
    log("download complete")

    if not any(IMAGES.rglob("*.jpg")):
        with zipfile.ZipFile(ZIP) as z:
            bad = z.testzip()
            if bad:
                sys.exit(f"corrupt member in zip: {bad}")
            log(f"extracting {len(z.namelist())} files")
            z.extractall(IMAGES)
    log(f"{sum(1 for _ in IMAGES.rglob('*.jpg'))} ISIC 2019 images on disk")

    cmd = [
        str(ROOT / ".venv/Scripts/python.exe"), "-u", "train.py",
        "--data-dir", "data/HAM10000",
        "--extra-isic2019", "data/ISIC2019",
        "--exclude-csv", "data/ISIC2018/ISIC2018_Task3_Test_GroundTruth.csv",
        "--model", "timm/convnextv2_tiny.fcmae_ft_in22k_in1k",
        "--epochs", "15", "--patience", "5", "--lr", "2e-4", "--weight-power", "0.5",
        "--workers", "6", "--out-dir", "models/convnextv2_combined", "--resume",
    ]
    log("training: " + " ".join(cmd[2:]))
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
