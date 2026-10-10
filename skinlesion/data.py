"""HAM10000 loading, leakage-free splitting and PyTorch datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import CLASS_CODES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and attach an absolute ``path`` per image.

    Images may sit directly in ``data_dir`` or in sub-folders such as
    ``HAM10000_images_part_1`` / ``part_2`` (the Kaggle / Dataverse layout).
    """
    data_dir = Path(data_dir)
    csv_path = next(data_dir.rglob("HAM10000_metadata.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {data_dir}")

    index = {p.stem: p for p in data_dir.rglob("*.jpg")}
    df = pd.read_csv(csv_path)
    df["path"] = df["image_id"].map(index)

    missing = int(df["path"].isna().sum())
    if missing:
        print(f"Warning: dropping {missing} metadata rows with no image file.")
        df = df.dropna(subset=["path"])
    if df.empty:
        raise FileNotFoundError(f"No HAM10000 images found under {data_dir}")

    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASS_CODES)})
    if df["label"].isna().any():
        raise ValueError(f"Unexpected diagnosis codes: {sorted(set(df['dx']) - set(CLASS_CODES))}")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def split_by_lesion(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """80/10/10 train/val/test split that keeps every lesion in one split.

    HAM10000 holds several photos of the same lesion (``lesion_id``). A plain
    random split puts near-duplicates in train and test and inflates accuracy,
    so we group by lesion and stratify by diagnosis.
    """
    folds = list(
        StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed).split(
            df, df["label"], groups=df["lesion_id"]
        )
    )
    test_idx = folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.concatenate([f[1] for f in folds[2:]])
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def class_weights(labels: np.ndarray, num_classes: int = len(CLASS_CODES)) -> torch.Tensor:
    """Inverse-frequency weights (mean-normalised) to counter class imbalance."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def build_transforms(train: bool, img_size: int = IMG_SIZE) -> v2.Compose:
    normalize = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return v2.Compose(
            [v2.Resize((img_size, img_size)), v2.ToImage(), *normalize]
        )
    return v2.Compose(
        [
            v2.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            v2.ToImage(),
            *normalize,
        ]
    )


class LesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: v2.Compose):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), int(self.labels[i])
