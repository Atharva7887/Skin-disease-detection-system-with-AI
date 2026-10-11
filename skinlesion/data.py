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


def load_metadata(data_dir: str | Path, csv_name: str = "HAM10000_metadata.csv") -> pd.DataFrame:
    """Read the metadata CSV (HAM10000 format) and attach an absolute ``path`` per image.

    Images may sit directly in ``data_dir`` or in sub-folders such as
    ``HAM10000_images_part_1`` / ``part_2`` (the Kaggle / Dataverse layout).
    """
    data_dir = Path(data_dir)
    csv_path = next(data_dir.rglob(csv_name), None)
    if csv_path is None:
        raise FileNotFoundError(f"{csv_name} not found under {data_dir}")

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


# ISIC 2019 ground-truth columns -> our class codes. SCC (no safe mapping onto the HAM10000
# classes) and UNK are dropped.
ISIC2019_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}


def load_isic2019(data_dir: str | Path, exclude_ids: set[str]) -> pd.DataFrame:
    """Extra labelled training images from ISIC 2019, in the same format as ``load_metadata``.

    ISIC 2019 training = HAM10000 + BCN20000 + MSK, so every image id in ``exclude_ids``
    (all of HAM10000 and any evaluation set) is removed to avoid duplicates and leakage.
    Images without a lesion_id are treated as their own lesion.
    """
    data_dir = Path(data_dir)
    gt = pd.read_csv(next(data_dir.rglob("ISIC_2019_Training_GroundTruth.csv")))
    meta = pd.read_csv(next(data_dir.rglob("ISIC_2019_Training_Metadata.csv")))

    gt["dx"] = gt[list(ISIC2019_MAP) + ["SCC", "UNK"]].idxmax(axis=1)
    gt = gt[gt["dx"].isin(ISIC2019_MAP)].copy()
    gt["dx"] = gt["dx"].map(ISIC2019_MAP)
    gt = gt.rename(columns={"image": "image_id"})
    gt = gt[~gt["image_id"].isin(exclude_ids)]
    gt = gt.merge(meta[["image", "lesion_id"]].rename(columns={"image": "image_id"}), on="image_id", how="left")
    gt["lesion_id"] = gt["lesion_id"].fillna(gt["image_id"])

    index = {p.stem: p for p in data_dir.rglob("*.jpg")}
    gt["path"] = gt["image_id"].map(index)
    n_missing = int(gt["path"].isna().sum())
    if n_missing:
        print(f"Warning: dropping {n_missing} ISIC 2019 rows with no image file.")
        gt = gt.dropna(subset=["path"])
    gt["label"] = gt["dx"].map({c: i for i, c in enumerate(CLASS_CODES)}).astype(int)
    return gt[["lesion_id", "image_id", "dx", "label", "path"]].reset_index(drop=True)


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


def class_weights(labels: np.ndarray, num_classes: int = len(CLASS_CODES), power: float = 1.0) -> torch.Tensor:
    """(Inverse-frequency)**power weights, mean-normalised, to counter class imbalance.

    power=1 is full inverse frequency; power=0.5 is a softer square-root weighting that
    over-predicts rare classes less; power=0 disables weighting.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = (counts.sum() / (num_classes * np.maximum(counts, 1))) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def build_transforms(train: bool, img_size: int = IMG_SIZE,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> v2.Compose:
    """``mean``/``std`` must match what the backbone was pretrained with (e.g. tf_efficientnetv2
    and MaxViT use 0.5/0.5, ConvNeXt and torchvision models use the ImageNet statistics)."""
    normalize = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
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


def eval_transform_for(ckpt: dict) -> v2.Compose:
    """Evaluation transform matching a checkpoint (older checkpoints default to ImageNet stats)."""
    return build_transforms(False, ckpt["img_size"], tuple(ckpt.get("mean", IMAGENET_MEAN)),
                            tuple(ckpt.get("std", IMAGENET_STD)))


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
