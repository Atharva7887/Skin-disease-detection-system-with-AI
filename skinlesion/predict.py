"""Inference wrapper used by the web app and the CLI."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .config import CLASS_NAMES
from .data import build_transforms
from .model import load_checkpoint


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.classes: list[str] = ckpt["classes"]
        self.transform = build_transforms(train=False, img_size=ckpt["img_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu()
        values, indices = probs.topk(min(top_k, len(self.classes)))
        return [
            {
                "code": self.classes[i],
                "name": CLASS_NAMES.get(self.classes[i], self.classes[i]),
                "probability": round(float(v), 4),
            }
            for v, i in zip(values, indices)
        ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify one image.")
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default="models/best.pt")
    args = ap.parse_args()
    with Image.open(args.image) as im:
        for row in Predictor(args.checkpoint).predict(im):
            print(f"{row['probability']:.1%}  {row['code']:6s} {row['name']}")
