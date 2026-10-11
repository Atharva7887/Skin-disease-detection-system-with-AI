"""Inference wrapper used by the web app and the CLI.

* ``checkpoint``: one path or several (a list, or a comma-separated string); several are ensembled
  by averaging softmax probabilities.
* ``tta``: average over the 8 flips/rotations of the image (small, consistent gain; 8x the compute).
* ``temperature``: calibration. The ensemble's average probabilities are sharpened/softened with
  ``p ** (1/T)`` (renormalised) so that stated confidence matches observed accuracy. It never changes
  which class is on top. Each prediction carries both ``probability`` (calibrated, for display) and
  ``raw_probability`` (uncalibrated); the risk flag is defined on the raw one.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .config import CLASS_NAMES
from .data import eval_transform_for
from .gradcam import grad_cam, overlay, to_data_url
from .model import load_checkpoint


def _as_paths(checkpoint) -> list[str]:
    if isinstance(checkpoint, (str, Path)):
        return [p.strip() for p in str(checkpoint).split(",") if p.strip()]
    return [str(p) for p in checkpoint]


def dihedral_views(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 symmetries of the square (4 rotations, each with and without a flip) of a (B,3,H,W) batch."""
    return [torch.rot90(x, k, (2, 3)) for k in range(4)] + [torch.rot90(x.flip(3), k, (2, 3)) for k in range(4)]


@torch.no_grad()
def model_probs(model, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    views = dihedral_views(x) if tta else [x]
    return torch.stack([torch.softmax(model(v).float(), dim=1) for v in views]).mean(0)


def apply_temperature(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return probs
    p = probs.clamp_min(1e-8) ** (1.0 / temperature)
    return p / p.sum(dim=-1, keepdim=True)


class Predictor:
    def __init__(self, checkpoint, device: str | None = None, cam_member: int = 0,
                 tta: bool = False, temperature: float = 1.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.members = []
        self.archs: list[str] = []
        self.cam_member = cam_member  # which ensemble member draws the Grad-CAM heatmap
        self.tta = tta
        self.temperature = temperature
        for path in _as_paths(checkpoint):
            model, ckpt = load_checkpoint(path, self.device)
            self.members.append((model, eval_transform_for(ckpt), ckpt["classes"]))
            self.archs.append(ckpt["arch"])
        if not self.members:
            raise ValueError("No checkpoint given")
        self.classes: list[str] = self.members[0][2]
        if any(m[2] != self.classes for m in self.members):
            raise ValueError("Checkpoints were trained with different class lists")

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        image = image.convert("RGB")
        raw = torch.stack([
            model_probs(model, tf(image).unsqueeze(0).to(self.device), self.tta)[0].cpu()
            for model, tf, _ in self.members
        ]).mean(0)
        shown = apply_temperature(raw, self.temperature)
        order = shown.argsort(descending=True)[: min(top_k, len(self.classes))]
        return [
            {
                "code": self.classes[i],
                "name": CLASS_NAMES.get(self.classes[i], self.classes[i]),
                "probability": round(float(shown[i]), 4),
                "raw_probability": round(float(raw[i]), 4),
            }
            for i in order.tolist()
        ]

    def heatmap(self, image: Image.Image, class_code: str) -> str:
        """Grad-CAM overlay for ``class_code`` as a JPEG data URL.

        Drawn from ONE member (``cam_member``, default: the first checkpoint), not averaged:
        on ISIC 2018, EfficientNetV2-S's Grad-CAM put 82% of its mass in the outer border ring
        (a feature-map edge artefact: blanking the border costs only ~3 points of accuracy),
        while ConvNeXt-V2's stays centred on the lesion (23% in the ring, which is 50% of the
        area). List the best-behaved model first in ``MODEL_PATH``.
        """
        model, tf, _ = self.members[self.cam_member]
        x = tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        cam = grad_cam(model, self.archs[self.cam_member], x, self.classes.index(class_code))
        return to_data_url(overlay(image, cam))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify one image.")
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default="models/best.pt", help="path, or comma-separated paths to ensemble")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    with Image.open(args.image) as im:
        for row in Predictor(args.checkpoint, tta=args.tta, temperature=args.temperature).predict(im):
            print(f"{row['probability']:.1%}  {row['code']:6s} {row['name']}")
