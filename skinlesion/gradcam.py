"""Grad-CAM: which image regions pushed the model towards a class.

A heatmap is an explanation of the model's behaviour, not a lesion outline and not a diagnosis.
It is useful mainly as a sanity check (e.g. the model attending to a ruler or hair instead of
the lesion).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .model import TIMM_PREFIX


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last module whose output is a (B, C, H, W) feature map."""
    if arch.startswith(TIMM_PREFIX):
        for attr in ("stages", "blocks"):  # ConvNeXt, MaxViT -> stages; timm EfficientNet -> blocks
            if hasattr(model, attr):
                return getattr(model, attr)[-1]
        if hasattr(model, "layer4"):  # timm ResNet
            return model.layer4[-1]
    elif arch == "resnet50":
        return model.layer4[-1]
    elif hasattr(model, "features"):  # torchvision EfficientNet / MobileNet
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer known for '{arch}'")


def grad_cam(model: nn.Module, arch: str, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Normalised (0..1) heatmap at the input resolution for one (1, 3, H, W) tensor."""
    layer = target_layer(model, arch)
    store: dict = {}

    def fwd_hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(False)
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp_min(1e-8)).cpu().numpy()


def _colormap(v: np.ndarray) -> np.ndarray:
    """Blue -> cyan -> yellow -> red, v in 0..1, returns uint8 (H, W, 3)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, size: int = 384) -> Image.Image:
    """Blend the heatmap over the (resized) image; the heatmap is resampled to that size."""
    image = image.convert("RGB")
    image = image.resize((size, int(size * image.height / image.width)) if image.width >= image.height
                         else (int(size * image.width / image.height), size))
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heat = _colormap(np.asarray(cam_img, dtype=np.float32) / 255.0)
    # Fade the colour in with intensity so low-attention areas keep the original pixels.
    a = (np.asarray(cam_img, dtype=np.float32) / 255.0)[..., None] * alpha * 2
    blended = np.asarray(image, dtype=np.float32) * (1 - a) + heat * a
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
