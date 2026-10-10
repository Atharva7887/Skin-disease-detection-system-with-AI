"""Model factory: ImageNet-pretrained backbones with a fresh 7-class head."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .config import CLASS_CODES

SUPPORTED = ("efficientnet_b0", "efficientnet_b2", "resnet50", "mobilenet_v3_large")
# Any other architecture can be used through timm as "timm/<name>", e.g.
# "timm/convnextv2_tiny.fcmae_ft_in22k_in1k" or "timm/eva02_small_patch14_224.mim_in22k".
TIMM_PREFIX = "timm/"


def build_model(name: str = "efficientnet_b0", pretrained: bool = True,
                num_classes: int = len(CLASS_CODES), dropout: float = 0.3) -> nn.Module:
    if name.startswith(TIMM_PREFIX):
        import timm

        return timm.create_model(name[len(TIMM_PREFIX):], pretrained=pretrained,
                                 num_classes=num_classes, drop_rate=dropout)
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED}")

    weights = "DEFAULT" if pretrained else None
    model = getattr(models, name)(weights=weights)

    if name.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "mobilenet_v3_large":
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:  # resnet50
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def head_parameters(model: nn.Module, name: str):
    """Parameters of the classification head (for the frozen-backbone warm-up)."""
    if name.startswith(TIMM_PREFIX):
        return model.get_classifier().parameters()
    return (model.fc if name == "resnet50" else model.classifier).parameters()


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    head = {id(p) for p in head_parameters(model, name)}
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable


def save_checkpoint(path, model: nn.Module, name: str, img_size: int, metrics: dict) -> None:
    torch.save(
        {
            "arch": name,
            "state_dict": model.state_dict(),
            "classes": CLASS_CODES,
            "img_size": img_size,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, device: str | torch.device = "cpu"):
    # weights_only=True: refuse to unpickle arbitrary objects from the file.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
