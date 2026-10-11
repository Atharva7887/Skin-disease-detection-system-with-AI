"""Dermoscopy pre-processing used at inference (and optionally training) time.

Each function maps a PIL RGB image to a PIL RGB image and is module-level so that it can be
pickled into DataLoader worker processes on Windows.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

WORK_SIZE = 448  # process at a fixed working size, the models resize to 224 afterwards anyway


def _work(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB").resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR))


def shades_of_gray(im: Image.Image, power: float = 6.0) -> Image.Image:
    """Colour constancy: estimate the illuminant with a Minkowski norm and divide it out."""
    a = _work(im).astype(np.float32)
    illum = np.power(np.mean(np.power(a, power), axis=(0, 1)), 1.0 / power)
    gain = illum.mean() / np.maximum(illum, 1e-6)
    return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))


def remove_hair(im: Image.Image) -> Image.Image:
    """DullRazor-style: black-hat filter finds thin dark structures, which are then inpainted."""
    a = _work(im)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return Image.fromarray(cv2.inpaint(a, mask, 1, cv2.INPAINT_TELEA))


def clahe_lightness(im: Image.Image) -> Image.Image:
    """Local contrast enhancement on the lightness channel only (colours untouched)."""
    lab = cv2.cvtColor(_work(im), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def hair_then_color(im: Image.Image) -> Image.Image:
    return shades_of_gray(remove_hair(im))


PREPROCESSORS = {
    "none": None,
    "shades_of_gray": shades_of_gray,
    "hair_removal": remove_hair,
    "clahe": clahe_lightness,
    "hair+color": hair_then_color,
}
