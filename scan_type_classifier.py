"""
scan_type_classifier.py — Modality Classifier (X-ray / MRI / CT / photo / other)
================================================================================
Kya karta hai:
  Uploaded image ka scan-type detect karta hai — chhote CNN se (24k params,
  pure numpy inference, koi torch/dependency nahi).

  Ye DONO masle solve karta hai:
  1. Random photo / diagram / document ko scan samajh kar reject nahi hota
     (heuristic gate grayscale photos ko pass kar deta tha)
  2. Galat model pe scan dalna — e.g. brain MRI ko fracture model pe dalne se
     pehle hi 422 "scan_type_mismatch" milta hai (bina jhooti prediction ke)

Usage (backend me):
    from scan_type_classifier import classify_scan_type
    result = classify_scan_type(pil_image)
    # {"label": "xray|mri|ct|photo|other", "confidence": 0.0-1.0, "available": bool}

Weights: modality_cnn.json (train script: tools/train_scan_classifier.py)
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, "modality_cnn.json")

INPUT_SIZE = 128
CLASSES = ["xray", "mri", "ct", "photo", "other"]

# Confidence is se neeche => classifier unsure => "other" treat karo
MIN_CONFIDENCE = 0.60

_weights: Optional[Dict[str, np.ndarray]] = None


def _load_weights() -> Optional[Dict[str, np.ndarray]]:
    global _weights
    if _weights is not None:
        return _weights
    if not os.path.exists(WEIGHTS_FILE):
        return None
    try:
        with open(WEIGHTS_FILE, "r") as f:
            raw = json.load(f)
        _weights = {
            k: np.frombuffer(base64.b64decode(v), dtype=np.float32).reshape(shape)
            for k, shape, v in ((kk, tuple(sh), vv) for kk, sh, vv in
                                [(k, raw[k]["shape"], raw[k]["data"]) for k in raw])
        }
    except Exception:
        _weights = None
    return _weights


def is_available() -> bool:
    return _load_weights() is not None


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> (1, 128, 128) float32, ImageNet-ish normalize."""
    g = img.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5  # [-1, 1]
    return a[None, :, :]


def _conv2d(x: np.ndarray, w: np.ndarray, stride: int = 1) -> np.ndarray:
    """x: (C_in, H, W), w: (C_out, C_in, KH, KW) -> (C_out, H', W'). Valid padding."""
    c_in, h, wd = x.shape
    c_out, _, kh, kw = w.shape
    oh = (h - kh) // stride + 1
    ow = (wd - kw) // stride + 1
    # sliding windows via as_strided-free approach: einsum on strided view
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(c_in, oh, ow, kh, kw),
        strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2]),
    )
    out = np.einsum("cijhw,ochw->oij", windows, w, optimize=True)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def predict_logits(img: Image.Image) -> Optional[np.ndarray]:
    w = _load_weights()
    if w is None:
        return None
    x = preprocess(img)
    x = _relu(_conv2d(x, w["conv1.weight"], stride=2) + w["conv1.bias"][:, None, None])
    x = _relu(_conv2d(x, w["conv2.weight"], stride=2) + w["conv2.bias"][:, None, None])
    x = _relu(_conv2d(x, w["conv3.weight"], stride=2) + w["conv3.bias"][:, None, None])
    # global average pool -> (C,)
    feat = x.mean(axis=(1, 2))
    logits = w["fc.weight"] @ feat + w["fc.bias"]
    return logits


def classify_scan_type(img: Image.Image) -> Dict[str, Any]:
    """
    Returns {"label": str, "confidence": float, "available": bool}.
    available=False => weights file missing (backend ko fallback use karna chahiye).
    """
    logits = predict_logits(img)
    if logits is None:
        return {"label": "unknown", "confidence": 0.0, "available": False}
    z = logits.astype(np.float64)
    z -= z.max()
    e = np.exp(z)
    probs = e / e.sum()
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "confidence": float(probs[idx]),
        "available": True,
    }


# ─────────────────────────────────────────────────────────────
# Color-saturation guard (defense-in-depth, heuristic)
# ─────────────────────────────────────────────────────────────
def saturation_stats(img: Image.Image) -> dict:
    """Image ka colorfulness measure — colored photos ko scan se pehle hi pakro."""
    a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    sat = (a.max(-1) - a.min(-1)) / (a.max(-1) + 1e-6)
    return {
        "mean_sat": float(sat.mean()),
        "p90_sat": float(np.percentile(sat, 90)),
        "gray_ratio": float((sat < 0.08).mean()),
    }


# Ye threshold tuned hai: real X-ray/MRI/CT me sat ~0 (grayscale scan) hota hai,
# colored photos (mobile camera) me mean_sat > 0.15 common hai.
# Note: colored medical overlays (jaise glioma MRI) bhi colored hote hain — isliye
# ye sirf "photo" rejection me use hota hai jab CNN bhi unsure ho.
SAT_PHOTO_THRESHOLD = 0.15


def looks_like_color_photo(img: Image.Image) -> bool:
    s = saturation_stats(img)
    return s["mean_sat"] >= SAT_PHOTO_THRESHOLD
