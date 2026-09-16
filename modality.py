"""
modality.py — Medical Scan Gate
================================
Ye module decide karta hai ke uploaded image sach me medical scan (X-ray / MRI / CT)
hai ya koi normal photo (selfie, hand ki tasveer, landscape waghera).

Kaam kaise karta hai:
1. Image se 7 simple features nikalta hai (grayscale ratio, saturation, edges, ...)
2. Trained logistic-regression gate (modality_gate.json — Kaggle notebook se aata hai)
   se score nikalta hai: score >= threshold  -> scan hai
3. Agar gate file mojood nahi (retraining se pehle) to built-in heuristic fallback
   chalti hai taake server kabhi crash na ho.

IMPORTANT: Ye module inference time pe SIRF numpy + PIL use karta hai —
na torch, na sklearn. Isliye lightweight aur fast hai.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GATE_FILE = os.path.join(BASE_DIR, "modality_gate.json")

FEATURE_NAMES = [
    "gray_ratio",     # kitna % image grayscale jaisi hai (scans ~1.0, photos kam)
    "sat_mean",       # average saturation (photos high, scans low)
    "sat_p90",        # 90th percentile saturation
    "edge_density",   # kitne % pixels strong edges hain (scans me bohat)
    "edge_strength",  # gradient magnitude ka mean (scans sharp/textured)
    "blue_minus_red", # color cast (photos me visible, scans ~0)
    "lap_std",        # laplacian variance — texture/sharpness
]

_gate_cache: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────
# Gate loading (modality_gate.json — Kaggle notebook ka output)
# ─────────────────────────────────────────────────────────────
def load_gate(force: bool = False) -> Optional[Dict[str, Any]]:
    """modality_gate.json ko load/cache karo. Trained gate nahi to None."""
    global _gate_cache
    if _gate_cache is not None and not force:
        return _gate_cache if _gate_cache.get("source") == "trained" else None
    if not os.path.exists(GATE_FILE):
        return None
    try:
        with open(GATE_FILE, "r") as f:
            data = json.load(f)
        required = {"weights", "bias", "mu", "sigma", "threshold"}
        if not required.issubset(data.keys()):
            return None
        data["source"] = "trained"
        _gate_cache = data
        return data
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Feature extraction (pure numpy + PIL)
# ─────────────────────────────────────────────────────────────
def compute_features(img: Image.Image) -> np.ndarray:
    """
    Image (PIL) se 7-dim feature vector nikalta hai.
    Train (notebook) aur inference (backend) DONO yehi same function
    wala logic use karte hain — isliye distributions match hote hain.
    """
    img = img.convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # 1) grayscale ratio — kitne pixels near-neutral hain
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    sat = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), 0.0)
    gray_ratio = float((sat < 0.08).mean())

    # 2,3) saturation stats
    sat_mean = float(sat.mean())
    sat_p90 = float(np.percentile(sat, 90))

    # 4,5) edges (Sobel magnitude)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_density = float((mag > 0.12).mean())
    edge_strength = float(mag.mean())

    # 6) color cast
    blue_minus_red = float((b - r).mean())

    # 7) laplacian variance (texture/sharpness)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1]
        - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    lap_std = float(lap.std())

    return np.array(
        [gray_ratio, sat_mean, sat_p90, edge_density, edge_strength,
         blue_minus_red, lap_std],
        dtype=np.float64,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ─────────────────────────────────────────────────────────────
# Main check — backend isko call karta hai
# ─────────────────────────────────────────────────────────────
def check_image(img: Image.Image) -> Dict[str, Any]:
    """
    Returns:
      passed (bool)  — True => scan jaisi lagti hai, model ko jane do
      score (float)  — 0..1 (1 = sure scan)
      reason (str)   — user-friendly wajah (reject hone par)
      source (str)   — "trained" ya "heuristic"
      features(dict) — debugging ke liye
    """
    try:
        feats = compute_features(img)
    except Exception:
        return {"passed": False, "score": 0.0, "source": "error",
                "reason": "Image could not be read properly.", "features": {}}

    fdict = {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, feats)}

    gate = load_gate()
    if gate is not None:
        mu = np.array(gate["mu"], dtype=np.float64)
        sigma = np.array(gate["sigma"], dtype=np.float64)
        w = np.array(gate["weights"], dtype=np.float64)
        threshold = float(gate.get("threshold", 0.5))
        z = float(((feats - mu) / sigma) @ w) + float(gate["bias"])
        score = _sigmoid(z)
        passed = score >= threshold
        reason = "" if passed else (
            "This image does not look like a medical scan. "
            "Please upload a proper X-ray / MRI / CT image."
        )
        return {"passed": passed, "score": round(score, 4), "source": "trained",
                "reason": reason, "features": fdict}

    # ── Heuristic fallback (jab tak trained gate deploy na ho) ──
    gray_ratio, sat_mean = fdict["gray_ratio"], fdict["sat_mean"]
    edge_density = fdict["edge_density"]
    looks_like_scan = (
        gray_ratio >= 0.80
        and sat_mean <= 0.25
        and edge_density >= 0.03
    )
    if looks_like_scan:
        return {"passed": True, "score": 0.75, "source": "heuristic",
                "reason": "", "features": fdict}
    return {
        "passed": False, "score": 0.25, "source": "heuristic",
        "reason": ("This image does not look like a medical scan "
                   "(it appears to be a regular photo). "
                   "Please upload a proper X-ray / MRI / CT image."),
        "features": fdict,
    }
