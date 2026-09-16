"""
calibration.py — Confidence Gate (Temperature Scaling)
=======================================================
Problem jo ye solve karta hai:
  Purane models har cheez pe 100% confidence dete the — chahe image random
  photo ho. Ye overconfidence galat calibration ki wajah se hota hai.

Solution:
  1. Temperature scaling — logits ko ek learned temperature T se divide karke
     softmax lagate hain. T > 1 => probabilities flatten (zyada honest).
     T Kaggle notebook ke validation set pe tune hota hai aur
     temperature.json me save hota hai.
  2. Decision rules:
     - confidence >= LOW_CONFIDENCE_THRESHOLD  => normal result
     - confidence <  LOW_CONFIDENCE_THRESHOLD  => "inconclusive" flag,
       UI warning, aur report me extra note.

Ye file backend ke run_inference se call hoti hai.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-model temperature file: temperature.json
# Format: {"fracture": {"temperature": 1.42}, "brain": {...}, "kidney": {...}}
TEMP_FILE = os.path.join(BASE_DIR, "temperature.json")

# Tumhari performance vs honesty ka balance:
# Is se neeche confidence => result "inconclusive" mark hoga.
LOW_CONFIDENCE_THRESHOLD = 60.0   # percent

_temp_cache: Optional[Dict[str, Any]] = None


def load_temperatures(force: bool = False) -> Dict[str, Any]:
    global _temp_cache
    if _temp_cache is not None and not force:
        return _temp_cache
    if os.path.exists(TEMP_FILE):
        try:
            with open(TEMP_FILE, "r") as f:
                _temp_cache = json.load(f)
        except Exception:
            _temp_cache = None
    return _temp_cache or {}


def get_temperature(model_type: str) -> float:
    """Model ka learned temperature (na mile to 1.0 = no change)."""
    data = load_temperatures()
    entry = data.get(model_type) or {}
    t = entry.get("temperature", 1.0)
    try:
        t = float(t)
    except (TypeError, ValueError):
        t = 1.0
    return max(0.5, min(5.0, t))


def calibrated_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax — numerically stable."""
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def evaluate_confidence(confidence_percent: float) -> Dict[str, Any]:
    """
    Confidence ke hisab se verdict deta hai.
    Backend response me 'reliability' block ke roop me jata hai.
    """
    c = float(confidence_percent)
    if c >= LOW_CONFIDENCE_THRESHOLD:
        return {
            "status": "ok",
            "inconclusive": False,
            "message": "",
        }
    return {
        "status": "inconclusive",
        "inconclusive": True,
        "message": (
            "Model is not confident enough for a reliable screening result. "
            "Please try a clearer, properly-exposed scan image — "
            "and confirm with a qualified doctor regardless."
        ),
    }
