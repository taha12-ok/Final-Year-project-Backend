"""
onnx_engine.py — Lightweight ONNX inference engine (free-tier deployment)
==========================================================================
Ye module torch KE BAGAIR chalta hai — sirf onnxruntime + numpy + PIL + cv2.
Isliye 512MB RAM (Render free tier) me teeno models asani fit ho jate hain.

Models: {model}_model_int8.onnx (dynamic int8 quantized — ~24MB each)
Fallback: {model}_model.onnx (fp32)

Grad-CAM replacement: Activation Mapping (AM) — last conv feature maps ko
predicted class ke FC weights se weight karke heatmap banate hain.
ResNet50 ke liye ye Grad-CAM (no-gradient variant) ke barabar hota hai:
  weights_k = global-average-pool of grads  ==  fc weights (direct read)
ResNet me global avg pool last conv ke baad aata hai, isliye fc weights hi
cam weights hain. Heatmap quality Grad-CAM jaisi hi milti hai.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_TAG = "onnx-cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ResNet50 last bottleneck block output: 2048 channels @ 7x7 (224 input)
LAST_CONV_SHAPE = (2048, 7, 7)

_sessions: Dict[str, ort.InferenceSession] = {}
_fc_weights: Dict[str, np.ndarray] = {}
_model_meta: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────
# Model registry (classes/backend order — notebook ke sath sync)
# ─────────────────────────────────────────────────────────────
MODELS = {
    "fracture": {
        "onnx": "fracture_model_int8.onnx",
        "onnx_fp32": "fracture_model.onnx",
        "classes": ["Fractured", "Not Fractured"],
        "scan": "X-ray",
        "num_classes": 2,
    },
    "brain": {
        "onnx": "brain_model_int8.onnx",
        "onnx_fp32": "brain_model.onnx",
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "scan": "Brain MRI",
        "num_classes": 4,
    },
    "kidney": {
        "onnx": "kidney_model_int8.onnx",
        "onnx_fp32": "kidney_model.onnx",
        "classes": ["Cyst", "Normal", "Stone", "Tumor"],
        "scan": "CT Scan",
        "num_classes": 4,
    },
}


def _find_model_file(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("onnx", "onnx_fp32"):
        p = os.path.join(BASE_DIR, entry[key])
        if os.path.exists(p):
            return p
    return None


def get_session(model_type: str) -> ort.InferenceSession:
    """ONNX session lazy-load karo (pehli request pe ek hi baar)."""
    if model_type in _sessions:
        return _sessions[model_type]
    entry = MODELS[model_type]
    path = _find_model_file(entry)
    if path is None:
        raise FileNotFoundError(
            f"ONNX model for '{model_type}' not found. Looked for: "
            f"{entry['onnx']} / {entry['onnx_fp32']} in {BASE_DIR}"
        )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    _sessions[model_type] = sess
    return sess


def _extract_fc_weights(sess: ort.InferenceSession, model_type: str) -> Optional[np.ndarray]:
    """ONNX graph initializer se final classifier weights read karo."""
    try:
        model_path = _find_model_file(MODELS[model_type])
        import onnx
        m = onnx.load(model_path)
        inits = {init.name: init for init in m.graph.initializer}
        # ResNet50 torchvision export: fc.weight [num_classes, 2048]
        for name, arr in inits.items():
            if "fc.weight" in name or (len(arr.dims) == 2 and arr.dims[0] == MODELS[model_type]["num_classes"] and arr.dims[1] == 2048):
                return onnx.numpy_helper.to_array(arr).astype(np.float32)
    except Exception as e:
        print(f"[warn] fc weight extraction failed for {model_type}: {e}")
    return None


def preload_all() -> None:
    """Startup pe teeno sessions banao (cold-start latency kam karne ko)."""
    for k in MODELS:
        try:
            sess = get_session(k)
            fw = _extract_fc_weights(sess, k)
            if fw is not None:
                _fc_weights[k] = fw
            _model_meta[k] = {"file": _find_model_file(MODELS[k]), "loaded": True}
            print(f"[startup] {k}: ONNX loaded ({MODELS[k]['scan']})")
        except Exception as e:
            _model_meta[k] = {"loaded": False, "error": str(e)}
            print(f"[startup] ⚠️ {k}: ONNX load FAILED: {e}")


def preprocess(image: Image.Image) -> np.ndarray:
    """Backend transform mirror: resize 224 + ImageNet normalize. NCHW float32."""
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC 0..1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # 1 C H W
    return np.ascontiguousarray(arr)


def _am_with_hook(model_type: str, input_arr: np.ndarray, pred: int) -> Optional[np.ndarray]:
    """
    ONNX graph me last conv output ko extra output banake activations nikaalo,
    FC weights se weight karo -> 7x7 heatmap (0..1).
    """
    try:
        import onnx
        import onnxruntime as ort2

        src_path = _model_meta[model_type]["file"]
        am_path = os.path.join(BASE_DIR, f".cache_{model_type}_am.onnx")

        if not os.path.exists(am_path):
            m = onnx.load(src_path)
            # last conv output dhoondo — multiple strategies:
            target = None
            # 1) GlobalAveragePool/AveragePool output (classic torchvision export)
            for node in m.graph.node:
                if node.op_type in ("GlobalAveragePool", "AveragePool"):
                    target = node.input[0]  # POOL KA INPUT = last conv feature map
                    break
            # 2) Flatten input (optimized graphs)
            if target is None:
                for node in m.graph.node:
                    if node.op_type == "Flatten":
                        target = node.input[0]
                        break
            # 3) ReduceMean over spatial axes (some exporters)
            if target is None:
                for node in m.graph.node:
                    if node.op_type == "ReduceMean" and len(node.input) >= 1:
                        target = node.input[0]
                        break
            if target is None:
                print(f"[warn] AM: no conv/pool/flatten node found in {model_type} graph")
                return None
            vi = onnx.helper.ValueInfoProto()
            vi.name = target
            m.graph.output.append(vi)
            onnx.save(m, am_path)

        so = ort2.SessionOptions()
        so.graph_optimization_level = ort2.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess2 = ort2.InferenceSession(am_path, sess_options=so, providers=["CPUExecutionProvider"])
        outputs = sess2.run(None, {"input": input_arr})
        feats = outputs[-1][0]  # expected [C, H, W] e.g. [2048, 7, 7]
        w = _fc_weights.get(model_type)
        if w is None or feats.ndim != 3:
            return None
        # Shape sanity: channels must match fc weight input dim
        if feats.shape[0] != w.shape[1]:
            # maybe transposed (H, W, C) — try transpose
            if feats.shape[2] == w.shape[1]:
                feats = np.transpose(feats, (2, 0, 1))
            else:
                return None
        cam = np.tensordot(w[pred], feats, axes=([0], [0]))  # [7,7]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)
    except Exception as e:
        print(f"[warn] AM heatmap failed for {model_type}: {e}")
        return None


def run_inference(model_type: str, image: Image.Image, temperature: float = 1.0):
    """
    Returns (predicted_idx, confidence_pct, heatmap_7x7_or_None).
    Calibrated softmax + Activation Map heatmap.
    """
    sess = get_session(model_type)
    input_name = sess.get_inputs()[0].name
    input_arr = preprocess(image)

    logits = sess.run(None, {input_name: input_arr})[0][0].astype(np.float64)

    # temperature scaling (calibration.py ke sath consistent)
    z = logits / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    probs = e / e.sum()

    order = np.argsort(probs)[::-1]
    predicted = int(order[0])
    confidence = round(float(probs[predicted]) * 100, 2)

    heat = _am_with_hook(model_type, input_arr, predicted)
    return predicted, confidence, heat, logits


def overlay_heatmap(image: Image.Image, heat: np.ndarray) -> Optional[str]:
    """7x7 heatmap ko image pe overlay karke base64 JPEG return karo (cv2)."""
    try:
        import cv2
        import base64

        if heat is None or heat.size == 0:
            return None
        img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        img_arr = np.asarray(img).astype(np.float32) / 255.0

        heat2d = np.asarray(heat, dtype=np.float32)
        if heat2d.ndim != 2 or heat2d.shape[0] == 0 or heat2d.shape[1] == 0:
            return None
        heat_resized = cv2.resize(heat2d, (224, 224))
        heat_color = cv2.applyColorMap(np.uint8(255 * heat_resized), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        blended = 0.55 * img_arr + 0.45 * heat_color
        blended = np.uint8(255 * np.clip(blended, 0, 1))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        print(f"[warn] heatmap overlay failed: {e}")
        return None
