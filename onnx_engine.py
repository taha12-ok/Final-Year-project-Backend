"""
onnx_engine.py — Lightweight ONNX inference engine (free-tier deployment)
==========================================================================
Ye module torch KE BAGAIR chalta hai — sirf onnxruntime + numpy + PIL.
Free-tier RAM profile (Back4App Containers 256MB / Render 512MB):
  - MAX_LOADED_MODELS env (default 3; Back4App image pe 2) — LRU eviction:
    sab se purana model session demand pe unload hota hai, dobara request pe
    khud load (~0.3s). Isliye 256MB me bhi teeno models available rehte hain.
  - ORT threads 1 + memory arena off — ORT ka resident RAM minimum.
  - AM (heatmap) session bhi per-model cache hota hai aur eviction ke sath
    clear hota hai; logits + activations EK hi forward pass me aate hain.

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
_am_sessions: Dict[str, ort.InferenceSession] = {}
_fc_weights: Dict[str, np.ndarray] = {}
_model_meta: Dict[str, Dict[str, Any]] = {}
_lru_order: list = []
MAX_LOADED = max(1, int(os.getenv("MAX_LOADED_MODELS", "3")))


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
    # free-tier tuning: 1 thread = sab se kam RAM; arena off = ORT apna memory pool chhota rakhta hai
    so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = os.getenv("ORT_MEM_ARENA", "0") == "1"
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    _sessions[model_type] = sess
    note_loaded(model_type)
    return sess


def note_loaded(model_type: str) -> None:
    """LRU mark + zaroorat par sab se purana session evict (256MB free tiers)."""
    if model_type in _lru_order:
        _lru_order.remove(model_type)
    _lru_order.append(model_type)
    while len(_lru_order) > MAX_LOADED:
        victim = _lru_order.pop(0)
        try:
            _sessions.pop(victim, None)
            _am_sessions.pop(victim, None)
            _model_meta[victim] = {
                "file": _find_model_file(MODELS[victim]), "loaded": False, "evicted": True,
            }
            print(f"[mem] '{victim}' evicted (MAX_LOADED_MODELS={MAX_LOADED})")
        except Exception as e:
            print(f"[mem] evict failed for {victim}: {e}")


def _extract_fc_weights(sess: ort.InferenceSession, model_type: str) -> Optional[np.ndarray]:
    """ONNX graph initializer se final classifier weights read karo."""
    try:
        model_path = _find_model_file(MODELS[model_type])
        import onnx
        m = onnx.load(model_path)
        inits = {init.name: init for init in m.graph.initializer}
        # ResNet50 torchvision export: fc.weight [num_classes, 2048]
        # int8 graphs me fc.weight_scale / fc.weight_zero_point bhi hote hain — skip!
        for name, arr in inits.items():
            if any(s in name for s in ("_scale", "_zero_point")):
                continue
            if "fc.weight" in name or (len(arr.dims) == 2 and arr.dims[0] == MODELS[model_type]["num_classes"] and arr.dims[1] == 2048):
                a = onnx.numpy_helper.to_array(arr).astype(np.float32)
                if a.ndim == 2:  # 1-D meta/garbage reject
                    return a
    except Exception as e:
        print(f"[warn] fc weight extraction failed for {model_type}: {e}")
    return None


def preload_all() -> None:
    """Startup pe (MAX_LOADED tak) sessions banao — cold-start latency kam."""
    for k in list(MODELS)[:MAX_LOADED]:
        try:
            sess = get_session(k)
            # NOTE: graph se fc weight extract YAHAN cache nahi karte —
            # int8/transposed graphs me layout alag ho sakta hai; lazy path
            # (_fc_weights_for) .npy sidecar ko priority deta hai (torch layout).
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


def _fc_weights_for(model_type: str) -> Optional[np.ndarray]:
    """fc weights: pehle .npy sidecar (retrained fp32 — authoritative), phir cache, phir graph."""
    if model_type not in _fc_weights:
        npy = os.path.join(BASE_DIR, f"{model_type}_fc.npy")
        if os.path.exists(npy):
            try:
                w = np.load(npy).astype(np.float32)
                if w.ndim == 2:
                    _fc_weights[model_type] = w
            except Exception as e:
                print(f"[warn] fc npy load failed for {model_type}: {e}")
    w = _fc_weights.get(model_type)
    if w is not None and w.ndim == 2:
        return w
    w = _extract_fc_weights(get_session(model_type), model_type)
    if w is not None and w.ndim == 2:
        _fc_weights[model_type] = w
    return w


def _build_am_graph(model_type: str) -> Optional[str]:
    """Src ONNX me last-conv output ko extra output banake .cache file likho (ek hi baar)."""
    src_path = _model_meta.get(model_type, {}).get("file") or _find_model_file(MODELS[model_type])
    if not src_path:
        print(f"[warn] AM: no ONNX file for {model_type}")
        return None
    am_path = os.path.join(BASE_DIR, f".cache_{model_type}_am.onnx")
    if os.path.exists(am_path):
        return am_path
    import onnx
    m = onnx.load(src_path)
    # last conv output dhoondo — multiple strategies:
    target = None
    # 1) GlobalAveragePool/AveragePool ka input (classic torchvision export)
    for node in m.graph.node:
        if node.op_type in ("GlobalAveragePool", "AveragePool"):
            target = node.input[0]
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
    return am_path


def _am_with_hook(model_type: str, input_arr: np.ndarray, pred: int) -> Optional[np.ndarray]:
    """
    Cached AM session — EK hi forward pass me logits + last-conv activations,
    FC weights se weight karke 7x7 heatmap (0..1). Eviction-safe.
    """
    try:
        sess2 = _am_sessions.get(model_type)
        if sess2 is None:
            am_path = _build_am_graph(model_type)
            if not am_path:
                return None
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
            so.inter_op_num_threads = 1
            so.enable_cpu_mem_arena = os.getenv("ORT_MEM_ARENA", "0") == "1"
            sess2 = ort.InferenceSession(am_path, sess_options=so, providers=["CPUExecutionProvider"])
            _am_sessions[model_type] = sess2
        w = _fc_weights_for(model_type)
        if w is None:
            print(f"[warn] AM: fc weights unavailable for {model_type}")
            return None
        # Layout normalize: torch layout [num_classes, 2048] chahiye (w[pred] = row).
        # Kuch graphs me initializer transposed [2048, num_classes] hota hai.
        nc = MODELS[model_type]["num_classes"]
        if w.ndim == 2 and w.shape == (2048, nc):
            w = np.ascontiguousarray(w.T)
        if w.ndim != 2 or w.shape != (nc, 2048):
            print(f"[warn] AM: fc weights {w.shape} unexpected for {model_type} (want ({nc}, 2048))")
            return None
        note_loaded(model_type)  # heatmap isi model ka hai — LRU me fresh rakho
        outputs = sess2.run(None, {sess2.get_inputs()[0].name: input_arr})
        feats = outputs[-1][0]  # appended output = last conv map [C, H, W]
        if feats.ndim != 3:
            print(f"[warn] AM: unexpected feats ndim {feats.shape} for {model_type}")
            return None
        # Shape sanity: channels must match fc weight input dim
        if feats.shape[0] != w.shape[1]:
            # maybe transposed (H, W, C) — try transpose
            if feats.shape[2] == w.shape[1]:
                feats = np.transpose(feats, (2, 0, 1))
            else:
                print(f"[warn] AM: feats {feats.shape} vs fc {w.shape} mismatch for {model_type}")
                return None
        cam = np.tensordot(w[pred], feats, axes=([0], [0]))  # [7,7]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)
    except Exception as e:
        import traceback
        print(f"[warn] AM heatmap failed for {model_type}: {e}\n{traceback.format_exc()}")
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
    """7x7 heatmap ko image pe overlay karke base64 JPEG (PIL — cv2-free)."""
    try:
        import base64
        import io

        if heat is None or heat.size == 0:
            return None
        heat2d = np.asarray(heat, dtype=np.float32)
        if heat2d.ndim != 2 or heat2d.shape[0] == 0 or heat2d.shape[1] == 0:
            return None
        img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        img_arr = np.asarray(img).astype(np.float32) / 255.0

        t = np.asarray(
            Image.fromarray(np.uint8(255 * np.clip(heat2d, 0, 1))).resize((224, 224), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        # jet-style colormap (cv2.COLORMAP_JET approximation)
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]
        heat_color = np.stack([
            np.interp(t, xs, [0.0, 0.0, 1.0, 1.0, 0.5]),
            np.interp(t, xs, [0.0, 1.0, 1.0, 0.0, 0.0]),
            np.interp(t, xs, [0.5, 1.0, 0.0, 0.0, 0.0]),
        ], axis=-1)

        blended = 0.55 * img_arr + 0.45 * heat_color
        blended = np.uint8(255 * np.clip(blended, 0, 1))
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[warn] heatmap overlay failed: {e}")
        return None
