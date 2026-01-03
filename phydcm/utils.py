"""
Utility functions for PhyDCM.

Design goals:
- Minimal import overhead (no TensorFlow import at module import time)
- Robust image loading for common raster formats and DICOM
- Clear, academic-grade error messages
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
from PIL import Image

from ._console import debug, warn
from ._lazy import optional_import, require_import


def is_dicom_file(filepath: str) -> bool:
    """
    Detect whether a file is DICOM, regardless of extension.

    This checks:
    - Common DICOM extensions
    - The 'DICM' magic marker at byte offset 128 (when present)
    - A few lightweight heuristic patterns for some non-standard DICOM files
    """
    try:
        fp = str(filepath)
        if fp.lower().endswith((".dcm", ".dicom")):
            return True

        with open(fp, "rb") as f:
            header = f.read(132)

        if len(header) >= 132 and header[128:132] == b"DICM":
            return True

        # Heuristic signatures for some DICOM variants without the marker.
        # This is intentionally conservative to avoid false positives.
        first_bytes = header[:4]
        return first_bytes in (b"\x08\x00\x00\x00", b"\x10\x00\x00\x00", b"\x20\x00\x00\x00", b"\x28\x00\x00\x00")
    except Exception:
        return False


def _apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply linear windowing to a DICOM image."""
    low = center - (width / 2.0)
    high = center + (width / 2.0)
    img = np.clip(image.astype(np.float32), low, high)
    img = (img - low) / max(high - low, 1e-6)
    return img


def load_dicom_image(filepath: str) -> np.ndarray:
    """
    Load a DICOM file and return an 8-bit RGB image array.

    If window parameters are present (WindowCenter/WindowWidth), they are applied.
    Otherwise, a robust min-max normalization is used.
    """
    pydicom = require_import("pydicom", purpose="reading DICOM images")
    ds = pydicom.dcmread(filepath, force=True)

    if not hasattr(ds, "pixel_array"):
        raise ValueError("The DICOM file does not contain pixel data.")

    img = ds.pixel_array

    # Convert to float for processing
    img = img.astype(np.float32)

    # Apply rescale if present
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        try:
            img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        except Exception:
            pass

    # Windowing
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        try:
            # These fields can be MultiValue
            wc = ds.WindowCenter[0] if hasattr(ds.WindowCenter, "__len__") else ds.WindowCenter
            ww = ds.WindowWidth[0] if hasattr(ds.WindowWidth, "__len__") else ds.WindowWidth
            img = _apply_window(img, float(wc), float(ww))
        except Exception:
            # Fallback to min-max normalization
            img = (img - np.min(img)) / max(np.ptp(img), 1e-6)
    else:
        img = (img - np.min(img)) / max(np.ptp(img), 1e-6)

    img8 = (img * 255.0).clip(0, 255).astype(np.uint8)

    # Convert grayscale to RGB
    if img8.ndim == 2:
        img8 = np.stack([img8, img8, img8], axis=-1)
    elif img8.ndim == 3 and img8.shape[-1] == 1:
        img8 = np.repeat(img8, 3, axis=-1)

    return img8


def preprocess_image(filepath: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image (raster or DICOM), resize, and return a float32 tensor in [0, 1].

    Returns:
        np.ndarray of shape (1, H, W, 3)
    """
    fp = str(filepath)
    try:
        if is_dicom_file(fp):
            img = load_dicom_image(fp)
            pil = Image.fromarray(img)
        else:
            pil = Image.open(fp).convert("RGB")

        pil = pil.resize(target_size, Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}") from e


def load_class_labels(json_path: str) -> Dict[str, str]:
    """
    Load class labels from a JSON file.

    The JSON is expected to map numeric or string indices to human-readable labels.
    """
    jp = Path(json_path)
    if not jp.exists():
        raise FileNotFoundError(f"Labels file not found: {json_path}")

    with jp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def load_trained_model(model_path: str) -> Any:
    """
    Load a trained Keras model.

    Notes:
    - TensorFlow/Keras is imported lazily to keep `import phydcm` lightweight.
    - The function raises an informative ImportError if the backend is unavailable.
    """
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Prefer TensorFlow Keras when available (common for .keras exports).
    tf = optional_import("tensorflow")
    if tf is not None:
        debug("TensorFlow detected; loading model via tensorflow.keras.")
        return tf.keras.models.load_model(str(mp))

    # Fallback: standalone Keras (Keras 3). Requires a configured backend.
    keras = optional_import("keras")
    if keras is not None:
        warn("TensorFlow is not available; attempting to load the model via standalone 'keras'.")
        return keras.models.load_model(str(mp))

    raise ImportError(
        "A Keras backend is required to load models. Install 'tensorflow' "
        "or a compatible 'keras' backend."
    )


__all__ = [
    "is_dicom_file",
    "load_dicom_image",
    "preprocess_image",
    "load_class_labels",
    "load_trained_model",
]
