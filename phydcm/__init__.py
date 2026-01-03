"""
PhyDCM: AI-assisted medical image classification toolkit.

This package is structured to keep import overhead minimal:
- Heavy dependencies (e.g., TensorFlow) are imported only when needed.
- Training utilities and model builders are available via lazy attributes.

Public API is preserved:
- Config
- PyHDCMPredictor
- get_datasets
- build_medvit_model
- train_model
- load_class_labels, load_trained_model, preprocess_image
"""
from __future__ import annotations

from .config import Config
from .predict import PyHDCMPredictor
from .utils import load_class_labels, preprocess_image, load_trained_model

__all__ = [
    "Config",
    "PyHDCMPredictor",
    "get_datasets",
    "build_medvit_model",
    "train_model",
    "load_class_labels",
    "load_trained_model",
    "preprocess_image",
]

def __getattr__(name: str):
    # Lazy attributes to avoid importing TensorFlow unless necessary.
    if name == "get_datasets":
        from .create_datasets import get_datasets
        return get_datasets
    if name == "build_medvit_model":
        from .medvit_model import build_medvit_model
        return build_medvit_model
    if name == "train_model":
        from .train import train_model
        return train_model
    raise AttributeError(f"module 'phydcm' has no attribute '{name}'")
