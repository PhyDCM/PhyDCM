"""
Inference utilities for PhyDCM.

This module is designed to be lightweight:
- No TensorFlow import at module import time
- Models are loaded lazily (or at initialization) from a model directory
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Union

import numpy as np

from ._console import info, success, warn, error
from .utils import preprocess_image, load_class_labels, load_trained_model


class PyHDCMPredictor:
    """
    Predictor for modality-specific classification models (MRI/CT/PET).

    Parameters
    ----------
    model_dir:
        Directory containing trained Keras models and label JSON files.
        If None, uses the packaged `phydcm/models` directory.
    img_size:
        Input size expected by the model (H, W).
    scan_type_filter:
        If provided, only these modalities are considered (e.g., ['mri', 'ct']).
    auto_load:
        If True, attempts to load available models at initialization.
        If False, you may call `load_models()` explicitly.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        img_size: Tuple[int, int] = (224, 224),
        scan_type_filter: Optional[list[str]] = None,
        auto_load: bool = True,
    ) -> None:
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(current_dir, "models")
        else:
            self.model_dir = model_dir

        self.img_size = img_size
        self.models: Dict[str, Any] = {}
        self.labels: Dict[str, Dict[str, str]] = {}

        self._allowed = None
        if scan_type_filter is not None:
            self._allowed = {s.lower().strip() for s in scan_type_filter}

        if auto_load:
            self.load_models()

    def _iter_modalities(self) -> list[str]:
        base = ["mri", "ct", "pet"]
        if self._allowed is None:
            return base
        return [m for m in base if m in self._allowed]

    def _candidate_model_paths(self, modality: str) -> list[Path]:
        """
        Supported model naming patterns (first match wins):
        - <modality>_best_model.keras
        - <modality>_final_model.keras
        - <modality>.keras
        - <modality>.h5
        """
        d = Path(self.model_dir)
        return [
            d / f"{modality}_best_model.keras",
            d / f"{modality}_final_model.keras",
            d / f"{modality}.keras",
            d / f"{modality}.h5",
        ]

    def _labels_path(self, modality: str) -> Path:
        return Path(self.model_dir) / f"{modality}_labels.json"

    def load_models(self) -> None:
        """Load all available models and labels from `model_dir`."""
        info("PhyDCM: scanning for available models...")

        for modality in self._iter_modalities():
            # Load labels (optional but recommended)
            labels_path = self._labels_path(modality)
            if labels_path.exists():
                try:
                    self.labels[modality] = load_class_labels(str(labels_path))
                except Exception as e:
                    warn(f"PhyDCM: labels could not be loaded for '{modality}': {e}")
            else:
                warn(f"PhyDCM: labels file not found for '{modality}' ({labels_path.name}).")

            # Load model
            model_path = None
            for cand in self._candidate_model_paths(modality):
                if cand.exists():
                    model_path = cand
                    break

            if model_path is None:
                error(f"PhyDCM: model not found for '{modality}' in '{self.model_dir}'.")
                continue

            try:
                self.models[modality] = load_trained_model(str(model_path))
                success(f"PhyDCM: model loaded for '{modality}' ({model_path.name}).")
            except Exception as e:
                error(f"PhyDCM: failed to load model for '{modality}': {e}")

    def predict(
        self,
        image_path: str,
        scan_type: str,
        *,
        return_probabilities: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Classify an image and return a diagnosis.

        Parameters
        ----------
        image_path:
            Path to a medical image (common raster formats or DICOM).
        scan_type:
            One of: 'mri', 'ct', 'pet'.
        return_probabilities:
            If True, returns probabilities for all classes.

        Returns
        -------
        str or dict
            Diagnosis string, or a dictionary with diagnosis, confidence and probabilities.
        """
        scan_type = scan_type.lower().strip()
        if scan_type not in self.models:
            raise ValueError(
                f"Scan type '{scan_type}' is not loaded. "
                f"Available: {', '.join(self.models.keys()) or 'none'}."
            )

        image = preprocess_image(image_path, self.img_size)

        try:
            preds = self.models[scan_type].predict(image, verbose=0)
        except TypeError:
            # Some backends do not accept verbose
            preds = self.models[scan_type].predict(image)

        preds = np.asarray(preds).reshape(-1)
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        label_map = self.labels.get(scan_type, {})
        diagnosis = label_map.get(str(idx), str(idx))

        if not return_probabilities:
            return diagnosis

        probabilities = {label_map.get(str(i), str(i)): float(p) for i, p in enumerate(preds)}
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def get_diagnosis(self, image_path: str, scan_type: str) -> str:
        """Convenience wrapper that returns diagnosis only."""
        return str(self.predict(image_path, scan_type, return_probabilities=False))

    def get_confidence(self, image_path: str, scan_type: str) -> float:
        """Convenience wrapper that returns confidence only."""
        out = self.predict(image_path, scan_type, return_probabilities=True)
        assert isinstance(out, dict)
        return float(out["confidence"])

    def available_scan_types(self) -> list[str]:
        """Return a list of loaded modalities."""
        return sorted(self.models.keys())


__all__ = ["PyHDCMPredictor"]
