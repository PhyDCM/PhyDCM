"""
Configuration for PhyDCM.

This module keeps defaults compatible with earlier releases, while allowing
environment-based overrides to support different deployments without code edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _env_path(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    """
    Central configuration container.

    Notes
    -----
    - DATA_DIR can be overridden using the environment variable `PHYDCM_DATA_DIR`.
    - OUTPUT_DIR can be overridden using `PHYDCM_OUTPUT_DIR`.
    - LOG_DIR can be overridden using `PHYDCM_LOG_DIR`.
    """
    # Training defaults (kept as in the original package unless overridden externally)
    EPOCHS: int = int(os.environ.get("PHYDCM_EPOCHS", "50"))
    BATCH_SIZE: int = int(os.environ.get("PHYDCM_BATCH_SIZE", "32"))
    IMAGE_SIZE: Tuple[int, int] = (
        int(os.environ.get("PHYDCM_IMAGE_H", "224")),
        int(os.environ.get("PHYDCM_IMAGE_W", "224")),
    )

    # Paths
    DATA_DIR: str = _env_path("PHYDCM_DATA_DIR", "/mnt/data/pyhdcm_medvit/data")
    OUTPUT_DIR: str = _env_path("PHYDCM_OUTPUT_DIR", str(Path.cwd() / "phydcm_outputs"))
    LOG_DIR: str = _env_path("PHYDCM_LOG_DIR", str(Path.cwd() / "phydcm_logs"))

    @staticmethod
    def get_num_classes(modality: str) -> int:
        mapping = {"mri": 4, "ct": 2, "pet": 3}
        modality = modality.lower().strip()
        if modality not in mapping:
            raise ValueError(f"Unknown modality '{modality}'. Expected one of: {', '.join(mapping)}.")
        return mapping[modality]

    @staticmethod
    def modality_train_dir(modality: str) -> str:
        return os.path.join(Config.DATA_DIR, modality, "train")

    @staticmethod
    def modality_val_dir(modality: str) -> str:
        return os.path.join(Config.DATA_DIR, modality, "val")
