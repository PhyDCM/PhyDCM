"""
Training utilities for PhyDCM (MedViT-style CNN/ViT hybrid).

TensorFlow is imported lazily to keep library import overhead low.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ._console import info, success
from ._lazy import require_import
from .config import Config
from .create_datasets import get_datasets
from .medvit_model import build_medvit_model


def train_model(
    modality: str,
    *,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> str:
    """
    Train a modality-specific model.

    Parameters
    ----------
    modality:
        One of: 'mri', 'ct', 'pet'.
    data_dir:
        Root data directory containing modality/train and modality/val.
        Defaults to Config.DATA_DIR.
    output_dir:
        Directory to store exported models. Defaults to Config.OUTPUT_DIR.
    log_dir:
        Directory to store CSV logs. Defaults to Config.LOG_DIR.
    epochs:
        Overrides Config.EPOCHS.
    batch_size:
        Overrides Config.BATCH_SIZE.

    Returns
    -------
    str
        Path to the best checkpoint model ('.keras').
    """
    modality = modality.lower().strip()
    num_classes = Config.get_num_classes(modality)

    data_root = data_dir or Config.DATA_DIR
    out_root = output_dir or Config.OUTPUT_DIR
    logs_root = log_dir or Config.LOG_DIR
    epochs = int(epochs or Config.EPOCHS)
    batch_size = int(batch_size or Config.BATCH_SIZE)

    train_dir = os.path.join(data_root, modality, "train")
    val_dir = os.path.join(data_root, modality, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    tf = require_import("tensorflow", purpose="model training")

    Path(out_root).mkdir(parents=True, exist_ok=True)
    Path(logs_root).mkdir(parents=True, exist_ok=True)

    info(f"Training started: modality={modality}, classes={num_classes}, epochs={epochs}, batch_size={batch_size}")

    train_ds, val_ds = get_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=Config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical" if num_classes > 2 else "binary",
    )

    model = build_medvit_model(input_shape=(*Config.IMAGE_SIZE, 3), num_classes=num_classes)

    loss = "categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=["accuracy"],
    )

    best_path = os.path.join(out_root, f"{modality}_best_model.keras")
    final_path = os.path.join(out_root, f"{modality}_final_model.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(logs_root, f"{modality}_training.csv")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_path)
    success(f"Training completed. Best checkpoint: {best_path}")
    success(f"Final model exported: {final_path}")
    return best_path


__all__ = ["train_model"]
