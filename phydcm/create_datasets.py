"""
Dataset utilities for PhyDCM.

TensorFlow is imported lazily to reduce import overhead for inference-only usage.
"""
from __future__ import annotations

from typing import Tuple, Any

from ._lazy import require_import


def get_datasets(
    train_dir: str,
    val_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    class_mode: str = "categorical",
) -> tuple[Any, Any]:
    """
    Create training and validation generators using Keras' ImageDataGenerator.

    Parameters
    ----------
    train_dir:
        Training directory structured as class subfolders.
    val_dir:
        Validation directory structured as class subfolders.
    image_size:
        Target image size (H, W).
    batch_size:
        Batch size.
    class_mode:
        Keras generator class_mode (e.g., 'categorical', 'binary').

    Returns
    -------
    (train_generator, val_generator)
    """
    tf = require_import("tensorflow", purpose="dataset construction and training")
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False,
    )

    return train_generator, val_generator


__all__ = ["get_datasets"]
