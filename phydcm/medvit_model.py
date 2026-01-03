"""
MedViT-style model definition used by PhyDCM.

TensorFlow is imported lazily to reduce the package import cost for inference-only
workflows that do not need model construction.
"""
from __future__ import annotations

from ._lazy import require_import


def build_medvit_model(input_shape=(224, 224, 3), num_classes: int = 4, dropout_rate: float = 0.3):
    tf = require_import("tensorflow", purpose="building Keras models")
    layers = tf.keras.layers
    models = tf.keras.models

    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional stem
    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Lightweight residual blocks
    for _ in range(2):
        res = x
        x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res])
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


__all__ = ["build_medvit_model"]
