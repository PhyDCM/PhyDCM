import tensorflow as tf
from tensorflow.keras import layers, models

def build_medvit_model(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape)

    # 🔷 Block 1: Convolutional stem
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # 🔷 Block 2: Vision Transformer-inspired encoding
    for _ in range(2):
        res = x
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)

        # تعديل: تحويل res ليكون نفس عدد القنوات (128)
        res = layers.Conv2D(128, (1, 1), padding='same')(res)
        res = layers.BatchNormalization()(res)

        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 🔷 Block 3: Deeper features
    for _ in range(2):
        res = x
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)

        # تعديل: تحويل res ليكون نفس عدد القنوات (256)
        res = layers.Conv2D(256, (1, 1), padding='same')(res)
        res = layers.BatchNormalization()(res)

        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 🔷 Flatten and Dense Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

__all__ = ['build_medvit_model']
