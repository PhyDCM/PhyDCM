from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_datasets(
    train_dir: str,
    val_dir: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    class_mode: str = 'categorical'
):
    """
    إعداد بيانات التدريب والتحقق باستخدام ImageDataGenerator.

    Args:
        train_dir (str): مسار مجلد التدريب.
        val_dir (str): مسار مجلد التحقق.
        image_size (tuple): حجم الصور المطلوب.
        batch_size (int): عدد الصور في كل دفعة.
        class_mode (str): نوع التوسيم، غالبًا 'categorical' للتصنيف.

    Returns:
        train_generator: مولّد بيانات التدريب.
        val_generator: مولّد بيانات التحقق.
    """

    # Augmentation لبيانات التدريب
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # إعادة تحجيم فقط لبيانات التحقق
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # تحميل بيانات التدريب
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )

    # تحميل بيانات التحقق
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    return train_generator, val_generator
