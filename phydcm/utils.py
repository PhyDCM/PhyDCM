import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import json

from PIL import Image

import cv2
import numpy as np
import os
import pydicom
from PIL import Image

def is_dicom_file(filepath):
    """يتأكد هل الملف هو DICOM حتى لو ما بي امتداد"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(132)
            return header[-4:] == b'DICM'
    except Exception:
        return False

def preprocess_image(image_path, img_size=(224, 224)):
    """تحميل وتحويل صورة (حتى لو بدون امتداد) إلى شكل مناسب للنموذج"""

    try:
        # تحقق إذا كان DICOM (حتى لو بدون .dcm)
        if is_dicom_file(image_path):
            dicom_data = pydicom.dcmread(image_path)
            image = dicom_data.pixel_array

            # تحويل لصورة 3 قنوات
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[-1] == 1:
                image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
            else:
                image = image.astype(np.uint8)
        else:
            # صورة عادية (bmp، jpg، png... إلخ)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("⚠️ فشل OpenCV في تحميل الصورة، سيتم تجربة PIL")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"⚠️ خطأ أثناء محاولة فتح الصورة: {e} - سيتم استخدام PIL")
        try:
            pil_img = Image.open(image_path).convert("RGB")
            image = np.array(pil_img)
        except Exception as e2:
            raise FileNotFoundError(f"❌ تعذر تحميل الصورة نهائيًا: {e2}")

    image = cv2.resize(image, img_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # إضافة بعد الدفعة
    return image



def load_class_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {str(k): v for k, v in data.items()}  # 🔥 حول كل مفتاح إلى str

def load_trained_model(model_path):
    """تحميل نموذج مدرب"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ملف النموذج غير موجود: {model_path}")
    model = load_model(model_path)
    return model
