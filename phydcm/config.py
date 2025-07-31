# phydcm/config.py
import os

# إعدادات التدريب
EPOCHS = 50
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# إعدادات المسارات
DATA_DIR = "/mnt/data/pyhdcm_medvit/data"
MRI_TRAIN_DIR = f"{DATA_DIR}/mri/train"
MRI_VAL_DIR = f"{DATA_DIR}/mri/val"
CT_TRAIN_DIR = f"{DATA_DIR}/ct/train"
CT_VAL_DIR = f"{DATA_DIR}/ct/val"
PET_TRAIN_DIR = f"{DATA_DIR}/pet/train"
PET_VAL_DIR = f"{DATA_DIR}/pet/val"

# حفظ النموذج والتسجيلات
MODEL_DIR = "/mnt/data/pyhdcm_medvit/outputs"
LOG_DIR = "/mnt/data/pyhdcm_medvit/logs"

class Config:
    DATA_DIR = r"C:\Users\lenovo\Desktop\phydcm\data"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.3
    
    # تعديل المسار ليكون داخل مجلد phydcm/models
    @staticmethod
    def get_output_dir():
        """in the models folder path"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "models")
    
    OUTPUT_DIR = get_output_dir()
    LOG_DIR = r"C:\Users\lenovo\Desktop\phydcm\logs"

    @staticmethod
    def get_num_classes(modality: str) -> int:
        # ترجع عدد الفئات حسب النوع
        mapping = {
            'mri': 4,
            'ct': 2,
            'pet': 3
        }
        if modality not in mapping:
            raise ValueError(f"Error: Unknown mode type: {modality}")
        return mapping[modality]