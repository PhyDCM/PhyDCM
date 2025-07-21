import os
import json
import numpy as np
from .utils import preprocess_image, load_class_labels, load_trained_model

class PyHDCMPredictor:
    def __init__(self, model_dir="outputs", img_size=(224, 224), scan_type_filter=None):
        self.model_dir = model_dir
        self.img_size = img_size
        self.models = {}
        self.labels = {}
        self._load_all_models(scan_type_filter)

    def _load_all_models(self, scan_type_filter=None):
        scan_types = ['mri', 'ct', 'pet']
        if scan_type_filter:
            scan_types = [scan_type_filter]

        for scan in scan_types:
            model_path = os.path.join(self.model_dir, f"{scan}_best_model.keras")
            labels_path = os.path.join(self.model_dir, f"{scan}_labels.json")
            try:
                self.models[scan] = load_trained_model(model_path)
                self.labels[scan] = load_class_labels(labels_path)
                print(f"✅ تم تحميل نموذج {scan} بنجاح")
            except FileNotFoundError as e:
                print(f"⚠️ {e}")

    def predict(self, image_path, scan_type, show_all=False, save_to_file=False):
        if scan_type not in self.models:
            raise ValueError(f"❌ نوع الفحص غير مدعوم أو النموذج غير محمّل: {scan_type}")
        
        img = preprocess_image(image_path, self.img_size)
        model = self.models[scan_type]
        preds = model.predict(img)

        pred_class_index = np.argmax(preds)
        pred_class_name = self.labels[scan_type].get(str(pred_class_index), "غير معروف")
        confidence = preds[0][pred_class_index]

        result = {
            "prediction": pred_class_name,
            "confidence": float(confidence)
        }

        if show_all:
            print("\n📊 جميع التوقعات:")
            for i, prob in enumerate(preds[0]):
                label = self.labels[scan_type].get(str(i), f"Unknown {i}")
                print(f"  - {label}: {prob:.3f}")

        if save_to_file:
            out_path = os.path.join(self.model_dir, "prediction_result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 تم حفظ النتيجة في: {out_path}")

        return result

    def __str__(self):
        return f"<PyHDCMPredictor models loaded: {list(self.models.keys())}>"

# --- للتنفيذ المباشر ---
if __name__ == "__main__":
    predictor = PyHDCMPredictor(scan_type_filter="mri")
    test_image_path = "data/pet/val/follicular_lymphoma/image_11.bmp"  # ← غيّره حسب بياناتك

    result = predictor.predict(
        test_image_path,
        scan_type="pet",
        show_all=True,       # ← يعرض كل احتمالات التصنيف
        save_to_file=True    # ← يحفظ النتيجة في ملف
    )

    print(f"\n🧠 التشخيص النهائي: {result['prediction']} بنسبة ثقة {result['confidence']:.2f}")
