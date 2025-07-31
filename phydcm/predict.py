import os
import numpy as np
from .utils import preprocess_image, load_class_labels, load_trained_model

class PyHDCMPredictor:
    def __init__(self, model_dir=None, img_size=(224, 224), scan_type_filter=None):
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(current_dir, "models")
        else:
            self.model_dir = model_dir
            
        self.img_size = img_size
        self.models = {}
        self.labels = {}
        
        print("🔄 Loading medical classification models...")
        self._load_models(scan_type_filter)
        
        if not self.models:
            print("⚠️ No models loaded. Please check model directory.")

    def _load_models(self, scan_type_filter=None):
        scan_types = ['mri', 'ct', 'pet'] if not scan_type_filter else [scan_type_filter]
        
        for scan in scan_types:
            model_path = os.path.join(self.model_dir, f"{scan}_best_model.keras")
            labels_path = os.path.join(self.model_dir, f"{scan}_labels.json")
            
            try:
                self.models[scan] = load_trained_model(model_path)
                self.labels[scan] = load_class_labels(labels_path)
                print(f"✅ {scan.upper()} model loaded successfully")
            except FileNotFoundError:
                print(f"❌ {scan.upper()} model not found")
                continue

    def predict(self, image_path, scan_type, return_probabilities=False):
        """
        Classify medical image and return diagnosis
        
        Args:
            image_path: Path to medical image
            scan_type: Type of scan ('mri', 'ct', 'pet')
            return_probabilities: If True, return all class probabilities
            
        Returns:
            str: Predicted diagnosis
            or dict: {'diagnosis': str, 'confidence': float, 'probabilities': dict}
        """
        if scan_type not in self.models:
            raise ValueError(f"❌ Scan type not supported or model not loaded: {scan_type}")
        
        try:
            image = preprocess_image(image_path, self.img_size)
            predictions = self.models[scan_type].predict(image, verbose=0)
        except Exception as e:
            raise RuntimeError(f"⚠️ Prediction failed: {str(e)}")
        
        predicted_index = np.argmax(predictions)
        diagnosis = self.labels[scan_type].get(str(predicted_index), "Unknown")
        confidence = float(predictions[0][predicted_index])
        
        if return_probabilities:
            probabilities = {
                self.labels[scan_type].get(str(i), f"Class_{i}"): float(prob)
                for i, prob in enumerate(predictions[0])
            }
            return {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'probabilities': probabilities
            }
        
        return diagnosis

    def get_confidence(self, image_path, scan_type):
        """Return confidence score for the predicted diagnosis"""
        if scan_type not in self.models:
            raise ValueError(f"❌ Scan type not supported or model not loaded: {scan_type}")
            
        try:
            image = preprocess_image(image_path, self.img_size)
            predictions = self.models[scan_type].predict(image, verbose=0)
            return float(np.max(predictions))
        except Exception as e:
            raise RuntimeError(f"⚠️ Confidence calculation failed: {str(e)}")

    def get_diagnosis(self, image_path, scan_type):
        """Return only the predicted diagnosis"""
        return self.predict(image_path, scan_type)

    def get_all_probabilities(self, image_path, scan_type):
        """Return all class probabilities"""
        result = self.predict(image_path, scan_type, return_probabilities=True)
        return result['probabilities']

    @property
    def available_models(self):
        """List of loaded scan types"""
        return list(self.models.keys())

# Example usage
if __name__ == "__main__":
    predictor = PyHDCMPredictor()
    test_image = r"F:\Data\MRI\glioma\Tr-gl_0021.jpg"
    
    # Get only diagnosis
    diagnosis = predictor.get_diagnosis(test_image, "mri")
    print(f"Diagnosis: {diagnosis}")
    
    # Get only confidence
    confidence = predictor.get_confidence(test_image, "mri")
    print(f"Confidence: {confidence:.3f}")
    