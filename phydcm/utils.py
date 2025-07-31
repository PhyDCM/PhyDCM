import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import json
from PIL import Image
import pydicom

def is_dicom_file(filepath):
    """Check if file is DICOM format regardless of extension"""
    try:
        if filepath.lower().endswith(('.dcm', '.dicom')):
            return True
        with open(filepath, 'rb') as f:
            header = f.read(132)
            if len(header) >= 132 and header[128:132] == b'DICM':
                return True
            # Check for DICOM data elements pattern
            f.seek(0)
            first_bytes = f.read(8)
            return any(first_bytes.startswith(pattern) for pattern in [b'\x08\x00', b'\x10\x00', b'\x20\x00', b'\x28\x00'])
    except Exception:
        return False

def load_dicom_image(filepath):
    """Load and process DICOM image with proper windowing"""
    dicom_data = pydicom.dcmread(filepath, force=True)
    if not hasattr(dicom_data, 'pixel_array'):
        raise ValueError("DICOM file contains no pixel data")
    
    image = dicom_data.pixel_array
    
    # Apply windowing if available
    if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
        center = float(dicom_data.WindowCenter)
        width = float(dicom_data.WindowWidth)
        img_min, img_max = center - width // 2, center + width // 2
        image = np.clip(image, img_min, img_max)
    
    # Normalize to 0-255 range
    if image.max() > 255:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    return image

def detect_image_format(filepath):
    """Detect image format based on file signature"""
    if is_dicom_file(filepath):
        return 'dicom'
    
    try:
        with open(filepath, 'rb') as f:
            header = f.read(16)
            signatures = {
                b'\xff\xd8\xff': 'standard',  # JPEG
                b'\x89PNG\r\n\x1a\n': 'standard',  # PNG
                b'BM': 'standard',  # BMP
                b'GIF87a': 'standard', b'GIF89a': 'standard',  # GIF
            }
            for sig, fmt in signatures.items():
                if header.startswith(sig):
                    return fmt
            if header.startswith(b'RIFF') and b'WEBP' in header:
                return 'standard'
    except Exception:
        pass
    return 'unknown'

def preprocess_image(image_path, img_size=(224, 224)):
    """Load and preprocess medical images for model input"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image_format = detect_image_format(image_path)
    
    try:
        if image_format == 'dicom':
            image = load_dicom_image(image_path)
        else:
            # Try OpenCV first, fallback to PIL
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = np.array(Image.open(image_path).convert("RGB"))
        
        if image is None:
            raise ValueError("Failed to load image")
        
        # Resize and normalize
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=0)
        
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def load_class_labels(json_path):
    """Load class labels from JSON configuration file"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Labels file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {str(k): v for k, v in data.items()}

def load_trained_model(model_path):
    """Load pre-trained neural network model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)