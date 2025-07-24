from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download and load model
model_path = hf_hub_download(
    repo_id="PhyDCM/phydcm-models",  # ✨ ضع هنا اسم الريبو في هَجِنْغ فيس، مثل: username/repo-name
    filename="model1.h5"
)

# Load with TensorFlow Keras
model = tf.keras.models.load_model(model_path)
