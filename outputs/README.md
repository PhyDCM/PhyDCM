# phydcm Models

Pre-trained Keras models for the phydcm library.

## Models included:
- model1.h5: Description of model 1
- model2.h5: Description of model 2

## Usage:
```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download and load model
model_path = hf_hub_download(
    repo_id="PhyDCM/phydcm-models",
    filename="model1.h5"
)
model = tf.keras.models.load_model(model_path)