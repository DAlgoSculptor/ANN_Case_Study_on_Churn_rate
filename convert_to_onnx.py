"""Convert Keras model to ONNX format for Python 3.14 compatibility"""
import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = Path("artifacts/churn_ann_model.keras")
ONNX_PATH = Path("artifacts/churn_ann_model.onnx")

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH)

# Call the model once with dummy data
dummy_input = np.zeros((1, 13), dtype=np.float32)  # 13 features after preprocessing
_ = model(dummy_input)

# Export to ONNX
model.export(str(ONNX_PATH), format="onnx")
print(f"✓ Model converted and saved to {ONNX_PATH}")
