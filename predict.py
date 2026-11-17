from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("models/chest_xray_model.keras")
IMAGE_SIZE = (224, 224)

parser = argparse.ArgumentParser(description="Classify one chest X-ray image.")
parser.add_argument("image", help="Path to a JPG or PNG chest X-ray image")
args = parser.parse_args()

if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run 'python train.py' first.")

image = tf.keras.utils.load_img(
    args.image, target_size=IMAGE_SIZE, color_mode="rgb"
)
array = tf.keras.utils.img_to_array(image)
array = np.expand_dims(array, axis=0)

model = tf.keras.models.load_model(MODEL_PATH)
probability = float(model.predict(array, verbose=0)[0][0])

if probability >= 0.5:
    label = "PNEUMONIA"
    score = probability
else:
    label = "NORMAL"
    score = 1.0 - probability

print(f"Predicted class: {label}")
print(f"Model score for predicted class: {score:.1%}")
print("Educational result only; not a medical diagnosis.")
