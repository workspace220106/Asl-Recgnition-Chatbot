import tensorflow as tf
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_model_full.keras")
LABELS_PATH = os.path.join(BASE_DIR, "training", "label_map.npy")

print("Loading model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

def predict(img):
    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)
    return labels[idx], float(preds[0][idx])
