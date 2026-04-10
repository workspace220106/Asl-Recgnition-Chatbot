import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

FEATURES_PATH = r"C:\Users\araji\Downloads\miniproject\training\features.npy"
LABELS_PATH = r"C:\Users\araji\Downloads\miniproject\training\labels.npy"
MODEL_SAVE_PATH = r"C:\Users\araji\Downloads\miniproject\models\asl_model_full.keras"

print("Loading data...")
X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# Check data shape
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

label_map = np.load(r"C:\Users\araji\Downloads\miniproject\training\label_map.npy")
num_classes = len(label_map)
print(f"Number of classes: {num_classes}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model structure
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,  # Fast training epochs
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Training finished. Evaluating...")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
