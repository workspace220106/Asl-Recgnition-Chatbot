import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import glob
import urllib.request

# Download the hand landmark model if it doesn't exist
model_path = r"C:\Users\araji\Downloads\miniproject\training\hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Download complete.")

# Initialize the HandLandmarker task
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = r"C:\Users\araji\Downloads\miniproject\dataset"
LABELS = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and len(d) == 1])
print("Detected LABELS:", LABELS)
SAVE_PATH_FEATURES = r"C:\Users\araji\Downloads\miniproject\training\features.npy"
SAVE_PATH_LABELS = r"C:\Users\araji\Downloads\miniproject\training\labels.npy"

all_features = []
all_labels = []

print("Starting feature extraction...")

for label_idx, label in enumerate(LABELS):
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        continue
        
    image_files = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True) + \
                  glob.glob(os.path.join(folder_path, "**", "*.jpeg"), recursive=True)
                  
    print(f"Processing {label}: Found {len(image_files)} images.")
    count = 0
    # Process all available images
    for img_path in image_files:
        # Load image with MediaPipe Image format
        try:
            mp_image = mp.Image.create_from_file(img_path)
            
            # Detect hand landmarks
            detection_result = detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    landmarks = []
                    # Normalize landmarks relative to wrist (landmark 0)
                    wrist_x = hand_landmarks[0].x
                    wrist_y = hand_landmarks[0].y
                    wrist_z = hand_landmarks[0].z
                    
                    for lm in hand_landmarks:
                        landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                    
                    # Scale normalization
                    landmarks = np.array(landmarks)
                    max_val = np.abs(landmarks).max()
                    if max_val > 0:
                        landmarks = landmarks / max_val
                    
                    all_features.append(landmarks.tolist())
                    all_labels.append(label_idx)
                    count += 1
        except Exception as e:
            # Ignore bad images
            pass
            
        # Optional: Print progress for every 500 images to know it's not stuck
        if count % 500 == 0 and count > 0:
            print(f"  Extracted {count} features from {label}")
            
    print(f"  Extracted {count} features from {label}")

print(f"Feature extraction complete. Total samples: {len(all_features)}")

# Save to disk
os.makedirs(os.path.dirname(SAVE_PATH_FEATURES), exist_ok=True)
np.save(SAVE_PATH_FEATURES, np.array(all_features))
np.save(SAVE_PATH_LABELS, np.array(all_labels))
# Save the label mapping
np.save(r"C:\Users\araji\Downloads\miniproject\training\label_map.npy", np.array(LABELS))

print("Saved features and labels to disk.")
