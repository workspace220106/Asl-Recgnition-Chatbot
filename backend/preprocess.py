import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Initialize the HandLandmarker task
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "training", "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def preprocess_image(img_bytes):
    # Decode bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Detect hand landmarks
    detection_result = detector.detect(mp_image)
    
    # Generate the 63 features if a hand is found
    if detection_result.hand_landmarks:
        print("DEBUG: Hand detected!")
        hand_landmarks = detection_result.hand_landmarks[0]
        landmarks = []
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
            
        return np.array([landmarks])  # Shape (1, 63)
    
    # Return None if no hand detected
    print("DEBUG: No hand detected in image.")
    return None
