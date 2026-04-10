import os
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import concurrent.futures

DATASET_PATH = r"C:\Users\araji\Downloads\miniproject\dataset"
LABELS = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d)) and len(d) == 1])
SAVE_PATH_FEATURES = r"C:\Users\araji\Downloads\miniproject\training\features.npy"
SAVE_PATH_LABELS = r"C:\Users\araji\Downloads\miniproject\training\labels.npy"
MODEL_PATH = r"C:\Users\araji\Downloads\miniproject\training\hand_landmarker.task"

# Set TensorFlow backend logging to FATAL to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
mp.Runtime = None # suppress some debugs... actually let's just let it be

detector = None

def init_worker():
    global detector
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

def process_image(args):
    img_path, label_idx = args
    try:
        mp_image = mp.Image.create_from_file(img_path)
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks = []
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            wrist_z = hand_landmarks[0].z
            
            for lm in hand_landmarks:
                landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            
            landmarks = np.array(landmarks)
            max_val = np.abs(landmarks).max()
            if max_val > 0:
                landmarks = landmarks / max_val
            return (landmarks.tolist(), label_idx)
    except Exception:
        pass
    return None

if __name__ == '__main__':
    all_features = []
    all_labels = []
    
    tasks = []
    print("Detected LABELS:", LABELS)
    for label_idx, label in enumerate(LABELS):
        folder_path = os.path.join(DATASET_PATH, label)
        images = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True) + \
                 glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True) + \
                 glob.glob(os.path.join(folder_path, "**", "*.jpeg"), recursive=True)
        for img in images:
            tasks.append((img, label_idx))
            
    print(f"Found {len(tasks)} images in total. Processing with multiprocessing...")
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        results = executor.map(process_image, tasks, chunksize=100)
        
        for i, res in enumerate(results):
            if res is not None:
                all_features.append(res[0])
                all_labels.append(res[1])
                success_count += 1
            if (i+1) % 2000 == 0:
                print(f"Processed {i+1}/{len(tasks)} images... (Valid hands: {success_count})")
                
    print(f"Completed! Total valid samples: {success_count}")
    
    os.makedirs(os.path.dirname(SAVE_PATH_FEATURES), exist_ok=True)
    np.save(SAVE_PATH_FEATURES, np.array(all_features))
    np.save(SAVE_PATH_LABELS, np.array(all_labels))
    np.save(r"C:\Users\araji\Downloads\miniproject\training\label_map.npy", np.array(LABELS))
    print("Features and labels saved successfully.")
