import cv2
import os
import time
import numpy as np
import mediapipe as mp

# ================= USER SETTINGS =================
ASL_WORD = "K"          # CHANGE THIS FOR EACH RUN (ONE WORD ONLY)
TOTAL_IMAGES = 2500
CAPTURE_INTERVAL = 1    # seconds
IMAGE_SIZE = (200, 200)
CAMERA_INDEX = 0
# ================================================

# Create save directory
SAVE_DIR = os.path.join("ASL_Dataset", ASL_WORD)
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print(f"\n📸 Capturing ASL word: {ASL_WORD}")
print("✋ Hand-only capture enabled")
print("👉 Keep only ONE hand in frame")
print("👉 Press 'q' to stop early\n")

count = 0

try:
    while count < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:# Get bounding box from hand landmarks
            hand_landmarks = result.multi_hand_landmarks[0]
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size != 0:
                hand_crop = cv2.resize(hand_crop, IMAGE_SIZE)

                filename = f"{ASL_WORD}_{count:04d}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR, filename), hand_crop)

                count += 1
                print(f"✅ Saved {count}/{TOTAL_IMAGES}")

                cv2.imshow("Hand Crop", hand_crop)

                # Wait capture interval (allow exit)
                for _ in range(CAPTURE_INTERVAL * 10):
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
        else:
            cv2.imshow("Hand Crop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

except KeyboardInterrupt:
    print("\n🛑 Capture stopped manually")

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"\n📦 Finished '{ASL_WORD}' | Images saved: {count}")
