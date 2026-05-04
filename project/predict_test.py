import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.svm import SVC

# Load saved data
data = np.load("data.npy")
labels = np.load("labels.npy")

# Train model
model = SVC(kernel='linear', class_weight='balanced')
model.fit(data, labels)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Test dataset path (your case: images directly, not folders)
test_dir = r"C:\Users\palani\Desktop\IPCV\asl_alphabet_test\asl_alphabet_test"

correct = 0
total = 0

# Loop through all test images
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    # Extract label from filename (A_test.jpg → A)
    label = img_name.split("_")[0]

    img = cv2.imread(img_path)

    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            points = []

            for lm in hand_landmarks.landmark:
                points.append([lm.x, lm.y])

            points = np.array(points)

            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    distances.append(dist)

            pred = model.predict([distances])[0]

            print(f"Actual: {label} | Predicted: {pred}")

            if pred == label:
                correct += 1

            total += 1

# Final accuracy
if total > 0:
    print("\nFinal Test Accuracy:", correct / total)
else:
    print("\nNo hands detected in test images.")