import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

input_dir = r"C:\Users\palani\Desktop\IPCV\dataset_5k"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

data = []
labels = []

for label in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
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

                data.append(distances)
                labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))