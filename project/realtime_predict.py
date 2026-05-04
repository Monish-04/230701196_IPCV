import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC

# Load trained data
data = np.load("data.npy")
labels = np.load("labels.npy")

# Train model
model = SVC(kernel='linear', class_weight='balanced')
model.fit(data, labels)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # ✅ DRAW LANDMARKS (this is what you wanted)
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            points = []

            # Collect landmark points
            for lm in hand_landmarks.landmark:
                points.append([lm.x, lm.y])

            points = np.array(points)
            points = points - points[0]  # normalize

            # Calculate distances
            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    distances.append(dist)

            # Predict
            pred = model.predict([distances])[0]

            # Display prediction
            cv2.putText(frame, f"Predicted: {pred}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()