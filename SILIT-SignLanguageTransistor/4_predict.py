import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle

# Load the trained model
model = load_model('gesture_model.h5')

# Load class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
labels = list(class_indices.keys())

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around the hand
            h, w, c = frame.shape
            landmarks = hand_landmarks.landmark
            x_min = w
            y_min = h
            x_max = y_max = 0

            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Expand the bounding box slightly
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop the hand region
            hand_frame = frame[y_min:y_max, x_min:x_max]

            if hand_frame.size != 0:
                # Preprocess the hand frame
                img = cv2.resize(hand_frame, (224, 224))
                img = np.expand_dims(img, axis=0) / 255.0

                # Make prediction
                predictions = model.predict(img)
                predicted_label = labels[np.argmax(predictions)]
                
                # Print prediction probabilities for debugging
                print(f"Predictions: {predictions} - Predicted Label: {predicted_label}")

                # Display the result
                cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
