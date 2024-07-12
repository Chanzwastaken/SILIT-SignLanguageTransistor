import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load the trained model
model = load_model('gesture_model.h5')  # Load the model in H5 format

# Load class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
labels = list(class_indices.keys())

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Define a class for video transformation
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the frame to find hands
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box around the hand
                h, w, c = img.shape
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
                hand_frame = img[y_min:y_max, x_min:x_max]

                if hand_frame.size != 0:
                    # Preprocess the hand frame
                    hand_img = cv2.resize(hand_frame, (224, 224))
                    hand_img = np.expand_dims(hand_img, axis=0) / 255.0

                    # Make prediction
                    predictions = model.predict(hand_img)
                    predicted_label = labels[np.argmax(predictions)]

                    # Display the result
                    cv2.putText(img, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# Streamlit app
def main():
    st.title("Hand Gesture Recognition")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer:
        st.write("**Instructions:** Make gestures in front of your webcam.")

if __name__ == "__main__":
    main()
