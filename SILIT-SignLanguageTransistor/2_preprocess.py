import os
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = image.shape
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
            hand_frame = image[y_min:y_max, x_min:x_max]

            if hand_frame.size != 0:
                # Resize the image
                hand_frame = cv2.resize(hand_frame, target_size)
                hand_frame = (hand_frame * 255).astype(np.uint8)  # Convert to uint8

                return hand_frame  # Return normalized image

    return None

# Example usage: Iterate through the dataset directory and preprocess images
dataset_dir = '2/dataset'
processed_dir = '2/processed_dataset'

# Create processed directory if not exists
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Iterate through the dataset directory
for gesture_name in os.listdir(dataset_dir):
    gesture_dir = os.path.join(dataset_dir, gesture_name)
    if os.path.isdir(gesture_dir):
        target_dir = os.path.join(processed_dir, gesture_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Process images in the gesture directory
        for filename in os.listdir(gesture_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(gesture_dir, filename)
                processed_image = preprocess_image(image_path)
                if processed_image is not None:
                    processed_image_path = os.path.join(target_dir, filename)
                    cv2.imwrite(processed_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

print(f"Processed images saved in {processed_dir}")
