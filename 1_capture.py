import cv2
import os
import mediapipe as mp

# Directory to save images
gesture_name = "love"  # Change this to the name of the gesture
save_dir = f"dataset/{gesture_name}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0
max_images = 500  # Maximum number of images to capture

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

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
                # Save the hand gesture image
                img_name = os.path.join(save_dir, f"{count}.png")
                cv2.imwrite(img_name, hand_frame)
                print(f"Captured {img_name}")
                count += 1

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key or when max_images is reached
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
