# SILIT - Sign Language Transistor
**SILIT** is an innovative tool designed to convert sign language communication into written or spoken language. Utilizing advanced computer vision techniques, SILIT recognizes and tracks the movement of hands, fingers, and other relevant features in real-time video feeds.

## Tools Used
SILIT leverages the following libraries and tools:
1. **Python**
2. **Mediapipe**
3. **TensorFlow**
4. **Keras**
5. **OpenCV**
6. **Streamlit**

## Objectives
SILIT serves several important purposes:
1. **Communication Accessibility**: Enhances communication between sign language users and those who do not understand sign language.
2. **Education**: Aids in the teaching and learning of sign language.
3. **Interpreting Support**: Assists human interpreters by providing additional resources and verification.
4. **Accessibility in Digital Media**: Makes digital content more accessible to sign language users.

## Snapshots
![image](https://github.com/user-attachments/assets/b0029ea5-e421-4974-836b-1e522597b618)
![image](https://github.com/user-attachments/assets/8acaa6da-897f-48a7-af17-ff51fae6a530)
![image](https://github.com/user-attachments/assets/f1e6c818-017e-4809-a6e3-c3f3f82d1f4c)
![image](https://github.com/user-attachments/assets/b6f3f4b9-12b0-4b7b-aeb2-70a61ba2da05)

## How to Use
Follow these steps to use SILIT:

1. **Install Required Libraries**
    ```bash
    pip install opencv-python opencv-python-headless mediapipe tensorflow matplotlib numpy streamlit
    ```
2. **Create Dataset**
    - Run `1_capture.py`. Ensure you change the name of the gesture and specify the number of images you want to capture. The images will be saved in a folder named `/dataset`.
3. **Preprocess Data**
    - Run `2_preprocess.py` to preprocess the captured data.
4. **Train Your Model**
    - Run `3_train.py` to train the model on your dataset.
5. **Make Predictions**
    - After training is complete, test your program using your dataset by running `4_predict.py`.
6. **Deploy (Optional)**
    - Deploy the program to Streamlit by running `5_deploy.py`.

## Key Features
- **Easy to Use**: Our code is user-friendly and well-documented, making it accessible for developers of all skill levels.
- **Open Source**: SILIT is open-source, allowing you to view, modify, and enhance the code to fit your specific needs.
- **Customizable**: The system is designed to be easily customizable, giving you the flexibility to add new gestures or modify existing ones.

**Enjoy!**
