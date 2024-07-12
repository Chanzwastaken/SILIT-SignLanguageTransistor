# SILIT
## ABOUT SILIT (SIGN LANGUAGE TRANSCRIPTOR)
SILIT is a tool or system designed to convert sign language communication into written or spoken language. The software employs computer vision techniques to recognize and track the movement of the hands, fingers, and other relevant features in the video.


## Tools used
Here are the library used tools in the field :
1. Python
2. Mediapipe
3. TensorFlow
4. Keras
5. Open CV
6. Streamlit

## Objective
SILIT serves several important purposes:
1. Communication Accessibility
2. Education
3. Interpreting Support
4. Accessibility in Digital Media

## Snapshot
![image](https://github.com/user-attachments/assets/b0029ea5-e421-4974-836b-1e522597b618)
![image](https://github.com/user-attachments/assets/8acaa6da-897f-48a7-af17-ff51fae6a530)
![image](https://github.com/user-attachments/assets/f1e6c818-017e-4809-a6e3-c3f3f82d1f4c)
![image](https://github.com/user-attachments/assets/b6f3f4b9-12b0-4b7b-aeb2-70a61ba2da05)

## How to use
1. Install required libraries
```
pip install opencv-python opencv-python-headless mediapipe tensorflow matplotlib numpy streamlit
```
2. Create dataset by run ```1_capture.py```. Don't forget to change name of the gesture and how many image that you want to capture. And it'll be saved in a folder named ```/dataset```
3. Run ```2_preprocess.py``` to preprocess the data
4. Train your data by running ```3_train.py```
5. After training completed, you can try your program using your dataset by running ```4_predict.py```
6. (optional) You can deploy the program to Streamlit by running ```5_deploy.py```

*ENJOYY*
