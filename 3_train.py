import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import matplotlib.pyplot as plt
import sys
import os
import io

# Ensure UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Define image size
img_size = 224  # Update to match the bounding box size in capture.py

# Prepare the dataset
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = data_gen.flow_from_directory('dataset',
                                         target_size=(img_size, img_size),
                                         batch_size=32,
                                         class_mode='categorical',
                                         subset='training')

val_gen = data_gen.flow_from_directory('dataset',
                                       target_size=(img_size, img_size),
                                       batch_size=32,
                                       class_mode='categorical',
                                       subset='validation')

# Save class indices
with open('class_indices.pkl', 'wb') as f:
    pickle.dump(train_gen.class_indices, f)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')  # Number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=20)

# Save the model
model.save('gesture_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
