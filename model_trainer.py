import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("fer2013.csv")

# Parse data
pixels = data['pixels'].tolist()
width, height = 48, 48
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split()]
    face = np.asarray(face).reshape(width, height)
    face = face.astype('float32') / 255.0
    faces.append(face)

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

# One-hot encode emotions
emotions = pd.get_dummies(data['emotion']).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes in FER2013
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_test, y_test))

# Save model
model.save("emotion_model_grayscale.h5")
