import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("emotion_model_grayscale.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    mesh_frame = frame.copy()
    live_frame = frame.copy()

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw landmarks
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(mesh_frame, (x, y), 1, (0, 255, 0), -1)

            # Bounding box for emotion detection
            x_min = min([int(lm.x * w) for lm in face_landmarks.landmark])
            x_max = max([int(lm.x * w) for lm in face_landmarks.landmark])
            y_min = min([int(lm.y * h) for lm in face_landmarks.landmark])
            y_max = max([int(lm.y * h) for lm in face_landmarks.landmark])

            # Add padding
            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            face_roi = live_frame[y_min:y_max, x_min:x_max]

            # Emotion prediction
            if face_roi.size > 0:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=(0, -1))  # Shape: (1, 48, 48, 1)

                prediction = model.predict(face_input, verbose=0)
                emotion = emotion_labels[np.argmax(prediction)]

                # Draw result
                cv2.putText(live_frame, emotion, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(live_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Show both views
    cv2.imshow("Live Feed", live_frame)
    cv2.imshow("Mesh Overlay", mesh_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
