import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and Haar cascade
model = load_model("face_mask_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("😷 Real-Time Face Mask Detection App")

# Start camera
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not detected.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = x + w + margin
        y2 = y + h + margin

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (128, 128))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)[0][0]

        label = "With Mask 😷" if pred < 0.5 else "Without Mask ❌"
        color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({pred:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert BGR to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
