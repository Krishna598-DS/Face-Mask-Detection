import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('face_mask_model.h5')  # Make sure this is your MobileNetV2 model

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        try:
            # Add margin for better cropping
            margin = 20
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = x + w + margin
            y_end = y + h + margin

            face = frame[y_start:y_end, x_start:x_end]

            # Resize and preprocess the face
            face = cv2.resize(face, (128, 128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            # Predict
            pred = model.predict(face)[0][0]
            print(f"Prediction Score: {pred:.4f}")

            # Apply prediction logic
            if pred < 0.5:
                label = "With Mask"
                color = (0, 255, 0)
            else:
                label = "Without Mask"
                color = (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({pred:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Error during prediction:", e)

    # Show the frame
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
