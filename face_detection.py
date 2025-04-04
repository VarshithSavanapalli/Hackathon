import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# Initialize MTCNN face detector
detector = MTCNN()

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MTCNN processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(frame_rgb)

    # Draw bounding boxes
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        
        if confidence > 0.9:  # Only consider high-confidence detections
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {round(confidence * 100, 1)}%', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection in Online Classes', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()