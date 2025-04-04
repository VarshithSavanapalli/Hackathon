import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Indices for eyes landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(eye_landmarks):
    # Calculate eye aspect ratio (vertical/horizontal)
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def get_landmarks_coords(landmarks, indices, frame_w, frame_h):
    return np.array([[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in indices])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "No Face Detected"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Get eye landmarks
            left_eye = get_landmarks_coords(face_landmarks.landmark, LEFT_EYE, w, h)
            right_eye = get_landmarks_coords(face_landmarks.landmark, RIGHT_EYE, w, h)

            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            EAR_THRESHOLD = 0.2
            eyes_open = left_ear > EAR_THRESHOLD and right_ear > EAR_THRESHOLD

            # Head is likely straight if both eyes are visible and near same size
            symmetry = abs(left_ear - right_ear) < 0.08

            if eyes_open and symmetry:
                status = "Paying Attention"
            else:
                status = "Not Paying Attention"

    # Show result
    cv2.putText(frame, f'Status: {status}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0) if status == "Paying Attention" else (0, 0, 255), 3)
    cv2.imshow('Attention Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
