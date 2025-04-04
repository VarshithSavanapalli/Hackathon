import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Eye landmarks for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    return result.multi_face_landmarks[0] if result.multi_face_landmarks else None

def euclidean(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_points, width, height):
    p = [(int(landmarks.landmark[i].x * width), int(landmarks.landmark[i].y * height)) for i in eye_points]
    A = euclidean(p[1], p[5])
    B = euclidean(p[2], p[4])
    C = euclidean(p[0], p[3])
    return (A + B) / (2.0 * C)

def head_direction(landmarks):
    nose = landmarks.landmark[1].x  # center of face
    left = landmarks.landmark[234].x
    right = landmarks.landmark[454].x
    if nose < left: return "Looking Right"
    elif nose > right: return "Looking Left"
    else: return "Centered"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    landmarks = get_landmarks(frame)

    label = "No Face"
    
    if landmarks:
        # Calculate EAR
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Head pose
        direction = head_direction(landmarks)

        # Engagement logic
        if avg_ear < 0.2:
            label = "Sleepy "
        elif direction != "Centered":
            label = "Distracted "
        else:
            label = "Engaged "

    # Display label
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    cv2.imshow("Engagement Monitoring", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
