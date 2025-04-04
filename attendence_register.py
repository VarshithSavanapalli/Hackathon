import cv2
import os
import time

def register_student(name):
    save_dir = f"known_faces/{name}"
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open video device")
        return

    print(f"[INFO] Capturing images for {name}. Will automatically capture 5 images...")
    count = 0
    max_images = 5
    capture_interval = 1  # seconds

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        img_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Saved {img_path}")
        count += 1
        time.sleep(capture_interval)

    cap.release()


def get_student_name():
    while True:
        name = input("Enter student name: ").strip()
        if name:
            return name
        print("Error: Name cannot be empty")

if __name__ == "__main__":
    student_name = get_student_name()
    register_student(student_name)
