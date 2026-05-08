import cv2
import numpy as np


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces


def align_face(image, face_rect, output_size=112):
    x, y, w, h = face_rect
    center = (x + w // 2, y + h // 2)
    face_roi = image[max(0, y):min(image.shape[0], y + h),
                     max(0, x):min(image.shape[1], x + w)]
    aligned = cv2.resize(face_roi, (output_size, output_size))
    return aligned


def process_video_frames(frames, align=True):
    processed_frames = []
    for frame in frames:
        if align:
            faces = detect_faces(frame)
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                aligned = align_face(frame, largest_face)
                processed_frames.append(aligned)
        else:
            processed_frames.append(frame)
    return processed_frames
