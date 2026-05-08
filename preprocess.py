import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.face_alignment import process_video_frames


def load_video_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def save_preprocessed_frames(frames, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_path, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def preprocess_videos(config):
    video_dir = config.video_dir
    output_dir = config.preprocessed_dir

    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return

    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, f))

    for video_path in tqdm(video_files, desc="Preprocessing videos"):
        rel_path = os.path.relpath(video_path, video_dir)
        subject = rel_path.split(os.sep)[0]
        sweep = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, subject, sweep)

        if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
            continue

        frames = load_video_frames(video_path)
        if frames:
            processed = process_video_frames(frames, align=True)
            save_preprocessed_frames(processed, output_path)

    print(f"Preprocessing complete. Processed {len(video_files)} videos.")
