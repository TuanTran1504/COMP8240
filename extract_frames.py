import cv2
import os

# Input video
video_path = "./data/raw_video/3.mp4"
output_dir = "frames/3"
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_rate = 5 # extract 1 frame every 5 frames

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % frame_rate == 0:
        filename = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1
    
    count += 1

cap.release()
print(f"Extracted {saved} frames to {output_dir}")
