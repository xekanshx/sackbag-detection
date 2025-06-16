
import cv2
import os

video_path = '/Users/mac/transline/for test running/100KG Heavy Duty Truck Loading Belt Conveyor - YiFan Conveyor (720p, h264).mp4'  # Replace with your actual video path
output_dir = '/Users/mac/transline/video frames 28'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture('/Users/mac/transline/for test running/100KG Heavy Duty Truck Loading Belt Conveyor - YiFan Conveyor (720p, h264).mp4')
original_fps = cap.get(cv2.CAP_PROP_FPS)
desired_fps = 3
frame_interval = int(original_fps // desired_fps)

frame_count = 0
saved_count = 0
success, frame = cap.read()

while success:
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
    frame_count += 1
    success, frame = cap.read()

cap.release()
print(f"Saved {saved_count} frames at {desired_fps} FPS.")
