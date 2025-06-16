from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO("/Users/mac/transline/10k_trained/best.pt")
model.fuse()
# Path to your input video
video_path = "/Users/mac/transline/for test running/Truck Loading Unloading Conveyor for Carton Boxes - Syoung Machinery (480p, h264).mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the current frame
    results = model.predict(frame, imgsz=640, conf=0.25)

    # Visualize predictions on frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Output", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
