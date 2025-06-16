from ultralytics import YOLO
import torch
import os
import matplotlib.pyplot as plt
import cv2

# Check device
DEVICE = "cpu"  # Use "cuda" if you have a GPU
print(f"Using device: {DEVICE}")

# Paths
DATA_YAML_PATH = "/Users/mac/transline/on site improvised/sackbags detection merged 4.v1i.yolov8/data.yaml"
MODEL_PATH = "/Users/mac/transline/on site improvised/best.pt"

# Load model
model = YOLO(MODEL_PATH)

# Run validation
metrics = model.val(
    data=DATA_YAML_PATH,
    imgsz=640,
    device=DEVICE,
    save=True  # <- This enables saving confusion matrix and other plots
)

# Print evaluation metrics
print("\n📊 Validation Metrics:")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP75: {metrics.box.map75:.4f}")
print(f"  Per-class mAP50-95: {metrics.box.maps}")

# Display saved confusion matrix and other plots
results_dir = model.training.args.save_dir  # Directory where results are saved
plots = [
    "confusion_matrix.png",
    "F1_curve.png",
    "PR_curve.png",
    "results.png"
]

print("\n🖼️ Showing saved plots:")
for plot_name in plots:
    plot_path = os.path.join(results_dir, plot_name)
    if os.path.exists(plot_path):
        print(f"Displaying: {plot_name}")
        img = cv2.imread(plot_path)
        cv2.imshow(plot_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ Plot not found: {plot_name}")
