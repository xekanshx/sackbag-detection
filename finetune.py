
from ultralytics import YOLO
import torch
import os

DEVICE = "cpu"
print(f"Using device: {DEVICE}")

DATA_YAML_PATH = "/Users/mac/transline/fine tuning/sack.v4i.yolov8/data.yaml"

MODEL_PATH = "/Users/mac/transline/15k trained/12.9klocal.pt"

model = YOLO(MODEL_PATH)

model.train(
    data=DATA_YAML_PATH,
    epochs=4,
    batch=-1,
    imgsz=640,
    patience=5,
    device=DEVICE,
    lr0=0.0001,
    resume=False
)
