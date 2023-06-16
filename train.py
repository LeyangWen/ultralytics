from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLO("yolov8x.pt")
model = YOLO("./runs/detect/train10/weights/last.pt")

# Use the model
model.train(data="DatathonHumanPretrain.yaml", epochs=100, lr0=0.0025, lrf=0.0025, warmup_epochs=0, batch=10, workers=0, device='cuda', optimizer="Adam", imgsz=640)     # train the model
metrics = model.val()
metrics_custom = model.val(data="DatathonHumanTrain.yaml", workers=0, device='cuda')  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format

# fine tuning
model.train(data="DatathonHumanTrain.yaml", epochs=200, batch=10, workers=0, device='cuda', optimizer="Adam", imgsz=640)     # train the model
metrics_custom = model.val()  # evaluate model performance on the validation set
