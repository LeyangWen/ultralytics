from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("yolov8x.pt")
# model = YOLO("./runs/detect/train16/weights/best.pt")


# # Use the model
# results = model.train(data=".././datasets/Datathon2023/data.yaml", epochs=500, imgsz=224,
#                       optimizer="Adam", seed=0, lr0=1e-3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# Use the model
model.train(data="DatathonHumanPretrain.yaml", epochs=100, batch=10, workers=0, device='cuda', optimizer="Adam", imgsz=640)     # train the model
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="onnx")  # export the model to ONNX format
