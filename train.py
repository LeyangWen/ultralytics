from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLO("yolov8x.pt")
# model = YOLO("./runs/detect/train10/weights/last.pt")
#
# # Use the model
# model.train(data="DatathonHumanPretrain.yaml", epochs=100, lr0=0.0025, lrf=0.0025, warmup_epochs=0, batch=10, workers=0, device='cuda', optimizer="Adam", imgsz=640)     # train the model
# metrics = model.val()
# metrics_custom = model.val(data="DatathonHumanTrain.yaml", workers=0, device='cuda')  # evaluate model performance on the validation set
# path = model.export(format="onnx")  # export the model to ONNX format

model = YOLO("./runs/detect/train18/weights/best.pt")
# model = YOLO(r"F:\F_coding_projects\ultralytics\runs\detect\i3CE2023-datathon-weights\2023-06-19-15-51\general.pt")
# fine tuning
model.train(data="DatathonHumanTrain.yaml", epochs=500, batch=10, workers=0, device='cuda', optimizer="Adam", imgsz=640, lr0=0.0025, lrf=0.0025, warmup_epochs=0)     # train the model
metrics_custom = model.val(data="DatathonHumanTrain.yaml", workers=0, device='cuda')
# metrics_custom = model.val(data="DatathonTrain.yaml", workers=0, device='cuda')

