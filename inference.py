from ultralytics import YOLO
import cv2
import torch
import os
import time
import datetime
import glob
import json
import logging
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.ERROR)

def yolo_predict_to_coco(results, image_path, image_id):
    """Converts YOLOv8 results to COCO format."""
    coco = []
    results = results[0]
    for i in range(len(results.boxes.cls)):
        this_coco = {}
        this_coco["image_id"] = image_id
        this_coco["category_id"] = int(results.boxes.cls[i].cpu())
        this_coco["bbox"] = results.boxes.xywh[i].cpu().numpy().tolist()
        this_coco["score"] = results.boxes.conf[i].cpu().numpy().tolist()
        this_coco['image_path'] = image_path
        # this_coco['category_all'] = {0: "Cranes", 1: "Excavators", 2: "Bulldozers", 3: "Scrapers", 4: "Trucks", 5:"Workers"}
        coco.append(this_coco)
    return coco


def replace_human_bbox(general_result, human_result):
    merged_result = []
    for general in general_result:
        if general['category_id'] != 5:
            merged_result.append(general)
    for human in human_result:
        merged_result.append(human)
    return merged_result


if __name__ == '__main__':
    start = time.time()
    print('loading model...')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_folder = r'Y:\datasets\Datathoni3CE2023\train\images'
    output_folder = r'Y:\datasets\Datathoni3CE2023\temp_output'
    model_folder = r'runs\detect\best_storage'
    model = YOLO(os.path.join(model_folder, "YOLOv8x_best_gunwoo_20230614.pt"))
    model_human = YOLO(os.path.join(model_folder, "YOLOv8x_best_human_85.pt"))
    # iterate through this folder for image files
    count = 0
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                count += 1
    print(f"Total number of images: {count}")

    image_id = 0
    coco_general = []
    coco_human = []
    coco_merged = []
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                image_id += 1
                print(f"Processing {image_id} / {count}: {file}", end='\r')
                image_path = os.path.join(root, file)
                # image_path starting from base_folder
                relative_image_path = os.path.relpath(image_path, test_folder)
                image = cv2.imread(image_path)
                results = model.predict(source=image, verbose=False, imgsz=224)
                results_human = model_human.predict(source=image, verbose=False, imgsz=640)
                general_result = yolo_predict_to_coco(results, relative_image_path, image_id)
                human_result = yolo_predict_to_coco(results_human, relative_image_path, image_id)
                merged_result = replace_human_bbox(general_result, human_result)
                coco_general.extend(general_result)
                coco_human.extend(human_result)
                coco_merged.extend(merged_result)
                if image_id % 25 == 1 and len(human_result) > 0:
                    display = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR))
                    display.save(os.path.join(output_folder, f"{image_id}.jpg"))
                    display_human = Image.fromarray(cv2.cvtColor(results_human[0].plot(), cv2.COLOR_RGB2BGR))
                    display_human.save(os.path.join(output_folder, f"{image_id}_human.jpg"))


    end = time.time()
    print(f"Time taken: {end-start} seconds for {count} images")

    # save coco_general
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'coco_general.json'), 'w') as f:
        json.dump(coco_general, f)
    with open(os.path.join(output_folder, 'coco_merged.json'), 'w') as f:
        json.dump(coco_merged, f)
    print(f"Saved coco_general.json and coco_merged.json to {output_folder}")