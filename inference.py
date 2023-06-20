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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from i3CE2023DatathonEval.eval import *
from i3CE2023DatathonEval.eval_mAP import *

logging.basicConfig(level=logging.ERROR)

def yolo_predict_to_coco(results, image_path, image_id):
    """Converts YOLOv8 results to COCO format."""
    coco = []
    results = results[0]
    for i in range(len(results.boxes.cls)):
        this_coco = {}
        if False:
            this_coco["image_id"] = image_id
        else:
            try:
                this_coco["image_id"] = int(image_path.split('.')[0].split('\\')[-1].split('/')[-1])
            except:
                this_coco["image_id"] = str(image_path.split('.')[0].split('\\')[-1].split('/')[-1])
        this_coco["category_id"] = int(results.boxes.cls[i].cpu())
        bbox_xyxy = results.boxes.xyxy[i].cpu().numpy()
        bbox_coco_xywh = np.array([bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]])
        this_coco["bbox"] = bbox_coco_xywh.tolist()
        this_coco['area'] = str(bbox_coco_xywh[2] * bbox_coco_xywh[3])
        this_coco["score"] = results.boxes.conf[i].cpu().tolist()
        this_coco['file_name'] = image_path
        # this_coco['category_all'] = {0: "Cranes", 1: "Excavators", 2: "Bulldozers", 3: "Scrapers", 4: "Trucks", 5:"Workers"}
        coco.append(this_coco)
    return coco


def intersection_over_union(gt, pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])
    # if there is no overlap between predicted and ground-truth box
    if xB < xA or yB < yA:
        return 0.0
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def replace_human_bbox(general_result, human_result):
    iou_threshold = 0.5
    merged_result = []
    general_human = []
    for general in general_result:
        if general['category_id'] != 5:
            merged_result.append(general)
        else:
            general_human.append(general)
    for human in human_result:
        human['category_id'] = 5
        human_bbox = human['bbox']
        for gen_idx, gen_human in enumerate(general_human):
            gen_bbox = gen_human['bbox']
            # check bbox overlap ratio
            iou = intersection_over_union(gen_bbox, human_bbox)
            if iou > iou_threshold:
                general_human.pop(gen_idx)
            merged_result.append(human)
    for gen_human in general_human:
        merged_result.append(gen_human)
    return merged_result


def clean_human_bbox(result):
    iou_threshold = 0.5
    cleaned_result = []
    for res in result:
        if res['category_id'] != 5:
            cleaned_result.append(res)
        else:
            human_bbox = res['bbox']
            for idx, res2 in enumerate(result):
                if res2['category_id'] == 5:
                    continue
                gen_bbox = res2['bbox']
                # check bbox overlap ratio
                iou = intersection_over_union(gen_bbox, human_bbox)
                if iou > iou_threshold:
                    result.pop(idx)
            cleaned_result.append(res)
    return cleaned_result


CLASSES = ['Cranes', 'Excavators', 'Bulldozers', 'Scrapers', 'Trucks', 'Workers']
class_colors = [(255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255)]  # Yellow


if __name__ == '__main__':
    start = time.time()
    print('loading model...')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################# modify here #############################################
    test_img_folder = r'Y:\datasets\Datathoni3CE2023\test\images'
    output_folder = r'Y:\datasets\Datathoni3CE2023\DatathonTest\output'
    coco_gt_file = None
    coco_gt_file = r'Y:\datasets\Datathoni3CE2023\after_augment\test\coco_gt.json'
    ############################################# modify here #############################################

    model_folder = r'F:\F_coding_projects\ultralytics\runs\detect\i3CE2023-datathon-weights\2023-06-19-15-51'
    model = YOLO(os.path.join(model_folder, "general.pt"))
    model_human = YOLO(os.path.join(model_folder, "human.pt"))
    # iterate through this folder for image files
    count = 0
    for root, dirs, files in os.walk(test_img_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                count += 1
    print(f"Total number of images: {count}")

    image_id = -1
    coco_general = []
    coco_human = []
    coco_merged = []
    for root, dirs, files in os.walk(test_img_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                image_id += 1
                print(f"Processing {image_id} / {count}: {file}", end='\r')
                image_path = os.path.join(root, file)
                # image_path starting from base_folder
                relative_image_path = os.path.relpath(image_path, test_img_folder)
                image = cv2.imread(image_path)
                results = model.predict(source=image, verbose=False, imgsz=640)
                results_human = model_human.predict(source=image, verbose=False, imgsz=640)
                general_result = yolo_predict_to_coco(results, relative_image_path, image_id)
                human_result = yolo_predict_to_coco(results_human, relative_image_path, image_id)
                merged_result = replace_human_bbox(general_result, human_result)
                merged_result = clean_human_bbox(merged_result)
                coco_general.extend(general_result)
                coco_human.extend(human_result)
                coco_merged.extend(merged_result)
                save = False
                if image_id % 1 == 0: #and len(human_result) > 0:
                    save = True
                # save = True
                if save:
                    # display = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR))
                    # display.save(os.path.join(output_folder,'compare', f"{image_id}.jpg"))
                    # display_human = Image.fromarray(cv2.cvtColor(results_human[0].plot(), cv2.COLOR_RGB2BGR))
                    # display_human.save(os.path.join(output_folder,'compare', f"{image_id}_human.jpg"))

                    for image_info in general_result:
                        x, y, w, h = image_info['bbox']
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        class_i = int(image_info['category_id'])

                        cv2.rectangle(image, (x, y), (x + w, y + h), class_colors[class_i], 2)

                        font_scale = 0.5
                        text = CLASSES[class_i]
                        # Get the text size
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
                        # Set the position for the background rectangle
                        bg_width = text_width + 10
                        bg_height = text_height + 10
                        text_x = x
                        text_y = y - 5
                        # Draw the background rectangle
                        cv2.rectangle(image, (x, y - bg_height), (x + bg_width, y), class_colors[class_i], cv2.FILLED)
                        # Put the text on top of the background rectangle
                        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (0, 0, 0), 2)

                    # cv2.imshow('temp', resized_image)
                    # cv2.waitKey(0)

                    cv2.imwrite(os.path.join(output_folder, file), image)


    end = time.time()
    print(f"Time taken: {end-start} seconds for {count} images")

    if coco_gt_file is not None:
        # save coco_general
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, 'coco_general.json'), 'w') as f:
            json.dump(coco_general, f)
        with open(os.path.join(output_folder, 'coco_merged.json'), 'w') as f:
            json.dump(coco_merged, f)
        print(f"Saved coco_general.json and coco_merged.json to {output_folder}")


        # coco_read_file = r'Y:\datasets\COCO\annotations\instances_train2017.json'

        coco_dt_file = os.path.join(output_folder,'test_submit', 'coco_merged.json')
        mAP = normal_cocoeval(coco_gt_file, coco_dt_file)
        print(f"merged mAP: {mAP}")

        coco_dt_file = os.path.join(output_folder,'test_submit', 'coco_submit.json')
        mAP = normal_cocoeval(coco_gt_file, coco_dt_file)
        print(f"general mAP: {mAP}")