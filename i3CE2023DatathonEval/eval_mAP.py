# please export results in cocoformat for your convenience
# don't use the data that was used to train the model when you evaluate

def normal_cocoeval(gt_path, dt_path):
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval

  # load ground truth data (annotations)
  gt_coco = COCO(gt_path)

  # load detection results
  dt_coco = gt_coco.loadRes(dt_path)

  # initialize COCO evaluation object
  cocoEval = COCOeval(gt_coco, dt_coco, iouType='bbox')

  # evaluate detections
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()

  # get mAP value
  mAP = cocoEval.stats[0]
  return mAP

def yolo_cocoeval():
  import torch
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval

  detections = torch.load('path_to_yolo_detections.pt')

  # Load COCO annotations
  coco_gt = COCO('path_to_coco_annotations.json')

  # Convert YOLO detections to COCO format
  coco_dt = []
  for image_id, detection in enumerate(detections):
      for x1y1, x2y2, conf, cls_conf, cls_pred in detection:
          category_id = int(cls_pred)
          bbox = [x1y1[0], x1y1[1], x2y2[0] - x1y1[0], x2y2[1] - x1y1[1]]
          coco_dt.append({
              'image_id': image_id,
              'category_id': category_id,
              'bbox': bbox,
              'score': conf * cls_conf
          })

  # Evaluate using COCO API
  coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

  # Get mAP
  mAP = coco_eval.stats[0]
  return mAP



def mask_rcnn_cocoeval():
  import torch
  import torchvision
  import coco_eval

  # Load test dataset and ground truth annotations
  test_dataset = load_dataset('path_to_test_dataset')
  gt_annotations = load_annotations('path_to_gt_annotations')

  # Load pre-trained Mask R-CNN model
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  # Evaluate model on test dataset
  detections = []
  for images, _ in test_dataset:
      # Run model on images
      predictions = model(images)

      # Extract predicted boxes, labels, and masks
      boxes = predictions[0]['boxes'].detach().cpu().numpy()
      labels = predictions[0]['labels'].detach().cpu().numpy()
      masks = predictions[0]['masks'].detach().cpu().numpy()

      # Add detections to list
      for i in range(len(labels)):
          detection = {
              'image_id': len(detections) + 1,
              'category_id': labels[i],
              'bbox': [float(x) for x in boxes[i]],
              'score': float(predictions[0]['scores'][i]),
              'segmentation': coco_eval.binary_mask_to_rle(masks[i])
          }
          detections.append(detection)

  # Evaluate using COCO API
  coco_evaluator = coco_eval.COCOEvaluator(gt_annotations, detections, iou_type='bbox')
  coco_evaluator.evaluate()
  coco_evaluator.accumulate()
  coco_evaluator.summarize()

  # Get mAP
  mAP = coco_evaluator.stats[0]
  return mAP
