# YOLOv8 Training Guide (Workplace Safety)

## 1. Dataset Structure (YOLO format)

Create this structure:

- datasets/workplace_ppe/images/train
- datasets/workplace_ppe/images/val
- datasets/workplace_ppe/labels/train
- datasets/workplace_ppe/labels/val
- datasets/workplace_ppe/data.yaml

Each label file contains rows with:

- class_id center_x center_y width height

All coordinates are normalized to [0, 1].

## 2. data.yaml example

```yaml
path: datasets/workplace_ppe
train: images/train
val: images/val
names:
  0: person
  1: helmet
  2: vest
  3: machinery
```

## 3. Train with augmentation

```bash
yolo task=detect mode=train model=yolov8n.pt data=datasets/workplace_ppe/data.yaml epochs=80 imgsz=640 batch=16 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 degrees=8 translate=0.1 scale=0.5
```

## 4. Evaluate

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=datasets/workplace_ppe/data.yaml
```

Track these metrics:

- mAP50
- mAP50-95
- precision
- recall

## 5. Inference in this app

Set model path in config.py:

- DEFAULT_MODEL_PATH = "runs/detect/train/weights/best.pt"

Then run Streamlit app.
