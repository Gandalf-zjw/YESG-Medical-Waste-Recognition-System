import sys
import os

if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(os.path.join(base_path, 'ultralytics-main'))

import torch
from ultralytics import YOLO
import cv2


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def detect(self, image_path, return_boxes=False):
        results = self.model(image_path)
        result_image = results[0].plot()

        boxes = results[0].boxes
        details = []
        box_list = []

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                class_name = self.model.names[cls_id]
                conf = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0]
                xmin, ymin = int(xyxy[0]), int(xyxy[1])

                details.append(f"类别: {class_name}, 置信度: {conf:.2f}, 坐标: {xyxy}")
                box_list.append((class_name, conf, xmin, ymin))

        result_text = "\n".join(details) if details else "未检测到目标"

        if return_boxes:
            return result_image, result_text, box_list
        else:
            return result_image, result_text
