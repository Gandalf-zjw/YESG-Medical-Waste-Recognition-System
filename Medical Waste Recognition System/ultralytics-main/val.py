import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp16/weights/best.pt')
    model.val(data=r"D:\yolov11\ultralytics-main\data\data.yaml",
              imgsz=640,
              batch=16,
              split='test',
              workers=0,
              device='0',
              )

