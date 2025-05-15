import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\best-results\yolov11-attention-ECA\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')
    model.train(data=r"D:\best-results\yolov11-attention-ECA\ultralytics-main\data\data.yaml",
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                pretrained=False,
                batch=16,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='AdamW',
                #resume=True,
                amp=True,
                project='runs/train',
                name='exp',
                )

