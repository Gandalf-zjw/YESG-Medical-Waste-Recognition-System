from ultralytics import YOLO
model = YOLO(r"D:\YESG\Medical Waste Recognition System\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml")  # 替换为你的YAML路径
print(model.model)  # 查看模型结构是否包含SwinTransformer
