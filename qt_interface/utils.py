import cv2
from PyQt5.QtGui import QImage, QPixmap

def cv_to_pixmap(cv_img):
    """将 OpenCV 图像转换为 QPixmap 用于 PyQt 显示"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img)
