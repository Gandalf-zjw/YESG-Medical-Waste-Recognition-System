import sys
import os

# ========== 加入本地 ultralytics-main 路径 ==========
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(os.path.join(base_path, 'ultralytics-main'))
# =====================================================

import time
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QLineEdit,
    QComboBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QBrush
from PyQt5.QtCore import Qt
from detector import Detector
from utils import cv_to_pixmap

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医疗废弃物智能识别系统")
        self.setGeometry(100, 50, 1300, 850)

        # 多语言支持变量
        self.translations = {
            'zh': {
                'title': "医疗废弃物智能识别系统",
                'image_path': "图像路径：",
                'upload': "上传图像",
                'model_switch': "模型切换：",
                'detection_result': "检测结果：",
                'time': "用时：",
                'count': "目标数：",
                'type': "类型：",
                'confidence': "置信度：",
                'save': "保存检测图",
                'exit': "退出系统",
                'table_title': "检测结果与位置信息：",
                'table_headers': ["序号", "类别", "置信度", "xmin", "ymin"]
            },
            'en': {
                'title': "Medical Waste Recognition System",
                'image_path': "Image Path:",
                'upload': "Upload Image",
                'model_switch': "Model Switch:",
                'detection_result': "Detection Result:",
                'time': "Time:",
                'count': "Number:",
                'type': "Category:",
                'confidence': "Confidence:",
                'save': "Save Image",
                'exit': "Exit",
                'table_title': "Detection Results and Coordinates:",
                'table_headers': ["Index", "Class", "Confidence", "xmin", "ymin"]
            }
        }
        self.current_lang = 'zh'

        # 设置背景图
        palette = QPalette()
        background = QPixmap("background.png")
        palette.setBrush(QPalette.Window, QBrush(background))
        self.setPalette(palette)

        # 路径兼容（打包 vs 开发）
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        self.model_paths = [
            os.path.join(base_path, "runs", "YESG-m-MCID.pt"),
            os.path.join(base_path, "runs", "YESG-m-SCID.pt"),
            os.path.join(base_path, "runs", "YESG-n-MCID.pt"),
            os.path.join(base_path, "runs", "YESG-n-SCID.pt"),
        ]

        self.model_names = [
            "YESG-m-MCID",
            "YESG-m-SCID",
            "YESG-n-MCID",
            "YESG-n-SCID"
        ]

        self.detector = Detector(self.model_paths[0])

        self.image_label = QLabel("PLease upload the image")
        self.image_label.setFixedSize(820, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { border: 2px solid #90A4AE; background-color: #FFFFFF; font-size: 16px; }")

        self.language_selector = QComboBox()
        self.language_selector.addItems(["English", "中文"])
        self.language_selector.currentIndexChanged.connect(self.switch_language)

        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_input.setStyleSheet("QLineEdit { background: #ffffff; padding: 4px; border-radius: 4px; }")

        self.upload_button = QPushButton()
        self.upload_button.clicked.connect(self.load_image)

        self.model_selector = QComboBox()
        self.model_selector.addItems(self.model_names)
        self.model_selector.currentIndexChanged.connect(self.change_model)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(70)
        self.result_text.setStyleSheet("QTextEdit { background: #ffffff; border-radius: 6px; }")

        self.table = QTableWidget(0, 5)
        self.table.setFixedHeight(180)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.time_label = QLabel()
        self.count_label = QLabel()
        self.class_label = QLabel()
        self.conf_label = QLabel()

        for label in [self.time_label, self.count_label, self.class_label, self.conf_label]:
            label.setStyleSheet("""
                font-weight: bold;
                font-size: 16px;
                color: #003366;
                background-color: #e3f2fd;
                padding: 6px 12px;
                border-radius: 6px;
            """)

        self.save_button = QPushButton()
        self.save_button.clicked.connect(self.save_result)
        self.quit_button = QPushButton()
        self.quit_button.clicked.connect(self.close)

        button_style = """
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-size: 15px;
                padding: 12px 24px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        self.save_button.setStyleSheet(button_style)
        self.quit_button.setStyleSheet(button_style)
        self.save_button.setFixedWidth(200)
        self.quit_button.setFixedWidth(200)

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.time_label)
        info_layout.addWidget(self.count_label)
        info_layout.addWidget(self.class_label)
        info_layout.addWidget(self.conf_label)

        button_box = QHBoxLayout()
        button_box.addStretch()
        button_box.addWidget(self.save_button)
        button_box.addWidget(self.quit_button)
        button_box.addStretch()

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.addWidget(self.language_selector)
        self.label_image_path = QLabel()
        self.label_model_switch = QLabel()
        self.label_detection_result = QLabel()
        right_layout.addWidget(self.label_image_path)
        right_layout.addWidget(self.file_input)
        right_layout.addWidget(self.upload_button)
        right_layout.addWidget(self.label_model_switch)
        right_layout.addWidget(self.model_selector)
        right_layout.addWidget(self.label_detection_result)
        right_layout.addLayout(info_layout)
        right_layout.addWidget(self.result_text)
        right_layout.addStretch()
        right_layout.addLayout(button_box)

        main_layout = QVBoxLayout()
        self.title_label = QLabel()
        self.title_label.setFont(QFont("Microsoft YaHei", 20))
        self.title_label.setStyleSheet("color: #003366;")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.image_label)
        top_layout.addLayout(right_layout, stretch=1)

        main_layout.addLayout(top_layout)
        self.table_title = QLabel()
        self.table_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.table_title)
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)
        self.switch_language(0)

    def switch_language(self, index):
        lang = 'en' if index == 0 else 'zh'
        self.current_lang = lang
        t = self.translations[lang]
        self.setWindowTitle(t['title'])
        self.title_label.setText(t['title'])
        self.label_image_path.setText(t['image_path'])
        self.upload_button.setText(t['upload'])
        self.label_model_switch.setText(t['model_switch'])
        self.label_detection_result.setText(t['detection_result'])
        self.time_label.setText(t['time'])
        self.count_label.setText(t['count'])
        self.class_label.setText(t['type'])
        self.conf_label.setText(t['confidence'])
        self.save_button.setText(t['save'])
        self.quit_button.setText(t['exit'])
        self.table_title.setText(t['table_title'])
        self.table.setHorizontalHeaderLabels(t['table_headers'])

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.file_input.setText(file_name)
            self.detect_image(file_name)

    def detect_image(self, path):
        try:
            start = time.time()
            result_img, result_text, boxes = self.detector.detect(path, return_boxes=True)
            end = time.time()

            pixmap = cv_to_pixmap(result_img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

            self.result_text.clear()
            for i, box in enumerate(boxes):
                cls, conf, xmin, ymin = box[0], box[1], box[2], box[3]
                if self.current_lang == 'zh':
                    self.result_text.append(f"类别: {cls}, 置信度: {conf:.2f}, 坐标: [{xmin:.2f}, {ymin:.2f}]")
                else:
                    self.result_text.append(f"Class: {cls}, Confidence: {conf:.2f}, Coords: [{xmin:.2f}, {ymin:.2f}]")

            self.time_label.setText(f"{self.translations[self.current_lang]['time']} {end - start:.3f} s")
            self.count_label.setText(f"{self.translations[self.current_lang]['count']} {len(boxes)}")

            self.table.setRowCount(0)
            for i, box in enumerate(boxes):
                cls, conf, xmin, ymin = box[0], box[1], box[2], box[3]
                self.class_label.setText(f"{self.translations[self.current_lang]['type']} {cls}")
                self.conf_label.setText(f"{self.translations[self.current_lang]['confidence']} {conf*100:.2f} %")
                self.table.insertRow(i)
                self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.table.setItem(i, 1, QTableWidgetItem(cls))
                self.table.setItem(i, 2, QTableWidgetItem(f"{conf*100:.2f}%"))
                self.table.setItem(i, 3, QTableWidgetItem(str(xmin)))
                self.table.setItem(i, 4, QTableWidgetItem(str(ymin)))

            self.predicted_image = result_img

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def save_result(self):
        if hasattr(self, 'predicted_image'):
            path, _ = QFileDialog.getSaveFileName(self, "保存图像", "result.jpg", "Images (*.jpg *.png)")
            if path:
                from cv2 import imwrite
                imwrite(path, self.predicted_image)

    def change_model(self, index):
        path = self.model_paths[index]
        self.detector = Detector(path)
        QMessageBox.information(self, "提示", f"已切换模型：{path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.png"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
