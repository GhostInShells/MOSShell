import sys
import threading
from typing import Callable
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PIL import Image, ImageDraw

__all__ = ["SimpleImageViewer", "run_img_viewer"]


class ImageSignaler(QObject):
    update_image = pyqtSignal(QImage)


class SimpleImageViewer(QMainWindow):
    def __init__(self, window_title: str = "PIL Live Viewer"):
        super().__init__()
        self.setWindowTitle(window_title)
        self.resize(800, 600)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)

        self.signaler = ImageSignaler()
        self.signaler.update_image.connect(self._display_image)
        self.current_qimage = None

    def _display_image(self, q_image):
        self.current_qimage = q_image
        self._update_pixmap()

    def _update_pixmap(self):
        if self.current_qimage:
            pixmap = QPixmap.fromImage(self.current_qimage)
            scaled = pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def set_pil_image(self, pil_img: Image.Image) -> None:
        """支持 PIL 图像的线程安全接口"""
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(data, pil_img.size[0], pil_img.size[1], QImage.Format.Format_RGB888)
        self.signaler.update_image.emit(qimg.copy())


def run_img_viewer(callback: Callable[[SimpleImageViewer], None]):
    app = QApplication(sys.argv)
    viewer = SimpleImageViewer()
    viewer.show()
    threading.Thread(target=callback, args=(viewer,), daemon=True).start()
    sys.exit(app.exec())
