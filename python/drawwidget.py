
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QLabel, QSpinBox
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QSize

import numpy as np

class DrawWidget(QWidget):
    def __init__(self, **kwargs):
        ret = super().__init__(**kwargs)
        self.setFixedSize(280, 280)
        self.path = QPainterPath()
        return ret
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        print('clicked')
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.path.lineTo(event.pos())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(Qt.NoPen)
        p.drawRect(self.rect())
        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(10)
        p.setPen(pen)
        p.drawPath(self.path)

    def clear(self):
        self.path = QPainterPath()
        self.update()

    def getImage(self):
        image = QImage(280, 280, QImage.Format_Grayscale8)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.setPen(QPen())
        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(10)
        painter.setPen(pen)
        painter.drawPath(self.path)
        return image;

    def getNumpyImage(self):
        image = self.getImage().scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ret = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                ret[j][i] = 255 - image.pixelColor(i, j).red()
        return ret

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = DrawWidget()
    widget.show()
    sys.exit(app.exec_())