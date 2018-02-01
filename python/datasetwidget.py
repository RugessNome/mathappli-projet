
from math import ceil

import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QSize

import numpy as np

import mnist


def image_to_pixmap(img):
    ret = QImage(28,28, QImage.Format_RGB32)
    for y in range(len(img)):
        for x in range(len(img[y])):
            c = 255-img[y][x]
            ret.setPixelColor(x, y, QColor(c, c, c))
    return QPixmap.fromImage(ret)

class ImageView(QWidget):
    def __init__(self, image, label, prediction = None, **kwargs):
        ret = super().__init__(**kwargs)
        layout = QVBoxLayout()
        self.imagewidget = QLabel()
        layout.addWidget(self.imagewidget)
        self.labelwidget = QLabel()
        layout.addWidget(self.labelwidget)
        self.setLayout(layout)
        return ret

    def reset(self, image, label, prediction = None):
        if image is None:
            self.labelwidget.setText('')
            self.imagewidget.setPixmap(QPixmap())
            return
        self.image = image
        self.label = label
        self.prediction = prediction
        self.ratio = 4.0
        self.labelwidget.setText(self.label_text())
        self.check_prediction()
        self.imagewidget.setPixmap(self.scaled_pixmap())

    def scaled_pixmap(self):
        pix = image_to_pixmap(self.image)
        pix = pix.scaled(int(28*self.ratio), int(28*self.ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    def label_text(self):
        if self.prediction is None:
            return str(self.label)
        return str(self.label) + '(' + str(self.prediction) + ')'

    def check_prediction(self):
        if self.prediction is None:
            return
        if self.prediction == self.label:
            return
        pal = self.labelwidget.palette()
        pal.setColor(Qt.ForegroundRole, Qt.red)
        self.labelwidget.setPalette(pal)


class DatasetWidget(QWidget):
    def __init__(self, dataset, predictions = None, **kwargs):
        ret = super().__init__(**kwargs)
        self.dataset = dataset
        self.data = mnist.load(dataset, mnist.MNIST_FORMAT_PAIR_OF_LIST)
        self.predictions = predictions
        self.current_page = -1
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.cells = []
        self.set_cells(4, 4)
        return ret

    def reset(self, dataset, predictions = None):
        self.dataset = dataset
        self.data = mnist.load(dataset, mnist.MNIST_FORMAT_PAIR_OF_LIST)
        self.predictions = predictions
        self.current_page = -1
        self.set_current_page(0)

    def set_dataset(self, d):
        if self.dataset == d:
            return

    def row_count(self):
        return self.grid_layout.rowCount()

    def set_row_count(self, rc):
        if rc == self.row_count():
            return
        self.set_cells(rc, self.column_count())
    
    def column_count(self):
        return self.grid_layout.columnCount()

    def set_column_count(self, cc):
        if cc == self.column_count():
            return
        self.set_cells(self.row_count(), cc)

    def set_cells(self, rows, cols):
        while self.grid_layout.count() != 0:
            self.grid_layout.takeAt(0)
        for c in self.cells:
            c.deleteLater()
        self.cells = []
        for x in range(cols):
            for y in range(rows):
                cell = ImageView(None, None)
                self.grid_layout.addWidget(cell, y, x)
                self.cells.append(cell)
        self.set_current_page(0)

    def dataset_size(self):
        images, labels = self.data
        return len(images)

    def get_image(self, i):
        images, labels = self.data
        if i >= len(images):
            return None
        return images[i]

    def get_label(self, i):
        images, labels = self.data
        if i >= len(labels):
            return None
        return labels[i]

    def get_prediction(self, i):
        if self.predictions is None or i >= len(self.predictions):
            return None
        return self.predictions[i]

    def page_count(self):
        per_page = self.row_count() * self.column_count()
        return int(ceil(self.dataset_size() / per_page))

    def set_current_page(self, p):
        if p == self.current_page or p >= self.page_count():
            return
        self.current_page = p
        per_page = self.row_count() * self.column_count()
        first = self.current_page * per_page
        for i in range(first, first+per_page):
            self.cells[i-first].reset(self.get_image(i), self.get_label(i), self.get_prediction(i))
    

if __name__ == '__main__':
    from PyQt5.QtWidgets import QSpinBox
    app = QApplication(sys.argv)
    widget = QWidget()
    datasetwidget = DatasetWidget('test')
    spinbox = QSpinBox()
    spinbox.setRange(0, datasetwidget.page_count())
    layout = QVBoxLayout()
    layout.addWidget(datasetwidget)
    layout.addWidget(spinbox)
    widget.setLayout(layout)
    widget.show()

    def change_page():
        datasetwidget.set_current_page(spinbox.value())
    spinbox.valueChanged.connect(change_page)

    sys.exit(app.exec_())