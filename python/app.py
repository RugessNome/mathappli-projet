
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtGui import QPixmap

from drawwidget import DrawWidget
from reco import Recognizer

draw = None
recognizer = None

def on_button_clicked():
    img = draw.getImage()
    #pix = QPixmap.fromImage(img)
    #pix.save('test.png')
    img = draw.getNumpyImage()
    result = recognizer.predict(img)
    #import mnist
    #mnist.show(img)
    QMessageBox.information(draw, 'Prediction', str(result), QMessageBox.Ok, QMessageBox.Ok)

def on_clear_button_clicked():
    draw.clear()

if __name__ == '__main__':
    recognizer = Recognizer(['loops', 'fourier_contour_a0', 'fourier_contour_b0', 'fourier_contour_c0', 'fourier_contour_d0',
                             'fourier_contour_a1', 'fourier_contour_b1', 'fourier_contour_c1', 'fourier_contour_d1'])
    app = QApplication(sys.argv)
    widget = QWidget()
    layout = QVBoxLayout()
    draw = DrawWidget()
    layout.addWidget(draw)
    button = QPushButton('Reco')
    layout.addWidget(button)
    button.clicked.connect(on_button_clicked)
    button = QPushButton('Clear')
    layout.addWidget(button)
    button.clicked.connect(on_clear_button_clicked)
    widget.setLayout(layout)
    widget.show()
    sys.exit(app.exec_())

