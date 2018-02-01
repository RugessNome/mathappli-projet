
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtGui import QPixmap

from drawwidget import DrawWidget
from reco import Recognizer
import cnn
import numpy as np

draw = None
recognizer = None

def on_button_clicked():
    img = draw.getImage()
    #pix = QPixmap.fromImage(img)
    #pix.save('test.png')
    img = draw.getNumpyImage()
    img = np.expand_dims(img, axis = 2)
    result = cnn.single_prediction(img)[0][0]
    #result = recognizer.predict(img)
    #import mnist
    #mnist.show(img)
    QMessageBox.information(draw, 'Prediction', str(result), QMessageBox.Ok, QMessageBox.Ok)

def on_show_button_clicked():
    img = draw.getNumpyImage()
    import mnist
    mnist.show(img)

def on_contour_button_clicked():
    img = draw.getNumpyImage()
    import features
    features.plot_fourier_approx(img, 7, 200)

def on_clear_button_clicked():
    draw.clear()

if __name__ == '__main__':
    cnn.fit_model(epochs = 10, batch_size = 32)
    #recognizer = Recognizer(['loops', 'zones', 'fourier_image', 'fourier_contour'])
    app = QApplication(sys.argv)
    widget = QWidget()
    layout = QVBoxLayout()
    draw = DrawWidget()
    layout.addWidget(draw)
    button = QPushButton('Reco')
    layout.addWidget(button)
    button.clicked.connect(on_button_clicked)
    button = QPushButton('Show')
    layout.addWidget(button)
    button.clicked.connect(on_show_button_clicked)
    button = QPushButton('Contour')
    layout.addWidget(button)
    button.clicked.connect(on_contour_button_clicked)
    button = QPushButton('Clear')
    layout.addWidget(button)
    button.clicked.connect(on_clear_button_clicked)
    widget.setLayout(layout)
    widget.show()
    sys.exit(app.exec_())

