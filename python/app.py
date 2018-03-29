
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtGui import QPixmap

from drawwidget import DrawWidget
import numpy as np

draw = None

classifiers = []

import cnn
cnn_cla = cnn.CNNClassifier()
cnn_cla.load('cnn_example3')
classifiers.append(('cnn', cnn_cla))

import reco
ann_cla = reco.NeuralNetworkClassifier()
ann_cla.load('ann')
classifiers.append(('neural network', ann_cla))

rf_cla = reco.FeatureBasedClassifier('rf_example1')
classifiers.append(('random forest', rf_cla))

rf_cla_2 = reco.FeatureBasedClassifier('rf_example2')
classifiers.append(('random forest (loops)', rf_cla_2))

rf_cla_3 = reco.FeatureBasedClassifier('rf_example3')
classifiers.append(('random forest (loops, moments)', rf_cla_3))

current_classifier_combobox = None

def on_button_clicked():
    img = draw.getImage()
    #pix = QPixmap.fromImage(img)
    #pix.save('test.png')
    img = draw.getNumpyImage()
    cla_name, cla = classifiers[current_classifier_combobox.currentIndex()]
    if type(cla) is cnn.CNNClassifier or type(cla) is reco.NeuralNetworkClassifier:
        img = img/255
    if type(cla) is cnn.CNNClassifier:
        img = np.expand_dims(img, axis = 2)
        img = np.expand_dims(img, axis = 0)
    elif type(cla) is reco.NeuralNetworkClassifier:
        img = img.reshape((28*28))
    elif type(cla) is reco.FeatureBasedClassifier:
        pass
    probas = cla.predict_proba([img])[0]
    result = ''
    for i in range(3):
        index = np.argmax(probas)
        result += '{0} : ({1:.4f} %)\n'.format(index, probas[index]) 
        probas[index] = 0
    QMessageBox.information(draw, 'Prediction ' + cla_name, result, QMessageBox.Ok, QMessageBox.Ok)

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


g_dataset_widget = None
def on_visualize_button_clicked():
    cla_name, cla = classifiers[current_classifier_combobox.currentIndex()]
    import mnist
    images, ytest = mnist.load('test', mnist.MNIST_FORMAT_PAIR_OF_LIST)
    if type(cla) is cnn.CNNClassifier or type(cla) is reco.NeuralNetworkClassifier:
        images = images/255
    else:
        assert type(cla) is reco.FeatureBasedClassifier
        import features
        images = features.get_features(cla.feature_names(), 'testing')

    if type(cla) is cnn.CNNClassifier:
        images = images.reshape((len(images), 28, 28, 1))
    elif type(cla) is reco.NeuralNetworkClassifier:
        images = images.reshape((len(images), 28*28))
    elif type(cla) is reco.FeatureBasedClassifier:
        pass

    predictions = None
    if type(cla) is cnn.CNNClassifier or type(cla) is reco.NeuralNetworkClassifier:
        predictions = cla.predict(images)
    elif type(cla) is reco.FeatureBasedClassifier:
        predictions = cla.predict_features(images)

    acc = (ytest == predictions).sum() / len(ytest)

    import datasetwidget
    global g_dataset_widget
    g_dataset_widget = datasetwidget.build_dataset_widget_with_controls('test', predictions)
    g_dataset_widget.setWindowTitle('Accuracy = {}'.format(acc))
    g_dataset_widget.show()

if __name__ == '__main__':
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
    current_classifier_combobox = QComboBox()
    for (name, _) in classifiers:
        current_classifier_combobox.addItem(name)
    layout.addWidget(current_classifier_combobox)
    button = QPushButton('Clear')
    layout.addWidget(button)
    button.clicked.connect(on_clear_button_clicked)
    button = QPushButton('Visualize')
    layout.addWidget(button)
    button.clicked.connect(on_visualize_button_clicked)
    widget.setLayout(layout)
    widget.show()
    sys.exit(app.exec_())

