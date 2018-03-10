#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:25:55 2017

@author: rouillon

"""
#USEFUL LIBRARIES :

import mnist

#from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# DATA PREPROCESSING :


def fetch_data():
    # Loading dataset mnist
    X_train, y_train = mnist.load('train', mnist.MNIST_FORMAT_PAIR_OF_LIST, shape=(28,28,1))
    X_test, y_test = mnist.load('test', mnist.MNIST_FORMAT_PAIR_OF_LIST, shape=(28,28,1))
    
    # Normalizing inputs 
    X_train = X_train / 255
    X_test = X_test / 255

    # Reducing the size of the training set for faster results (optional)
    #upperbound=10000
    #X_train = X_train[0:upperbound]
    #y_train = y_train[0:upperbound]
    
    # Categorical encoding of outputs
    y_train_vect = to_categorical(y_train)
    y_test_vect = to_categorical(y_test)

    return X_train, y_train_vect, X_test, y_test_vect


def augment_data(images, labels, batch_size, batch_count, rotation = 0, hshift = 0.25, vshift = 0.25, zoom = 0.25):
    # Data augmentation
    assert(batch_size <= len(images))
    assert(len(images) / batch_size == int(len(images) / batch_size))
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rotation_range=rotation, width_shift_range=hshift, height_shift_range=vshift, zoom_range=zoom, data_format='channels_last')
    datagen.fit(images)
    flow = datagen.flow(images, labels, batch_size=batch_size)
    x_batch, y_batch = flow.__next__()
    for i in range(batch_count-1):
        x_batch_iter, y_batch_iter = flow.__next__()
        x_batch = np.concatenate((x_batch, x_batch_iter))
        y_batch = np.concatenate((y_batch, y_batch_iter))
    assert(len(x_batch) == batch_count * batch_size)
    assert(len(y_batch) == batch_count * batch_size)
    return x_batch, y_batch

# CNN IMPLEMENTATION :

def build_classifier():
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Overfitting reduction - Dropout
    classifier.add(Dropout(0.1)) 
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation='relu'))
    classifier.add(Dropout(0.1)) # Overfitting reduction - Dropout
    classifier.add(Dense(units = 10, activation='softmax'))
    # Compiling the CNN
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return classifier

def step_by_step_build():
    classifier = Sequential()
    print("Adding a convolution layer")
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    print("Input shape is :")
    print(classifier.input_shape)
    print("Output shape is :")
    print(classifier.output_shape)
    print("Adding a MaxPooling layer")
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    print(classifier.output_shape)
    print("Adding a second Conv2D layer")
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    print(classifier.output_shape)
    print("Adding a second MaxPooling2D layer")
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    print(classifier.output_shape)
    print("Adding a Flatten layer")
    classifier.add(Flatten())
    print(classifier.output_shape)
    print("Adding a Dropout layer")
    classifier.add(Dropout(0.1))
    print(classifier.output_shape)
    print("Adding a Dense layer")
    classifier.add(Dense(units = 128, activation='relu'))
    print(classifier.output_shape)
    print("Adding a Dropout layer")
    classifier.add(Dropout(0.1))
    print(classifier.output_shape)
    print("Adding a Dense layer")
    classifier.add(Dense(units = 10, activation='softmax'))
    print(classifier.output_shape)
    try:
        from keras.utils import plot_model
        plot_model(classifier, to_file='model.png')
        print("Model plotted in model.png")
    except:
        pass
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return classifier

def save_classifier(classifier, name='cnn'):
    import os
    import os.path
    if not os.path.isdir("cache/"):
        os.mkdir('cache')
    classifier.save('cache/' + name)

def load_classifier(name='cnn'):
    from keras.models import load_model
    global classifier
    classifier = load_model('cache/' + name)
    return classifier

# Fits the model
def fit_model(X_train, y_train, epochs = 10, batch_size = 32):
    classifier = build_classifier()
    from sklearn.model_selection import train_test_split
    X, X_validation, y, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    classifier.fit(X, y, validation_data = (X_validation, y_validation), epochs = epochs, batch_size=batch_size)
    return classifier
    
# Evaluates the model : variance and mean of accuracy (takes a long time)
def eval_model(X_train, y_train, nb_epochs = 10, batch_size = 32, cv = 10):
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_size, nb_epoch = nb_epochs)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = cv)
    print("Acurracy mean :", accuracies.mean())
    print("Accuracy variance :", var_accuracy = accuracies.std())
    return accuracies


# PREDICTIONS AND LOSS  :

def ypred(pred):
    y=[]
    for i in range(0, len(pred)):
        pred_i = pred[i,].tolist()
        index = pred_i.index(max(pred_i))
        y.append(index)
    y = np.array(y)
    return y  

# Determines loss, accuracy and confusion matrix
def predict(classifier, X_test, y_test):
    # Evaluation of the loss (depends on the metrics)
    loss = classifier.evaluate(X_test, y_test, batch_size=32)
    # Evaluation of the pourcentage of failure 
    print("\n \n Success : ",loss[1]*100,"%")
    # Confusion Matrix
    # Reminder : A confusion matrix C is such that C_{i, j} is equal to the number 
    # of observations known to be in group i but predicted to be in group j.
    y_predict_vect = classifier.predict(X_test)
    y_predict = ypred(y_predict_vect)
    cm = confusion_matrix(ypred(y_test),y_predict)
    return loss, cm
    
# Visualize confusion matrix as a heatmap
def cm_visualisation(cm):
    ax = sb.heatmap(cm, cmap="BuPu")
    ax.invert_yaxis()
    plt.yticks(rotation=0); 
    plt.show()
    
# Returns the 3 first results with their probabilities for the prediction of image
# image must be sized (28,28,1)
# use np.expand_dims() if you want to increase dimensions
def single_prediction(classifier, image,printResult=True):
    test = np.expand_dims(image, axis = 0)
    l=classifier.predict(test)
    l=l[0].tolist()
    ind1 = l.index(max(l))
    proba1 = l[ind1]
    l[ind1] = 0
    ind2 = l.index(max(l))
    proba2 = l[ind2]
    l[ind2] = 0
    ind3 = l.index(max(l))
    proba3 = l[ind3]
    if printResult == True :
        print("prédictions : \n", str(ind1), " p =", str(round(proba1,5)), "\n", 
                                  str(ind2), " p =", str(round(proba2,5)), "\n", 
                                  str(ind3), " p =", str(round(proba3,5)),)
    return [[ind1,proba1],[ind2,proba2],[ind3,proba3]]
    
# TUNING THE CNN : 
    
def build_classifier_tuned(optimizer):
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Overfitting reduction - Dropout
    classifier.add(Dropout(0.1)) 
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation='relu'))
    classifier.add(Dropout(0.1)) # Overfitting reduction - Dropout
    classifier.add(Dense(units = 10, activation='softmax'))
    # Compiling the CNN
    classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return classifier

# Determines the best parameters defined in the dictionnary parameters (takes a while...)
def best_parameters():
    X_train, y_train, X_test, y_test = fetch_data()
    global best_parameters, best_accuracy
    classifier = KerasClassifier(build_fn = build_classifier_tuned)
    parameters = {'batch_size' : [25,32],
                  'nb_epoch' : [10, 25],
                  'optimizer' : ['adam', 'rmsprop']}
    grid_search = GridSearchCV(estimator = classifier, 
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 1)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_


def example1():
    # This is supposed to show that the CNN has difficulties generalizing
    X_train, y_train, X_test, y_test = fetch_data()
    X_test, y_test = augment_data(X_test, y_test, 10000, 2)
    cla = None
    try:
        cla = load_classifier('cnn_example1')
    except:
        cla = fit_model(X_train, y_train, epochs=2)
        save_classifier(cla, 'cnn_example1')
    loss, cm = predict(cla, X_test, y_test)
    print(loss)
    cm_visualisation(cm)

def example2():
    # This shows how the CNN performs when trained with augmented data 
    # and tested with normal data.
    X_train, y_train, X_test, y_test = fetch_data()
    X_train, y_train = augment_data(X_train, y_train, batch_size=30000, batch_count=3)
    cla = None
    try:
        cla = load_classifier('cnn_example2')
    except:
        cla = fit_model(X_train, y_train, epochs=3)
        save_classifier(cla, 'cnn_example2')
    loss, cm = predict(cla, X_test, y_test)
    print(loss)
    cm_visualisation(cm)

def example3():
    # Last example : both the training and testing data are augmented
    X_train, y_train, X_test, y_test = fetch_data()
    X_train, y_train = augment_data(X_train, y_train, batch_size=30000, batch_count=3)
    X_test, y_test = augment_data(X_test, y_test, batch_size=10000, batch_count=4)
    cla = None
    try:
        cla = load_classifier('cnn_example3')
    except:
        cla = fit_model(X_train, y_train, epochs=3)
        save_classifier(cla, 'cnn_example3')
    loss, cm = predict(cla, X_test, y_test)
    print(loss)
    cm_visualisation(cm)


class CNNClassifier(object):
    def __init__(self, epochs=10, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.cnn = build_classifier()
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        Y_categorical = to_categorical(Y)
        self.cnn.fit(training_data, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X):
        Y = self.cnn.predict(X)
        return [np.argmax(y) for y in Y]

    def predict_proba(self, X):
        return self.cnn.predict(X)

    def load(self, name='ann'):
        self.cnn = load_classifier(name)

    def save(self, name='cnn_classifier'):
        save_classifier(cla, name)

