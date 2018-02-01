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

# Loading dataset mnist
X_train, y_train = mnist.load(mnist.MNIST_TRAINING_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST, shape=(28,28,1))
X_test, y_test = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST, shape=(28,28,1))

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


# Fits the model
def fit_model(epochs = 10, batch_size = 32):
    global classifier
    classifier = build_classifier()
    classifier.fit(X_train, y_train_vect, validation_data = (X_test, y_test_vect), epochs = epochs, batch_size=batch_size)
    
# Evaluates the model : variance and mean of accuracy (takes a long time)
def eval_model(nb_epochs = 10, batch_size = 32, cv = 10):
    global accuracies
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_size, nb_epoch = nb_epochs)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train_vect, cv = cv)
    print("Acurracy mean :", accuracies.mean())
    print("Accuracy variance :", var_accuracy = accuracies.std())


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
def predict(classifier):
    # Evaluation of the loss (depends on the metrics)
    global loss, cm
    loss = classifier.evaluate(X_test, y_test_vect, batch_size=32)
    # Evaluation of the pourcentage of failure 
    print("\n \n Success : ",loss[1]*100,"%")
    # Confusion Matrix
    # Reminder : A confusion matrix C is such that C_{i, j} is equal to the number 
    # of observations known to be in group i but predicted to be in group j.
    y_predict_vect = classifier.predict(X_test)
    y_predict = ypred(y_predict_vect)
    cm = confusion_matrix(y_test,y_predict)
    
# Visualize confusion matrix as a heatmap
def cm_visualisation():
    ax = sb.heatmap(cm, cmap="BuPu")
    ax.invert_yaxis()
    plt.yticks(rotation=0); 
    
# Returns the 3 first results with their probabilities for the prediction of image
# image must be sized (28,28,1)
# use np.expand_dims() if you want to increase dimensions
def single_prediction(image,printResult=True):
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