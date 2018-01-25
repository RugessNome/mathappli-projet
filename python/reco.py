
import mnist 
from mnist import show
import features

import numpy as np



def test_feature_set(feats):
    Xtrain = features.get_features(feats, 'training')
    Xtest = features.get_features(feats, 'testing')
    images, ytrain = mnist.load(mnist.MNIST_TRAINING_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
    images, ytest = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import neighbors #k-nearest neighboors
    from sklearn import linear_model

    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    Xtrain = sc_x.fit_transform(Xtrain)
    Xtest = sc_x.transform(Xtest)

    print("Fitting KNeighborsClassifier...")
    knn = neighbors.KNeighborsClassifier()
    knn.fit(Xtrain, ytrain)
    y_predict_knn = knn.predict(Xtest)

    print("Fitting SGDClassifier...")
    lin = linear_model.SGDClassifier()
    lin.fit (Xtrain, ytrain)
    y_predict_lin = lin.predict(Xtest)

    print("Fitting GradientBoostingClassifier...")
    gbc = GradientBoostingClassifier()
    gbc.fit(Xtrain, ytrain)
    y_predict_gbc = gbc.predict(Xtest)

    print("Fitting RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(Xtrain, ytrain)
    y_predict_rf = rf.predict(Xtest)

    def prediction_rate(preds):
        count=0
        for i in range(0,len(preds)):
            if preds[i]==ytest[i]:
                count+=1
        return 100 * count / len(preds)

    print("Prediction rates:")
    print("KNeighborsClassifier : ", prediction_rate(y_predict_knn))
    print("SGDClassifier : ", prediction_rate(y_predict_lin))
    print("GradientBoostingClassifier : ", prediction_rate(y_predict_gbc))
    print("RandomForestClassifier : ", prediction_rate(y_predict_rf))


class Recognizer(object):
    def __init__(self, feature_list, **kwargs):
        super().__init__(**kwargs)
        from sklearn.ensemble import RandomForestClassifier
        self.features = feature_list
        Xtrain = features.get_features(self.features, 'training')
        images, ytrain = mnist.load(mnist.MNIST_TRAINING_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        Xtrain = self.scaler.fit_transform(Xtrain)
        self.classifier = RandomForestClassifier(n_estimators=10)
        self.classifier.fit(Xtrain, ytrain)
                
    
    def predict(self, img):
        Xpredic = features.get_features(self.features, image = img)
        Xpredic = self.scaler.transform(Xpredic)
        return self.classifier.predict(Xpredic)[0]


def example1():
    r = Recognizer(['loops', 'fourier_contour_a0', 'fourier_contour_b0', 'fourier_contour_c0', 'fourier_contour_d0',
                    'fourier_contour_a1', 'fourier_contour_b1', 'fourier_contour_c1', 'fourier_contour_d1'])
    images, labels = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
    for i in range(10):
        print('Prediction = ', r.predict(images[i]))
        print('Expected = ', labels[i])