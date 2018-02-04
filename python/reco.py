
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
        Xpredic = [features.get_features(self.features, img)]
        Xpredic = self.scaler.transform(Xpredic)
        return self.classifier.predict(Xpredic)[0]


def example1():
    r = Recognizer(['loops', 'zones', 'fourier_contour'])
    images, labels = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
    for i in range(10):
        print('Prediction = ', r.predict(images[i]))
        print('Expected = ', labels[i])


def example2():
    import network
    net = network.Network([28*28, 30, 10])
    training = mnist.load('training', mnist.MNIST_FORMAT_LIST_OF_PAIR)
    test = mnist.load('test', mnist.MNIST_FORMAT_LIST_OF_PAIR)

    def vectorize(y):
        # Turns a desired output (e.g. 7) into a vector having a single 1
        # at the desired position (e.g. (0, 0, 0, 0, 0, 0, 1, 0, 0))
        ret = np.zeros(10)
        ret[y] = 1
        return ret

    training = [(x.reshape(28*28) / 255, vectorize(y)) for x,y in training]
    test = [(x.reshape(28*28) / 255, y) for x,y in test]

    net.fit(training, 30, 10, 3.0, test)

