
import mnist 
from mnist import show
import features

import numpy as np

def test_feature_set(feats, scale = True):
    Xtrain = features.get_features(feats, 'training')
    Xtest = features.get_features(feats, 'testing')
    images, ytrain = mnist.load(mnist.MNIST_TRAINING_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
    images, ytest = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_PAIR_OF_LIST)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import neighbors #k-nearest neighboors
    from sklearn import linear_model
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        Xtrain = sc_x.fit_transform(Xtrain)
        Xtest = sc_x.transform(Xtest)

    def prediction_rate(preds):
        count=0
        for i in range(0,len(preds)):
            if preds[i]==ytest[i]:
                count+=1
        return 100 * count / len(preds)

    def test_classifier(cl, name):
        print("Fitting ", name, "...")
        cl.fit(Xtrain, ytrain)
        y_predict = cl.predict(Xtest)
        print("Prediction rate : ", prediction_rate(y_predict))

    #test_classifier(neighbors.KNeighborsClassifier(), "KNeighborsClassifier")
    #test_classifier(linear_model.SGDClassifier(), "SGDClassifier")
    #test_classifier(GradientBoostingClassifier(), "GradientBoostingClassifier")
    test_classifier(RandomForestClassifier(n_estimators=10), "RandomForestClassifier")


class FeatureBasedClassifier(object):
    def __init__(self, classifier, feature_list = None, **kwargs):
        super().__init__(**kwargs)
        if type(classifier) is str:
            self.load(classifier)
        else:
            self.classifier = classifier
            self.classifier.feature_names = feature_list
        
    def feature_names(self):
        return self.classifier.feature_names

    def fit(self, X, Y):
        feats = np.array([features.get_features(self.feature_names(), x) for x in X])
        self.classifier.fit(feats, Y)

    def fit_features(self, feats, Y):
        self.classifier.fit(feats, Y)

    def predict(self, X):
        feats = np.array([features.get_features(self.feature_names(), x) for x in X])
        return self.classifier.predict(feats)

    def predict_proba(self, X):
        feats = np.array([features.get_features(self.feature_names(), x) for x in X])
        return self.classifier.predict_proba(feats)

    def save(self, name):
        from sklearn.externals import joblib
        joblib.dump(self.classifier, 'cache/' + name) 
    
    def load(self, name):
        from sklearn.externals import joblib
        self.classifier = joblib.load('cache/' + name) 




def vectorize(y):
    # Turns a desired output (e.g. 7) into a vector having a single 1
    # at the desired position (e.g. (0, 0, 0, 0, 0, 0, 1, 0, 0))
    ret = np.zeros(10)
    ret[y] = 1
    return ret

class NeuralNetworkClassifier(object):
    def __init__(self, layers=[30], epochs=30, batch_size=10, learning_rate = 3.0, **kwargs):
        super().__init__(**kwargs)
        import network
        sizes = [28*28] + layers + [10]
        self.network = network.Network(sizes)
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        training_data = [(x, vectorize(y)) for x, y in zip(X,Y)]
        self.network.fit(training_data, self.epochs, self.batch_size, self.learning_rate)

    def predict(self, X):
        return [np.argmax(self.network.feedforward(x)) for x in X]

    def predict_proba(self, X):
        def make_proba(x):
            return x / x.sum()
        return [make_proba(self.network.feedforward(x)) for x in X]

    def load(self, name='ann'):
        self.network.load(name)

    def save(self, name='ann'):
        self.network.save(name)


def example1():
    from sklearn.ensemble import RandomForestClassifier
    cla = FeatureBasedClassifier(RandomForestClassifier(), ['zones', 'fourier_image'])
    try:
        cla.load('rf_example1')
    except:
        images, labels = mnist.load('train', mnist.MNIST_FORMAT_PAIR_OF_LIST)
        feats = features.get_features(['zones', 'fourier_image'], 'training')
        cla.fit_features(feats, labels)
        cla.save('rf_example1')
    images, labels = mnist.load('test', mnist.MNIST_FORMAT_PAIR_OF_LIST)
    for i in range(10):
        print('Prediction = ', cla.predict([images[i]]))
        print('Expected = ', labels[i])

def example2():
    import network
    net = network.Network([28*28, 30, 10])
    training = mnist.load('training', mnist.MNIST_FORMAT_LIST_OF_PAIR)
    test = mnist.load('test', mnist.MNIST_FORMAT_LIST_OF_PAIR)

    training = [(x.reshape(28*28) / 255, vectorize(y)) for x,y in training]
    test = [(x.reshape(28*28) / 255, y) for x,y in test]

    net.fit(training, 30, 10, 3.0, test)

