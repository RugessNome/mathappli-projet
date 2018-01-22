import os
import struct
import numpy as np


# Function for importing the MNIST data set from local files
def __read(dataset, path):

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl


MNIST_TRAINING_DATA = 0
MNIST_TEST_DATA = 1

MNIST_FORMAT_LIST_OF_PAIR = 1
MNIST_FORMAT_PAIR_OF_LIST = 2

def load(what, format = MNIST_FORMAT_LIST_OF_PAIR, shape = None, path = None):
    """
    Returns the MNIST data set in the desired format.

    Parameters :
      - ``what``: either MNIST_TRAINING_DATA to load the training data
         or MNIST_TEST_DATA to load the test data
      - ``format`` : can be one of the following
        - MNIST_FORMAT_LIST_OF_PAIR : the data will be returned as 
          a list of pair (img, label)
        - MNIST_FORMAT_PAIR_OF_LIST : the data will be returned as 
          a pair of lists (images, labels) where images[i] and 
          labels[i] are the image and label of the i-th image.
      - ``shape`` : if not None, the image data will be reshaped,
        by default the shape is (28,28).
      - ``path`` : the directory in which the dataset is to be found,
        this parameter can be omitted if the keras module is available.
    
    This function will attempt to load the dataset from local files 
    found in ``path`` unless it is none, in which case the dataset will 
    be loaded from the keras module
    """
    images, label = None, None

    if path != None:
        images, labels = __read('training' if what == MNIST_TRAINING_DATA else 'testing', path)
    else:
        try:
            import keras.datasets.mnist
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            images, labels = (X_train, y_train) if what == MNIST_TRAINING_DATA else (X_test, y_test)
        except ModuleNotFoundError:
            raise RuntimeError('Could not load MNIST dataset')

    if shape != None:
        images = np.array([img.reshape(shape) for img in images])
    if format == MNIST_FORMAT_PAIR_OF_LIST:
        return images, labels
    else:
        return [(images[i], labels[i]) for i in range(len(images))]


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    if image.shape != (28,28):
        image = image.reshape((28,28))
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

