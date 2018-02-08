
import random
import numpy as np

# Code adapted from http://neuralnetworksanddeeplearning.com

class Network(object):

    def __init__(self, sizes):
        """
        Constructs a neural network that is randomly initialized using 
        the normal distribution.
        Parameters:
        - `sizes` : a list containing the number of neurons in each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Random initialization of biases and weights.
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Returns the output of the network for the input `a`.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def fit(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Trains the neural network using mini-batch stochastic gradient descent.  
        Parameters:
        - `training_data` : a list of tuples `(x, y)` representing the training inputs 
        and the desired outputs.  
        - `epochs` : an integer representing the number of epochs.
        - `mini_batch_size` : an integer representing the number of 
        elements which are used to evaluate an approximation of 
        the derivative of the cost function.
        - `test_data` (optional) : if provided then the network will be evaluated 
        against the test data after each epoch, and partial progress printed out.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Updates the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        Parameters:
        - ``mini_batch`` : a list of tuples ``(x, y)``
        - ``eta`` :  learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuples ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        Parameters:
        - `x` : the neural net input
        - `y` : the desired output
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2])
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return (output_activations-y)

    def save(self, name='ann'):
        import os
        import os.path
        if not os.path.isdir("cache/"):
            os.mkdir('cache')
        np.save('cache/' + name + '_weights', np.array(self.weights))
        np.save('cache/' + name + '_biases', np.array(self.biases))

    def load(self, name='ann'):
        self.weights = np.load('cache/' + name + '_weights.npy')
        self.biases = np.load('cache/' + name + '_biases.npy')
        self.num_layers = len(self.biases)
        self.sizes = [len(b) for b in self.biases]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))