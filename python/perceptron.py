
import numpy as np

# Heaviside step function
def Heaviside(x):
	return 1 if x >= 0 else 0
	
H = Heaviside

class Perceptron(object):
	def __init__(self, w, b, **kwargs):
		self.weight = np.array(w)
		self.bias = b
		return super().__init__(**kwargs)

	def feed(self, x):
		return H(np.dot(self.weight, x) + self.bias)


def or_perceptron(n = 2):
	return Perceptron(np.ones(n), -1)
		
def test_or_perceptron():
	p = or_perceptron()
	print("(0,0) -> ", p.feed([0, 0]))
	print("(1,0) -> ", p.feed([1, 0]))
	print("(0,1) -> ", p.feed([0, 1]))
	print("(1,1) -> ", p.feed([1, 1]))
	
def and_perceptron(n = 2):
	return Perceptron(np.ones(n), -n)
	
def test_and_perceptron():
	p = and_perceptron()
	print("(0,0) -> ", p.feed([0, 0]))
	print("(1,0) -> ", p.feed([1, 0]))
	print("(0,1) -> ", p.feed([0, 1]))
	print("(1,1) -> ", p.feed([1, 1]))

def nand_perceptron():
	return Perceptron(np.array([-2, -2]), 3)
	
def test_nand_perceptron():
	p = nand_perceptron()
	print("(0,0) -> ", p.feed([0, 0]))
	print("(1,0) -> ", p.feed([1, 0]))
	print("(0,1) -> ", p.feed([0, 1]))
	print("(1,1) -> ", p.feed([1, 1]))


# Evaluates the perceptron p with the list of X
# and Y (list)
# Returns a 2D array Z with indexing convention Z[x][y]
def eval(p, X, Y):
	Z = []
	for x in X:
		z = [p.feed([x, y]) for y in Y]
		Z.append(z)
	return np.array(Z)


# Plots an image outputed by eval()
def plotimg(Z):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # imshow uses the convention Z[row][column] so we need to 
    # transpose the output from eval
    im = plt.imshow(np.transpose(Z), interpolation='bilinear', cmap=cm.RdYlGn,
                    origin='lower', extent=[-3, 3, -3, 3],
                    vmax=abs(Z).max(), vmin=-abs(Z).max())
    plt.show()

def example1():
    p = Perceptron(np.array([-1, 1]), 0)

    delta = 0.025
    X = Y = np.arange(-3.0, 3.0, delta)
    Z = eval(p, X, Y)

    plotimg(Z)



class PerceptronNetwork(object):
	def __init__(self, layers):
		self.layers = layers

	def add_layer(self, l):
		self.layers.append(l)

	def layer(self, index):
		return self.layers[index]

	def add_and_layer(self):
		nb_input = len(self.layers[-1])
		p = Perceptron(np.array([1]*nb_input), -nb_input)
		self.layers.append([p])

	def feed(self, x):
		for layer in self.layers:
			y = []
			for p in layer:
				y.append(p.feed(x))
			x = y
		return x[0] if len(x) == 1 else np.array(x)


def example2():
    net = PerceptronNetwork([])
    net.add_layer([Perceptron([8/3, -1], 2), Perceptron([-8/3, -1], 2), Perceptron([2/3, 1], 1), Perceptron([-2/3, 1], 1)])
    net.add_and_layer()

    delta = 0.0125
    X = Y = np.arange(-3.0, 3.0, delta)
    Z = eval(net, X, Y)

    plotimg(Z)


def example3():
    net = PerceptronNetwork([])
    net.add_layer([Perceptron([8/3, -1], 2), Perceptron([-8/3, -1], 2), Perceptron([2/3, -1], -1), Perceptron([-2/3, -1], -1)])
    net.add_layer([Perceptron([1, 1, 0, 0], -2), Perceptron([0, 0, 1, 1], -2)])
    net.add_layer([Perceptron([1, -1], -1)])

    delta = 0.0125
    X = Y = np.arange(-3.0, 3.0, delta)
    Z = eval(net, X, Y)

    plotimg(Z)
