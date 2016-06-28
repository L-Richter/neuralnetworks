import numpy as np
from collections import namedtuple


ActivationFunction = namedtuple('ActivationFunction', ['fun', 'der'])
CostFunction = namedtuple('CostFunction', ['fun', 'der'])


def sigmoid_fun(ar):
    return 1 / (1 + np.exp(-ar))


def sigmoid_derivative(ar):
    return np.exp(-ar) / (1 + np.exp(-ar)) ** 2

sigmoid = ActivationFunction(fun=sigmoid_fun, der=sigmoid_derivative)


def squared_fun(output, results):
    return 0.5 * np.sum((output - results) ** 2)


def squared_derivative(output, results):
    return output - results

sqared_cost = CostFunction(fun=squared_fun, der=squared_derivative)


class Network:
    def __init__(self, layer_sizes):
        self._n_layers = len(layer_sizes)
        self._layers = []
        self._input_layer = InputLayer(layer_sizes[0])
        self._layers.append(self._input_layer)
        for size in layer_sizes[1:]:
            self._layers.append(Layer(size))

        last_layer = None
        for layer in self._layers:
            if last_layer is not None:
                last_layer.set_next_layer(layer)
                layer.set_previous_layer(last_layer)
            last_layer = layer

        self._output_layer = self._layers[-1]
        self._initialize_layers()

        self._output_layer.set_cost(sqared_cost)

    def _initialize_layers(self):
        for layer in self._layers[1:-1]:
            layer.set_random_params()
        self._output_layer.set_zero_params()

    def fit_sgd(self, training_data, epochs=1, batch_size=None, learning_rate=1):
        if batch_size is None:
            batch_size = len(training_data)

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            for k in range(0, len(training_data), batch_size):
                self._input_layer.update(training_data[k:k + batch_size], learning_rate)
            print('Epoch {} done'.format(epoch))

    def feedforward(self, input_data):
        return self._input_layer.feedforward(input_data)

    def get_delta(self, input_data):
        return self._input_layer.get_delta(input_data)


class Layer:
    def __init__(self, size, activation_function=sigmoid):
        self._size = size
        self._activation_function = activation_function
        self._weights = None
        self._biases = np.random.randn(size, 1)
        self._next_layer = None
        self._previous_layer = None
        self._tmp_w = 0
        self._tmp_b = 0

    def set_weights(self, weights):
        self._weights = weights

    def set_biases(self, biases):
        self._biases = biases

    def set_next_layer(self, layer):
        self._next_layer = layer

    def set_previous_layer(self, layer):
        self._previous_layer = layer

    def set_cost(self, cost):
        self._cost = cost

    def get_size(self):
        return self._size

    def get_weights(self):
        return self._weights

    def set_random_params(self):
        surrounding_sizes = self._previous_layer.get_size() + self._next_layer.get_size()
        high = np.sqrt(6 / surrounding_sizes)
        biases = np.random.uniform(low=-high, high=high, size=(self.get_size(), 1))
        weights = np.random.uniform(low=-high, high=high, size=(self.get_size(), self._previous_layer.get_size()))
        self.set_biases(biases)
        self.set_weights(weights)

    def set_zero_params(self):
        biases = np.zeros((self.get_size(), 1))
        weights = np.zeros((self.get_size(), self._previous_layer.get_size()))
        self.set_biases(biases)
        self.set_weights(weights)

    def feedforward(self, input_data):
        output = self._get_output(input_data)
        if self._next_layer is None:
            return output
        else:
            return self._next_layer.feedforward(output)

    def _get_output(self, input_data):
        return self._activation_function.fun(self._get_z(input_data))

    def _get_z(self, input_data):
        return np.dot(self._weights, input_data) + self._biases

    def forward_back_pass(self, input_data, results, learning_rate):
        zs = self._get_z(input_data)
        activations = self._activation_function.fun(zs)
        if self._next_layer is None:
            delta = self._cost.der(activations, results) * self._activation_function.der(zs)
        else:
            delta_next = self._next_layer.forward_back_pass(activations, results, learning_rate)
            delta = np.dot(self._next_layer.get_weights().T, delta_next) *\
                self._activation_function.der(zs)
        self._tmp_w = learning_rate * input_data.T * delta
        self._tmp_b = learning_rate * delta
        return delta

    def update(self):
        self._weights = self._weights - self._tmp_w
        self._biases = self._biases - self._tmp_b

        self._tmp_w = 0
        self._tmp_b = 0

        if self._next_layer is not None:
            self._next_layer.update()

    def reset_pass(self):
        self._tmp_w = 0
        self._tmp_b = 0
        if self._next_layer is not None:
            self._next_layer.reset_pass()


class InputLayer:
    def __init__(self, size):
        self._size = size
        self._next_layer = None

    def set_next_layer(self, layer):
        self._next_layer = layer

    def get_size(self):
        return self._size

    def feedforward(self, input_data):
        return self._next_layer.feedforward(input_data)

    def update(self, data, learning_rate):
        for input_data, results in data:
            self._next_layer.forward_back_pass(input_data, results, learning_rate / len(data))
        self._next_layer.update()

    def get_delta(self, data):
        for input_data, results in data:
            delta = self._next_layer.forward_back_pass(input_data, results, 1) / len(data)
        self._next_layer.reset_pass()
        return np.dot(self._next_layer.get_weights().T, delta)
