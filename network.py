import numpy as np


def sigmoid(ar):
    return 1 / (1 + np.exp(-ar))


class Network:
    def __init__(self, layer_sizes):
        self._n_layers = len(layer_sizes)
        self._layers = []
        self._input_layer = InputLayer(size)
        self._layers.append(self._input_layer)
        for size in layer_sizes[1:]:
            self._layers.append(Layer(size))

        last_layer = None
        for layer in self._layers:
            if last_layer is not None:
                last_layer.set_next_layer(layer)
                layer.set_previous_layer(last_layer)
            last_layer = layer

        self._initalize_layers()

    def _initialize_layers(self):
        for layer in self._layers[1:]:
            layer.set_random_params()


class Layer:
    def __init__(self, size, activation_function=sigmoid):
        self._size = size
        self._activation_function = activation_function
        self._weights = None
        self._biases = np.random.randn(size, 1)
        self._next_layer = None
        self._previous_layer = None

    def set_weights(self, weights):
        self._weights = weights

    def set_biases(self, biases):
        self._biases = biases

    def set_next_layer(self, layer):
        self._next_layer = layer

    def set_previous_layer(self, layer):
        self._previous_layer = layer

    def get_size(self):
        return self._size

    def set_random_params(self):
        biases = np.random.randn(self.get_size(), 1)
        weights = np.random.randn(self.get_size(), self._previous_layer.get_size())
        self.set_biases(biases)
        self.set_weights(weights)

    def feedforward(self, input_data):
        output = self._get_output(input_data)
        if self._next_layer is None:
            return output
        else:
            return self._next_layer.feedforward(output)

    def _get_output(self, input_data):
        return self._activation_function(self._get_z(input_data))

    def _get_z(self, input_data):
        return np.dot(self._weights, input_data) + self._biases


class InputLayer:
    def __init__(self, size):
        self._size = size
        self._next_layer = None

    def set_next_layer(self, layer):
        self._next_layer = layer

    def feedforward(self, input_data):
        return self._next_layer.feedforward(input_data)
