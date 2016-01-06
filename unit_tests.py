import unittest
import numpy as np
import network as nw


class TestActivationFunctions(unittest.TestCase):
    def test_sigmoid(self):
        test_data = np.array([[0], [-1], [0.5]])
        test_result = np.array([[0.5], [1 / (1 + np.exp(1))], [1 / (1 + np.exp(-0.5))]])

        np.testing.assert_almost_equal(nw.sigmoid_fun(test_data), test_result)

    def test_sigmoid_derivative(self):
        test_data = np.array([[0], [-1], [0.5]])
        test_result = np.array([[0.25], [np.exp(1) / ((1 + np.exp(1)) ** 2)], [np.exp(0.5) / ((1 + np.exp(0.5)) ** 2)]])

        np.testing.assert_almost_equal(nw.sigmoid_derivative(test_data), test_result)


class TestCostFunction(unittest.TestCase):
    def test_squared_cost(self):
        test_input = np.array([1, 0.5, -3, 2, 5])
        test_results = np.array([0.5, 0, -2, -1, 5])
        cost = 5.25

        self.assertEqual(nw.squared_fun(test_input, test_results), cost)

    def test_squared_cost_derivative(self):
        test_input = np.array([1, 0.5, -3, 2, 5])
        test_results = np.array([0.5, 0, -2, -1, 5])
        derivative = np.array([0.5, 0.5, -1, 3, 0])

        np.testing.assert_equal(nw.squared_derivative(test_input, test_results), derivative)


class TestLayers(unittest.TestCase):
    def test_z(self):
        unit_layer = nw.Layer(1)
        small_layer = nw.Layer(4)
        big_layer = nw.Layer(1000)

        unit_weights = np.array([[0.5]])
        small_weights = np.array([[0.5, -0.5, 0.5, 0.5], [0, 0, 0, 0], [4, -0.5, 0, 0], [0, 0, 0, 0]])
        big_weights = np.ones([1000, 500])

        unit_bias = np.array([[-1]])
        small_bias = np.array([[1], [2], [-3], [4]])
        big_bias = np.ones([1000, 1])

        unit_layer.set_weights(unit_weights)
        small_layer.set_weights(small_weights)
        big_layer.set_weights(big_weights)

        unit_layer.set_biases(unit_bias)
        small_layer.set_biases(small_bias)
        big_layer.set_biases(big_bias)

        unit_input = np.array([[1.5]])
        small_input = np.array([[1], [2], [3], [4]])
        big_input = np.zeros([500, 1])

        unit_output = np.array([[-0.25]])
        small_output = np.array([[4], [2], [0], [4]])
        big_output = np.ones([1000, 1])

        np.testing.assert_almost_equal(unit_layer._get_z(unit_input), unit_output)
        np.testing.assert_almost_equal(small_layer._get_z(small_input), small_output)
        np.testing.assert_almost_equal(big_layer._get_z(big_input), big_output)

    def test_output(self):
        unit_layer = nw.Layer(1)
        small_layer = nw.Layer(4)
        big_layer = nw.Layer(1000)

        unit_weights = np.array([[0.5]])
        small_weights = np.array([[0.5, -0.5, 0.5, 0.5], [0, 0, 0, 0], [4, -0.5, 0, 0], [0, 0, 0, 0]])
        big_weights = np.ones([1000, 500])

        unit_bias = np.array([[-1]])
        small_bias = np.array([[1], [2], [-3], [4]])
        big_bias = np.ones([1000, 1])

        unit_layer.set_weights(unit_weights)
        small_layer.set_weights(small_weights)
        big_layer.set_weights(big_weights)

        unit_layer.set_biases(unit_bias)
        small_layer.set_biases(small_bias)
        big_layer.set_biases(big_bias)

        unit_input = np.array([[1.5]])
        small_input = np.array([[1], [2], [3], [4]])
        big_input = np.zeros([500, 1])

        unit_output = np.array([[1 / (1 + np.exp(0.25))]])
        small_output = np.array([[1 / (1 + np.exp(-4))], [1 / (1 + np.exp(-2))], [0.5], [1 / (1 + np.exp(-4))]])
        big_output = 1 / (1 + np.exp(-1)) * np.ones([1000, 1])

        np.testing.assert_almost_equal(unit_layer._get_output(unit_input), unit_output)
        np.testing.assert_almost_equal(small_layer._get_output(small_input), small_output)
        np.testing.assert_almost_equal(big_layer._get_output(big_input), big_output)

    def test_feed_forward(self):
        input_layer = nw.InputLayer(4)
        small_layer = nw.Layer(4)
        output_layer = nw.Layer(4)

        input_layer.set_next_layer(small_layer)
        small_layer.set_next_layer(output_layer)

        small_weights = np.array([[0.5, -0.5, 0.5, 0.5], [0, 0, 0, 0], [4, -0.5, 0, 0], [0, 0, 0, 0]])
        small_bias = np.array([[1], [2], [-3], [4]])

        small_layer.set_weights(small_weights)
        small_layer.set_biases(small_bias)

        output_weights = np.array([[0, 0, 2, 0]])
        output_bias = np.array([[-1]])

        output_layer.set_weights(output_weights)
        output_layer.set_biases(output_bias)

        small_input = np.array([[1], [2], [3], [4]])
        small_output = np.array([[0.5]])

        np.testing.assert_almost_equal(input_layer.feedforward(small_input), small_output)


class TestNetwork(unittest.TestCase):
    def test_initialization(self):
        small_sizes = [2, 1]
        big_sizes = [5, 10, 1, 30, 35]

        small_network = nw.Network(small_sizes)
        big_network = nw.Network(big_sizes)

        for layer, size in enumerate(small_sizes):
            self.assertEqual(small_network._layers[layer]._size, size)

        for layer, size in enumerate(big_sizes):
            self.assertEqual(big_network._layers[layer]._size, size)


if __name__ == '__main__':
    unittest.main()
