import numpy as np


class StudentAI:
    weights_matrix_list = None

    def __init__(self, number_of_entry_values):
        self.number_of_entry_values = number_of_entry_values

    def neuron(self, inputs, weights, bias):
        return np.dot(inputs, weights) + bias

    def neural_network(self, input_vector, weights_matrix):
        return np.matmul(weights_matrix, input_vector)

    def deep_neural_network(self, input_vector, weights_matrix_list):
        inputs = input_vector
        try:
            for weights_matrix in weights_matrix_list:
                inputs = self.neural_network(inputs, weights_matrix)
            return inputs
        except:
            raise Exception('Input vector dimensions doesnt match weight matrix')

    def add_layer(self, n, weight_range_values):
        min_value = weight_range_values[0]
        max_value = weight_range_values[1]

        if self.weights_matrix_list is None:
            entry_values_count = self.number_of_entry_values
            matrix_layer = np.matrix(np.random.uniform(min_value, max_value, (n, entry_values_count)))
            self.weights_matrix_list = [matrix_layer]
        else:
            entry_values_count = self.weights_matrix_list[-1].shape[0]
            matrix_layer = np.matrix(np.random.uniform(min_value, max_value, (n, entry_values_count)))
            self.weights_matrix_list.append(matrix_layer)

    def predict(self, input_values):
        try:
            return self.deep_neural_network(input_values, self.weights_matrix_list)
        except Exception as error:
            print(error)

    def load_weights(self, file_name):
        loaded_matrix = np.matrix(np.genfromtxt(file_name, delimiter=" "))
        try:
            if self.weights_matrix_list is None:
                if loaded_matrix.shape[1] != self.number_of_entry_values:
                    raise Exception('Loaded matrix is in incorrect shape')
                self.weights_matrix_list = [loaded_matrix]
            else:
                if loaded_matrix.shape[1] != self.weights_matrix_list[-1].shape[0]:
                    raise Exception('Loaded matrix is in incorrect shape')
                self.weights_matrix_list.append(loaded_matrix)
        except Exception as error:
            print(error)

    def train(self, input_values, expected_values, train_count, alpha):
        for i in range(train_count):
            for column_index in range(input_values.shape[1]):
                input_series = input_values[:, column_index]
                expected_series = expected_values[:, column_index]
                number_of_layers = len(self.weights_matrix_list)
                delta = None

                weight_delta_list = []
                for weight_matrix_index in range(number_of_layers - 1, -1, -1):
                    if weight_matrix_index == number_of_layers - 1:
                        delta = self.calculate_layer_output_delta(input_series, expected_series)
                    else:
                        delta = self.calculate_layer_delta_from_next_layer(delta, self.weights_matrix_list[
                            weight_matrix_index + 1])
                    weight_delta = self.calculate_weight_delta(input_series, delta)
                    weight_delta_list.append(weight_delta)
                    print("Layer: " + str(weight_matrix_index))
                    print("Delta: " + str(delta))
                    print("Weight delta: " + str(weight_delta))
                weight_delta_list.reverse()
                for weight_matrix_index in range(number_of_layers):
                    self.weights_matrix_list[weight_matrix_index] = self.weights_matrix_list[
                                                                        weight_matrix_index] - weight_delta_list[
                                                                        weight_matrix_index] * alpha

    def train_layer(self, input_values, expected_values, weight_matrix_index, alpha):
        pass

    def print_weights(self, weight_matrix_index):
        print("Weights: ")
        print(self.weights_matrix_list[weight_matrix_index])

    def print_output(self, input_values):
        print("Output: ")
        print(self.predict(input_values).T)

    def print_error(self, input_values, expected_values):
        print("Error: ")
        print(self.get_error_for_serie(input_values, expected_values))

    def calculate_weight_delta(self, input_values, delta):
        weight_delta = np.outer(delta, input_values)
        print("Weight delta: ")
        print(weight_delta)
        return weight_delta

    def calculate_delta(self, input_values, expected_values, weights_matrix):
        n = len(expected_values)
        output = self.neural_network(input_values, weights_matrix)
        delta = 2 * (1 / n) * (output - expected_values)
        return delta

    def calculate_layer_output_delta(self, input_values, expected_values):
        n = len(expected_values)
        output = self.predict(input_values)
        delta = 2 * (1 / n) * (output - expected_values)
        return delta

    def calculate_layer_delta_from_next_layer(self, next_layer_delta, weights_matrix):
        delta = np.outer(next_layer_delta, weights_matrix)
        return delta

    def get_error(self, input_values, expected_values):
        error = 0
        series_count = input_values.shape[1]
        for serie_index in range(series_count):
            error += self.get_error_for_serie(input_values[:, serie_index], expected_values[:, serie_index])
        return error

    def get_error_for_serie(self, input_values, expected_values):
        n = expected_values.shape[0]
        prediction = self.predict(input_values)
        error_sum = 0
        for i in range(n):
            neuron_error = ((prediction[:, i] - expected_values[i]) ** 2)
            error_sum += neuron_error
        error = error_sum / n
        return error

    def rectified_linear_unit(self, x):
        return np.maximum(x, 0)

    def rectified_linear_unit_derivative(self, x):
        return np.where(x <= 0, 0, 1)