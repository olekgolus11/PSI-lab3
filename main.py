from StudentAI import StudentAI
import numpy as np

studentAI = StudentAI(1)

studentAI.add_layer(1, [0.1, 0.1])
studentAI.add_layer(1, [0.3, 0.3])

input_values = np.matrix(0.5)
expected_values = np.matrix(0.1)
alfa = 0.01

# studentAI.train_layer(input_values, expected_values, 1, alfa)
studentAI.train(input_values, expected_values, 1, alfa)

print(studentAI.weights_matrix_list)