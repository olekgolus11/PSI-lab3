from StudentAI import StudentAI
import numpy as np

# studentAI = StudentAI(1)
# studentAI.add_custom_layer(np.matrix('0.1'))
# studentAI.add_custom_layer(np.matrix('0.3'))
# input = np.matrix('0.5')
# expected_value = np.matrix('0.1')
# alfa = 0.01
#
# studentAI.train(input, expected_value, 1, False)


studentAI = StudentAI(1)

studentAI.add_custom_layer(np.matrix('0.1; -0.1; 0.3'))
studentAI.add_custom_layer(np.matrix('0.7 0.9 -0.4'))

input_values = np.matrix(0.5)
expected_values = np.matrix(0.1)

alfa = 0.01

studentAI.train(input_values, expected_values, 1, alfa, True)

# studentAI.train(input_values, expected_values, 1, alfa, with_activation=True)
