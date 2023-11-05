from StudentAI import StudentAI
import numpy as np

layer_1 = np.matrix('0.1 0.1 -0.3; 0.1 0.2 0.0; 0.0 0.7 0.1; 0.2 0.4 0.0; -0.3 0.5 0.1')
layer_2 = np.matrix('0.7 0.9 -0.4 0.8 0.1; 0.8 0.5 0.3 0.1 0.0; -0.3 0.9 0.3 0.1 -0.2')
input_values = np.matrix('0.5 0.1 0.2 0.8; 0.75 0.3 0.1 0.9; 0.1 0.7 0.6 0.2')

studentAI = StudentAI(3)
studentAI.add_custom_layer(layer_1)
studentAI.add_custom_layer(layer_2)

print(studentAI.predict(input_values))