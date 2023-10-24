import numpy as np

from StudentAI import StudentAI

studentAI = StudentAI(3)
studentAI.add_layer(9, [0, 1])
studentAI.add_layer(4, [0, 1])

train_data = np.genfromtxt("train.txt")
number_of_train_inputs = train_data.shape[0]
train_data_input = train_data[:, :-1]
expected_value = train_data[:, -1]

train_data_expected = np.zeros((train_data.shape[0], 4))
for i in range(number_of_train_inputs):
    train_data_expected[i, int(expected_value[i] - 1)] = 1

for j in range(50):
    for i in range(number_of_train_inputs):
        input = np.matrix(train_data_input[i, :])
        expected = np.matrix(train_data_expected[i, :])
        studentAI.train(input.T, expected.T, 1, 0.01)

test_data = np.genfromtxt("test.txt")
number_of_test_inputs = test_data.shape[0]
test_data_input = test_data[:, :-1]
test_data_expected = test_data[:, -1]

correct = 0
for i in range(number_of_test_inputs):
    input = np.matrix(test_data_input[i, :])
    expected = test_data_expected[i]
    result = studentAI.predict(input.T)
    result = np.argmax(result) + 1
    if result == expected:
        correct += 1

print(f"Accuracy: {(correct / number_of_test_inputs) * 100}%, {correct} / {number_of_test_inputs}")
