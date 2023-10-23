from StudentAI import StudentAI
import numpy as np

studentAI = StudentAI(784)
studentAI.add_layer(40, [-0.1, 0.1])
studentAI.add_layer(10, [-0.1, 0.1])

# Load train-images.idx3-ubyte
with open('train-images.idx3-ubyte', 'rb') as f:
    train_images = f.read()

# Load train-labels.idx1-ubyte
with open('train-labels.idx1-ubyte', 'rb') as f:
    train_labels = f.read()

# Load t10k-images.idx3-ubyte
with open('t10k-images.idx3-ubyte', 'rb') as f:
    test_images = f.read()

# Load t10k-labels.idx1-ubyte
with open('t10k-labels.idx1-ubyte', 'rb') as f:
    test_labels = f.read()

# Convert bytes to numpy arrays
train_images = np.frombuffer(train_images, dtype=np.uint8).copy()
train_labels = np.frombuffer(train_labels, dtype=np.uint8).copy()
test_images = np.frombuffer(test_images, dtype=np.uint8).copy()
test_labels = np.frombuffer(test_labels, dtype=np.uint8).copy()

# Remove magic numbers
train_images = train_images[16:]
train_labels = train_labels[8:]
test_images = test_images[16:]
test_labels = test_labels[8:]

# Reshape images
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
# why reshaping? https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

# Normalize images
train_images = train_images / 255
test_images = test_images / 255

# train
for i in range(60000):
    input_values = np.transpose(np.matrix(train_images[i]))
    expected_values = np.zeros((1, 10))
    expected_values[0, train_labels[i]] = 1
    expected_values = np.transpose(np.matrix(expected_values))
    studentAI.train(input_values, expected_values, 1, 0.03, True)

# test
correct = 0
for i in range(10000):
    input_values = np.transpose(np.matrix(test_images[i]))
    expected_values = np.zeros((1, 10))
    expected_values[0, test_labels[i]] = 1
    expected_values = np.transpose(np.matrix(expected_values))
    result = studentAI.predict_with_activation(input_values)
    if np.argmax(result) == np.argmax(expected_values):
        correct += 1

print(f"Correct: {correct} / 10000")