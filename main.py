import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


training_input = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weight = 2 * np.random.random((3, 1)) - 1

print('Random Starting synaptic weight: ')
print(synaptic_weight)

for iteration in range(100000):
    input_layer = training_input

    outputs = sigmoid(np.dot(input_layer, synaptic_weight))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weight += np.dot(input_layer.T, adjustments)
print("Synaptic Weights after training: ", synaptic_weight)

print('Outputs after training: ')
print(outputs)
