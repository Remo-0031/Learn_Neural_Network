import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weight = 2 * np.random.random((3, 1)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weight += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weight))
        return output


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("random synaptic weights: ")
    print(neural_network.synaptic_weight)

    training_input = np.array([[0, 0, 1],
                               [1, 1, 1],
                               [1, 0, 1],
                               [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_input,training_outputs,10000)

    print("Synaptic Weights after Training: ")
    print(neural_network.synaptic_weight)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New Situation: input data = ", A, B, C)
    print("output Data: ")
    print(neural_network.think(np.array([A, B, C])))