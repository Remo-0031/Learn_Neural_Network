import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)

'''
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights,biases):
    neuron_output = 0
    for n_input,weights in zip(inputs,neuron_weights):
        neuron_output += n_input*weights
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)
'''
