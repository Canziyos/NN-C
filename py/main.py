from activations import sigmoid, relu
from neuron import Neuron
from dense_layer import DenseLayer

# Test a single neuron.
input_size = 3
n = Neuron(input_size)   # A neuron that takes 3 inputs.
print("Neuron Weights:", n.weights)
print("Neuron Bias:", n.bias)
print("Neuron Output (raw):", n.pass_forward([2, 2, 2]))

# Test a dense layer with 3 inputs and 4 neurons, using sigmoid activation.
layer = DenseLayer(input_s=3, n_neurons=4, activation=sigmoid)
print("\nDense Layer (sigmoid) outputs:", layer.forward([2, 2, 2]))

# Test the same dense layer setup but with ReLU activation.
layer_relu = DenseLayer(input_s=3, n_neurons=4, activation=relu)
print("\nDense Layer (ReLU) outputs:", layer_relu.forward([2, 2, 2]))
