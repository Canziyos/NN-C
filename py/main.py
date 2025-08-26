from activations import sigmoid, relu, tanh
from neuron import Neuron
from dense_layer import DenseLayer
from dense_nn import DenseNN

# Test a single neuron.
input_size = 3
n = Neuron(input_size)
print("Neuron Weights:", n.weights)
print("Neuron Bias:", n.bias)
print("Neuron Output (raw):", n.forward([2, 2, 2]))

# Test a dense layer with 3 inputs and 4 neurons, using sigmoid activation.
layer = DenseLayer(input_s=3, n_neurons=4, activation=sigmoid)
print("\nDense Layer (sigmoid) outputs:", layer.forward([2, 2, 2]))

# the same dense layer setup but with ReLU activation.
layer_relu = DenseLayer(input_s=3, n_neurons=4, activation=relu)
print("\nDense Layer (ReLU) outputs:", layer_relu.forward([2, 2, 2]))

# ---------------------------------
# Full DenseNN configurations.
# ----------------------------------

net = DenseNN([
    DenseLayer(input_s=3, n_neurons=5, activation="relu"),
    DenseLayer(input_s=5, n_neurons=2, activation="tanh"),
    DenseLayer(input_s=2, n_neurons=1, activation="sigmoid")
])

net.summary()
