from activations import sigmoid, relu, tanh, sigmoid_prime
from neuron import Neuron
from dense_layer import DenseLayer
from dense_nn import DenseNN
from utils import MSE, output_layer_delta, output_gradient, hidden_layer_delta

# ---------------------------------
# Basic tests
# ---------------------------------

# Test a single neuron.
n = Neuron(input_s=3)
print("Neuron Weights:", n.weights)
print("Neuron Bias:", n.bias)
print("Neuron Output (raw):", n.forward([2, 2, 2]))

# Test a dense layer with 3 inputs and 4 neurons, using sigmoid activation.
layer = DenseLayer(input_s=3, n_neurons=4, activation=sigmoid)
print("\nDense Layer (sigmoid) outputs:", layer.forward([2, 2, 2]))

# Test the same dense layer setup but with ReLU activation.
layer_relu = DenseLayer(input_s=3, n_neurons=4, activation=relu)
print("\nDense Layer (ReLU) outputs:", layer_relu.forward([2, 2, 2]))

# ---------------------------------
# Full DenseNN configuration
# ---------------------------------

net = DenseNN([
    DenseLayer(input_s=2, n_neurons=2, activation=sigmoid),
    DenseLayer(input_s=2, n_neurons=2, activation=sigmoid),
    DenseLayer(input_s=2, n_neurons=1, activation=sigmoid) 
])

net.summary()

# ---------------------------------
# Delta tests (XOR sample).
# ---------------------------------

x = [0, 1]
y_true = [1]

# Forward pass.
y_pred = net.forward(x)
print("\nForward pass prediction:", y_pred)

# Output layer delta.
last_layer = net.last_layer
delta_out = output_layer_delta(y_true, y_pred, last_layer.z_values, sigmoid_prime)
print("Delta (output layer):", delta_out)

# Output layer gradients.
prev_a = net.layer_outputs[-2]  # activations from 2nd hidden layer.
w_grads, b_grads = output_gradient(delta_out, prev_a)
print("Output weight gradients:", w_grads)
print("Output bias gradients:", b_grads)

# Hidden layer (layer before output) delta.
hidden_layer = net.hidden_layers[-1]
hidden_deltas = hidden_layer_delta(hidden_layer.z_values, sigmoid_prime,
                                   [n.weights for n in last_layer.neurons], delta_out)
print("Hidden layer deltas:", hidden_deltas)

# First hidden layer delta.
first_hidden = net.first_layer
second_hidden = net.hidden_layers[0]
first_deltas = hidden_layer_delta(first_hidden.z_values, sigmoid_prime,
                                  [n.weights for n in second_hidden.neurons], hidden_deltas)
print("First hidden layer deltas:", first_deltas)
