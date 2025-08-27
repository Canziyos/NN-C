from activations import sigmoid, sigmoid_prime
from neuron import Neuron
import random, math

class DenseLayer:
    def __init__(self, input_s=None, n_neurons=None,
                 init="xavier",
                 activation=sigmoid, activation_prime=sigmoid_prime):
        self.input_s = input_s
        self.n_neurons = n_neurons
        self.activation = activation
        self.activation_prime = activation_prime
        self.z_values = []
        self.a_values = []

        init = init.lower()
        self.neurons = []

        for _ in range(self.n_neurons):
            # Init weights per neuron according to scheme.
            if init == "xavier":
                limit = math.sqrt(6 / (input_s + n_neurons))
                weights = [random.uniform(-limit, limit) for _ in range(input_s)]
            elif init == "he":
                limit = math.sqrt(6 / input_s)
                weights = [random.uniform(-limit, limit) for _ in range(input_s)]
            else:  # fallback
                weights = [random.uniform(-0.5, 0.5) for _ in range(input_s)]

            bias = 0.0
            neuron = Neuron(input_s=input_s)
            neuron.weights = weights
            neuron.bias = bias
            self.neurons.append(neuron)

    def forward(self, inputs):
        # Compute raw outputs from all neurons
        self.z_values = [neuron.forward(inputs) for neuron in self.neurons]

        # Apply activation
        if self.activation.__name__ == "softmax":
            self.a_values = self.activation(self.z_values)   # vector.
        else:
            self.a_values = [self.activation(z) for z in self.z_values]  # elementwise.

        return self.a_values
