from activations import sigmoid, sigmoid_prime
from neuron import Neuron

class DenseLayer:
    def __init__(self, input_s=None, n_neurons=None, activation=sigmoid, activation_prime = sigmoid_prime):
        self.input_s = input_s
        self.n_neurons = n_neurons
        self.activation = activation
        self.activation_prime = activation_prime

        # Create a list of Neuron objects.
        self.neurons = [Neuron(input_s=self.input_s) 
                        for _ in range(self.n_neurons)]
        
        # for forward-pass values.
        self.z_values = []
        self.a_values = []

    def forward(self, inputs):
        # Compute raw outputs from all neurons.
        self.z_values = [neuron.forward(inputs) for neuron in self.neurons]

        # Apply activation to each raw output.
        self.a_values = [self.activation(z) for z in self.z_values]

        return self.a_values

