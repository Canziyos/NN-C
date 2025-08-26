from activations import sigmoid
from neuron import Neuron

class DenseLayer:
    def __init__(self, input_s=None, n_neurons=None, activation=sigmoid):
        self.input_s = input_s
        self.n_neurons = n_neurons
        self.activation = activation

        # Create a list of Neuron objects.
        self.neurons = [Neuron(input_s=self.input_s) 
                        for _ in range(self.n_neurons)]
        

    def forward(self, inputs):
        # Compute raw outputs from all neurons.
        raw_outputs = [neuron.pass_forward(inputs) for neuron in self.neurons]

        # Apply activation to each raw output.
        return [self.activation(z) for z in raw_outputs]

