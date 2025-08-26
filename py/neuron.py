import random

class Neuron:
    def __init__(self, input_s=None):

        self.input_s = input_s

        # Generate one weight for each input.
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(self.input_s)]

        # Generate random bias.
        self.bias = random.uniform(-0.5, 0.5)


    def pass_forward(self, inputs):
        if len(inputs) != self.input_s:
            raise ValueError("Input length does not match number of weights!")

        # Weighted sum.
        z = sum(inputs[i] * self.weights[i] for i in range(self.input_s)) + self.bias

        return z

