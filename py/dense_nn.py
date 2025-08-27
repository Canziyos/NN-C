from dense_layer import DenseLayer
from utils import output_gradient, hidden_layer_delta, output_layer_delta
class DenseNN:
    def __init__(self, layers):
        # only DenseLayer objects are passed.
        for l in layers:
            if not isinstance(l, DenseLayer):
                raise TypeError("DenseNN only accepts DenseLayer objects.")

        self.layers = layers
        self.n_layers = len(layers)
        self.first_layer = layers[0]
        self.last_layer = layers[-1]
        self.hidden_layers = layers[1:-1]
        self.hidden_sizes = [len(l.neurons) for l in self.hidden_layers]

        self.layer_inputs = []
        self.layer_outputs = []

    def forward(self, inputs):
        self.layer_inputs = []
        self.layer_outputs = []

        x = inputs
        for layer in self.layers:
            self.layer_inputs.append(x)
            x = layer.forward(x)
            self.layer_outputs.append(x)
        return x

    def backward(self, y_true, y_pred):
        deltas = [None] * self.n_layers
        grads = []

        # 1. Output layer delta
        last = self.last_layer
        deltas[-1] = output_layer_delta(y_true, y_pred,
                                        last.z_values, last.activation_prime)

        # 2. Hidden layers deltas (backward loop).
        for l in reversed(range(self.n_layers - 1)):  # skip output.
            next_layer = self.layers[l+1]
            deltas[l] = hidden_layer_delta(
                self.layers[l].z_values,
                self.layers[l].activation_prime,
                [n.weights for n in next_layer.neurons],
                deltas[l+1]
            )

        # 3. Gradients for each layer
        for l, layer in enumerate(self.layers):
            prev_a = self.layer_inputs[l]   # inputs to this layer
            w_grads, b_grads = output_gradient(deltas[l], prev_a)
            grads.append({"w": w_grads, "b": b_grads})

        return grads



    def summary(self):
        print("\nDenseNN Summary:")
        print(f"  Total layers: {self.n_layers}")
        for i, layer in enumerate(self.layers):
            n_neurons = len(layer.neurons)
            activation_name = layer.activation.__name__
            input_size = layer.input_s
            print(f"  Layer {i+1}: {n_neurons} neurons, activation={activation_name}, input_size={input_size}")
