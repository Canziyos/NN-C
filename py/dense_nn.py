from dense_layer import DenseLayer

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

    def summary(self):
        print("\nDenseNN Summary:")
        print(f"  Total layers: {self.n_layers}")
        for i, layer in enumerate(self.layers):
            n_neurons = len(layer.neurons)
            activation_name = layer.activation.__name__
            input_size = layer.input_s
            print(f"  Layer {i+1}: {n_neurons} neurons, activation={activation_name}, input_size={input_size}")
