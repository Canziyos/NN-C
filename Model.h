#ifndef MODEL_H
#define MODEL_H

// Neural network structure.
typedef struct {
    int num_layers;              // Number of layers.
    int *neurons_per_layer;      // Array holding the number of neurons per layer.

    double ***weights;           // 3D array: weights[layer][from_neuron][to_neuron]
    double **biases;             // 2D array: biases[layer][neuron].

    double **activations;        // 2D array to store activations at each layer.
} NeuralNetwork;

void forward_propagation(NeuralNetwork *nn, double *input);

#endif
