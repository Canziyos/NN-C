#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "Model.h"
#include "math_utils.h"

void forward_propagation(NeuralNetwork *nn, double *input) {
    // Initialize activations for the input layer
    for (int i = 0; i < nn->neurons_per_layer[0]; i++) {
        nn->activations[0][i] = input[i];  // Copy input to activations
        printf("Copied input[%d] = %.3f to activations[%d]\n", i, input[i], i);  // Debug print
    }

    // Forward propagate through each layer
    for (int layer = 0; layer < nn->num_layers - 1; layer++) {
        matrix_multiply(nn->weights[layer], nn->activations[layer], nn->activations[layer + 1], 
                        nn->neurons_per_layer[layer + 1], nn->neurons_per_layer[layer]);

        // Apply sigmoid activation to each neuron in the next layer
        for (int j = 0; j < nn->neurons_per_layer[layer + 1]; j++) {
            nn->activations[layer + 1][j] = sigmoid(nn->activations[layer + 1][j]);
        }
    }
}
