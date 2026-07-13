#include "Model.h"
#include "math_utils.h"

void forward_propagation(NeuralNetwork *nn, double *input)
{
    for (int i = 0; i < nn->neurons_per_layer[0]; i++) {
        nn->activations[0][i] = input[i];
    }

    for (int layer = 0; layer < nn->num_layers - 1; layer++) {
        matrix_multiply(
            nn->weights[layer],
            nn->activations[layer],
            nn->activations[layer + 1],
            nn->neurons_per_layer[layer + 1],
            nn->neurons_per_layer[layer]
        );

        for (int neuron = 0; neuron < nn->neurons_per_layer[layer + 1]; neuron++) {
            const double weighted_sum =
                nn->activations[layer + 1][neuron] + nn->biases[layer][neuron];

            nn->activations[layer + 1][neuron] = sigmoid(weighted_sum);
        }
    }
}
