#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "Model.h"

void initialize_network(NeuralNetwork *nn, int num_layers, int *neurons_per_layer) {
    // Step 1: Allocate memory for neurons per layer and copy values.
    nn->num_layers = num_layers;
    nn->neurons_per_layer = malloc(num_layers * sizeof(int));
    if (nn->neurons_per_layer == NULL) {
        printf("Memory allocation failed for neurons_per_layer.\n");
        exit(1);
    }
    for (int i = 0; i < num_layers; i++) {
        nn->neurons_per_layer[i] = neurons_per_layer[i];
    }

    // Step 2: Allocate memory for weights, biases, and activations for each layer
    nn->weights = malloc((num_layers - 1) * sizeof(double **));
    nn->biases = malloc((num_layers - 1) * sizeof(double *));
    nn->activations = malloc(num_layers * sizeof(double *));
    if (nn->weights == NULL || nn->biases == NULL || nn->activations == NULL) {
        printf("Memory allocation failed for weights, biases, or activations.\n");
        exit(1);
    }

    srand((unsigned int)time(NULL));  // Seed random generator

    for (int layer = 0; layer < num_layers - 1; layer++) {
        // Allocate memory for weight matrix: [neurons in current layer][neurons in next layer]
        nn->weights[layer] = malloc(neurons_per_layer[layer] * sizeof(double *));
        if (nn->weights[layer] == NULL) {
            printf("Memory allocation failed for weights[%d].\n", layer);
            exit(1);
        }
        for (int i = 0; i < neurons_per_layer[layer]; i++) {
            nn->weights[layer][i] = malloc(neurons_per_layer[layer + 1] * sizeof(double));
            if (nn->weights[layer][i] == NULL) {
                printf("Memory allocation failed for weights[%d][%d].\n", layer, i);
                exit(1);
            }
        }

        // Allocate memory for biases: [neurons in the next layer]
        nn->biases[layer] = malloc(neurons_per_layer[layer + 1] * sizeof(double));
        if (nn->biases[layer] == NULL) {
            printf("Memory allocation failed for biases[%d].\n", layer);
            exit(1);
        }

        // Initialize weights and biases with random values between -0.5 and 0.5
        for (int i = 0; i < neurons_per_layer[layer]; i++) {
            for (int j = 0; j < neurons_per_layer[layer + 1]; j++) {
                nn->weights[layer][i][j] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
        for (int j = 0; j < neurons_per_layer[layer + 1]; j++) {
            nn->biases[layer][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    // Allocate and initialize activations for each layer.
    for (int layer = 0; layer < num_layers; layer++) {
        nn->activations[layer] = malloc(neurons_per_layer[layer] * sizeof(double));
        if (nn->activations[layer] == NULL) {
            printf("Memory allocation failed for activations[%d].\n", layer);
            exit(1);
        }
        // Initialize activations to zero.
        for (int i = 0; i < neurons_per_layer[layer]; i++) {
            nn->activations[layer][i] = 0.0;
        }
    }
}

void free_network(NeuralNetwork *nn) {
    // Free weights.
    for (int layer = 0; layer < nn->num_layers - 1; layer++) {
        for (int i = 0; i < nn->neurons_per_layer[layer]; i++) {
            if (nn->weights[layer][i] != NULL) free(nn->weights[layer][i]);
        }
        if (nn->weights[layer] != NULL) free(nn->weights[layer]);
    }
    if (nn->weights != NULL) free(nn->weights);

    // Free biases.
    for (int layer = 0; layer < nn->num_layers - 1; layer++) {
        if (nn->biases[layer] != NULL) free(nn->biases[layer]);
    }
    if (nn->biases != NULL) free(nn->biases);

    // Free activations.
    for (int layer = 0; layer < nn->num_layers; layer++) {
        if (nn->activations[layer] != NULL) free(nn->activations[layer]);
    }
    if (nn->activations != NULL) free(nn->activations);

    // Free neurons per layer.
    if (nn->neurons_per_layer != NULL) free(nn->neurons_per_layer);

    printf("Network memory successfully freed.\n");
}
