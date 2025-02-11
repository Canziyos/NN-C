#include <stdio.h>
#include "Model.h"
#include "init.h"
#include "debug_utils.h"

int main() {
    int layers = 3;
    int neurons_per_layer[] = {2, 4, 1};  // 2 inputs, 4 hidden neurons, 1 output neuron.

    NeuralNetwork nn;
    initialize_network(&nn, layers, neurons_per_layer);

    // Define multiple test inputs
    double test_inputs[][2] = {
        {0.5, -0.3},  // Original test case
        {1.0, 0.8},   // Test case with positive inputs
        {-0.2, 0.5},  // Mixed signs
        {0.0, 0.0},   // Edge case: all zero inputs
        {-1.0, -0.8}  // Negative inputs
    };

    int num_tests = sizeof(test_inputs) / sizeof(test_inputs[0]);

    // Loop through each test case
    for (int test = 0; test < num_tests; test++) {
        printf("\n=== Test Case %d ===\n", test + 1);
        forward_propagation(&nn, test_inputs[test]);

        // Print activations for all layers
        print_activations(nn.activations[0], neurons_per_layer[0], "the input layer");
        print_activations(nn.activations[1], neurons_per_layer[1], "the hidden layer");
        print_activations(nn.activations[2], neurons_per_layer[2], "the output layer");
    }

    // Free allocated memory
    free_network(&nn);

    return 0;
}
