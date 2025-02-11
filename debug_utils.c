#include <stdio.h>
#include "debug_utils.h"

void print_activations(double *activations, int num_neurons, const char *layer_name) {
    printf("\nActivations of %s:\n", layer_name);
    for (int i = 0; i < num_neurons; i++) {
        printf("Activation[%d] = %.3f\n", i, activations[i]);
    }
}
