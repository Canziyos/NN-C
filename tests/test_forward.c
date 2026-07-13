#include <math.h>
#include <stdio.h>

#include "Model.h"
#include "init.h"

int main(void)
{
    int layer_sizes[] = {2, 2, 1};
    double input[] = {0.6, -0.1};

    const double expected = 0.622507;
    const double tolerance = 1e-6;

    NeuralNetwork nn;
    initialize_network(&nn, 3, layer_sizes);

    nn.weights[0][0][0] = 0.5;
    nn.weights[0][0][1] = -0.4;
    nn.weights[0][1][0] = 0.25;
    nn.weights[0][1][1] = 0.8;

    nn.biases[0][0] = 0.1;
    nn.biases[0][1] = -0.2;

    nn.weights[1][0][0] = 1.2;
    nn.weights[1][1][0] = -0.7;
    nn.biases[1][0] = 0.05;

    forward_propagation(&nn, input);

    const double actual = nn.activations[2][0];
    const double error = fabs(actual - expected);

    free_network(&nn);

    if (error > tolerance) {
        fprintf(
            stderr,
            "FAIL: expected %.6f, got %.6f, error %.6f\n",
            expected,
            actual,
            error
        );
        return 1;
    }

    printf("PASS: deterministic forward propagation\n");
    return 0;
}
