#include "nnc/nn.h"

#include <stdio.h>

int main(void)
{
    static const float weights[] = {
        0.8F, -0.2F,
        -0.4F, 0.9F
    };
    static const float biases[] = {0.1F, -0.1F};
    static const nn_dense_layer_t layers[] = {
        {
            .input_count = 2,
            .output_count = 2,
            .weights = weights,
            .biases = biases,
            .activation = NN_ACTIVATION_SIGMOID
        }
    };

    float workspace[4];
    nn_model_t model;
    nn_status_t status = nn_model_init(&model, layers, 1, workspace, 4);
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "init failed: %s\n", nn_status_string(status));
        return 1;
    }

    const float input[] = {0.6F, -0.1F};
    float output[2];
    status = nn_predict(&model, input, 2, output, 2);
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "prediction failed: %s\n", nn_status_string(status));
        return 1;
    }

    printf("output: %.6f %.6f\n", (double)output[0], (double)output[1]);
    return 0;
}
