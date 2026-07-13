#include "nnc/nn.h"
#include "trained_xor_model.h"

#include <stdio.h>

int main(void)
{
    static const float inputs[][2] = {
        {0.0F, 0.0F},
        {0.0F, 1.0F},
        {1.0F, 0.0F},
        {1.0F, 1.0F}
    };
    static const int expected[] = {0, 1, 1, 0};

    float workspace[TRAINED_XOR_WORKSPACE_FLOATS];
    nn_model_t model;
    nn_status_t status = nn_model_init(
        &model,
        trained_xor_layers,
        TRAINED_XOR_LAYER_COUNT,
        workspace,
        TRAINED_XOR_WORKSPACE_FLOATS
    );
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "init failed: %s\n", nn_status_string(status));
        return 1;
    }

    for (size_t sample = 0; sample < 4U; ++sample) {
        float output[1];
        status = nn_predict(&model, inputs[sample], 2U, output, 1U);
        const int predicted = output[0] >= 0.5F;
        if (status != NN_STATUS_OK || predicted != expected[sample]) {
            fprintf(
                stderr,
                "sample %zu: expected %d, got %.6f\n",
                sample,
                expected[sample],
                (double)output[0]
            );
            return 1;
        }
    }

    printf("PASS: host-trained XOR model matches all labels\n");
    return 0;
}
