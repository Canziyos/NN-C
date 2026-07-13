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
        return 1;
    }

    for (size_t sample = 0; sample < 4U; ++sample) {
        float output[1];
        status = nn_predict(&model, inputs[sample], 2U, output, 1U);
        if (status != NN_STATUS_OK) {
            return 1;
        }
        printf(
            "%.0f XOR %.0f -> %.6f (%d)\n",
            (double)inputs[sample][0],
            (double)inputs[sample][1],
            (double)output[0],
            output[0] >= 0.5F
        );
    }
    return 0;
}
