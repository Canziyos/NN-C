#include "generated_model.h"
#include "nnc/nn.h"

#include <stdio.h>

int main(void)
{
    float workspace[EXAMPLE_MODEL_WORKSPACE_FLOATS];
    nn_model_t model;

    nn_status_t status = nn_model_init(
        &model,
        example_model_layers,
        EXAMPLE_MODEL_LAYER_COUNT,
        workspace,
        EXAMPLE_MODEL_WORKSPACE_FLOATS
    );
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "init failed: %s\n", nn_status_string(status));
        return 1;
    }

    const float input[] = {0.6F, -0.1F};
    float output[1];
    status = nn_predict(&model, input, 2U, output, 1U);
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "prediction failed: %s\n", nn_status_string(status));
        return 1;
    }

    printf("generated model output: %.6f\n", (double)output[0]);
    return 0;
}
