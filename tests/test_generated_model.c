#include "generated_model.h"
#include "nnc/nn.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    float workspace[EXAMPLE_MODEL_WORKSPACE_FLOATS];
    nn_model_t model;

    const nn_status_t init_status = nn_model_init(
        &model,
        example_model_layers,
        EXAMPLE_MODEL_LAYER_COUNT,
        workspace,
        EXAMPLE_MODEL_WORKSPACE_FLOATS
    );
    if (init_status != NN_STATUS_OK) {
        fprintf(stderr, "init failed: %s\n", nn_status_string(init_status));
        return 1;
    }

    const float input[] = {0.6F, -0.1F};
    float output[1];
    const nn_status_t predict_status = nn_predict(
        &model,
        input,
        2U,
        output,
        1U
    );
    if (predict_status != NN_STATUS_OK) {
        fprintf(
            stderr,
            "prediction failed: %s\n",
            nn_status_string(predict_status)
        );
        return 1;
    }

    const float expected = 0.622507F;
    const float error = fabsf(output[0] - expected);
    if (error > 1e-6F) {
        fprintf(
            stderr,
            "expected %.7f, got %.7f\n",
            (double)expected,
            (double)output[0]
        );
        return 1;
    }

    printf("PASS: generated model output matches reference\n");
    return 0;
}
