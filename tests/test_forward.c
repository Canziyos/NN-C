#include "nnc/nn.h"

#include <math.h>
#include <stdio.h>

static int expect_close(float actual, float expected, float tolerance)
{
    const float error = fabsf(actual - expected);
    if (error <= tolerance) {
        return 0;
    }

    fprintf(
        stderr,
        "expected %.7f, got %.7f, error %.7f\n",
        (double)expected,
        (double)actual,
        (double)error
    );
    return 1;
}

static int test_deterministic_sigmoid_network(void)
{
    static const float hidden_weights[] = {
        0.5F, 0.25F,
        -0.4F, 0.8F
    };
    static const float hidden_biases[] = {0.1F, -0.2F};
    static const float output_weights[] = {1.2F, -0.7F};
    static const float output_biases[] = {0.05F};

    static const nn_dense_layer_t layers[] = {
        {
            .input_count = 2,
            .output_count = 2,
            .weights = hidden_weights,
            .biases = hidden_biases,
            .activation = NN_ACTIVATION_SIGMOID
        },
        {
            .input_count = 2,
            .output_count = 1,
            .weights = output_weights,
            .biases = output_biases,
            .activation = NN_ACTIVATION_SIGMOID
        }
    };

    float workspace[4];
    nn_model_t model;
    nn_status_t status = nn_model_init(
        &model,
        layers,
        sizeof(layers) / sizeof(layers[0]),
        workspace,
        sizeof(workspace) / sizeof(workspace[0])
    );
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "model init failed: %s\n", nn_status_string(status));
        return 1;
    }

    const float input[] = {0.6F, -0.1F};
    float output[1];
    status = nn_predict(&model, input, 2, output, 1);
    if (status != NN_STATUS_OK) {
        fprintf(stderr, "prediction failed: %s\n", nn_status_string(status));
        return 1;
    }

    return expect_close(output[0], 0.622507F, 1e-6F);
}

static int test_linear_and_relu_layers(void)
{
    static const float linear_weights[] = {
        2.0F, -1.0F,
        -3.0F, 1.0F
    };
    static const float linear_biases[] = {0.5F, 0.0F};
    static const float relu_weights[] = {
        1.0F, 1.0F
    };
    static const float relu_biases[] = {2.0F};

    static const nn_dense_layer_t layers[] = {
        {
            .input_count = 2,
            .output_count = 2,
            .weights = linear_weights,
            .biases = linear_biases,
            .activation = NN_ACTIVATION_LINEAR
        },
        {
            .input_count = 2,
            .output_count = 1,
            .weights = relu_weights,
            .biases = relu_biases,
            .activation = NN_ACTIVATION_RELU
        }
    };

    float workspace[4];
    nn_model_t model;
    const float input[] = {2.0F, 1.0F};
    float output[1];

    if (nn_model_init(&model, layers, 2, workspace, 4) != NN_STATUS_OK
        || nn_predict(&model, input, 2, output, 1) != NN_STATUS_OK) {
        return 1;
    }

    return expect_close(output[0], 0.5F, 1e-6F);
}

static int test_validation(void)
{
    static const float weights[] = {1.0F};
    static const float biases[] = {0.0F};
    static const nn_dense_layer_t layer = {
        .input_count = 1,
        .output_count = 1,
        .weights = weights,
        .biases = biases,
        .activation = NN_ACTIVATION_TANH
    };

    float workspace[2];
    nn_model_t model;

    if (nn_workspace_floats(&layer, 1) != 2U) {
        return 1;
    }

    if (nn_model_init(&model, &layer, 1, workspace, 1)
        != NN_STATUS_WORKSPACE_TOO_SMALL) {
        return 1;
    }

    if (nn_model_init(&model, &layer, 1, workspace, 2) != NN_STATUS_OK) {
        return 1;
    }

    const float input[] = {0.25F};
    float output[1];
    return nn_predict(&model, input, 0, output, 1)
        == NN_STATUS_DIMENSION_MISMATCH ? 0 : 1;
}

int main(void)
{
    int failures = 0;
    failures += test_deterministic_sigmoid_network();
    failures += test_linear_and_relu_layers();
    failures += test_validation();

    if (failures != 0) {
        fprintf(stderr, "FAIL: %d test group(s) failed\n", failures);
        return 1;
    }

    printf("PASS: all nnc tests\n");
    return 0;
}
