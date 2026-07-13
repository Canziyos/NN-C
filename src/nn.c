#include "nnc/nn.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int activation_is_valid(nn_activation_t activation)
{
    return activation >= NN_ACTIVATION_LINEAR
        && activation <= NN_ACTIVATION_TANH;
}

static nn_status_t validate_layers(
    const nn_dense_layer_t *layers,
    size_t layer_count,
    size_t *max_width
)
{
    if (layers == NULL || layer_count == 0 || max_width == NULL) {
        return NN_STATUS_INVALID_ARGUMENT;
    }

    size_t widest = 0;

    for (size_t layer = 0; layer < layer_count; ++layer) {
        const nn_dense_layer_t *current = &layers[layer];

        if (current->input_count == 0
            || current->output_count == 0
            || current->weights == NULL
            || current->biases == NULL
            || !activation_is_valid(current->activation)) {
            return NN_STATUS_INVALID_MODEL;
        }

        if (layer > 0
            && layers[layer - 1].output_count != current->input_count) {
            return NN_STATUS_DIMENSION_MISMATCH;
        }

        if (current->output_count > widest) {
            widest = current->output_count;
        }
    }

    *max_width = widest;
    return NN_STATUS_OK;
}

static float activate(float value, nn_activation_t activation)
{
    switch (activation) {
    case NN_ACTIVATION_LINEAR:
        return value;
    case NN_ACTIVATION_RELU:
        return value > 0.0F ? value : 0.0F;
    case NN_ACTIVATION_SIGMOID:
        if (value >= 0.0F) {
            return 1.0F / (1.0F + expf(-value));
        }
        {
            const float exponential = expf(value);
            return exponential / (1.0F + exponential);
        }
    case NN_ACTIVATION_TANH:
        return tanhf(value);
    default:
        return value;
    }
}

size_t nn_workspace_floats(const nn_dense_layer_t *layers, size_t layer_count)
{
    size_t max_width = 0;

    if (validate_layers(layers, layer_count, &max_width) != NN_STATUS_OK
        || max_width > SIZE_MAX / 2U) {
        return 0;
    }

    return max_width * 2U;
}

nn_status_t nn_model_init(
    nn_model_t *model,
    const nn_dense_layer_t *layers,
    size_t layer_count,
    float *workspace,
    size_t workspace_count
)
{
    if (model == NULL || workspace == NULL) {
        return NN_STATUS_INVALID_ARGUMENT;
    }

    size_t max_width = 0;
    const nn_status_t status = validate_layers(layers, layer_count, &max_width);
    if (status != NN_STATUS_OK) {
        return status;
    }

    if (max_width > SIZE_MAX / 2U
        || workspace_count < max_width * 2U) {
        return NN_STATUS_WORKSPACE_TOO_SMALL;
    }

    model->layers = layers;
    model->layer_count = layer_count;
    model->workspace = workspace;
    model->workspace_count = workspace_count;
    model->max_width = max_width;
    return NN_STATUS_OK;
}

nn_status_t nn_predict(
    nn_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count
)
{
    if (model == NULL || input == NULL || output == NULL) {
        return NN_STATUS_INVALID_ARGUMENT;
    }

    if (model->layers == NULL
        || model->layer_count == 0
        || model->workspace == NULL
        || model->max_width == 0) {
        return NN_STATUS_INVALID_MODEL;
    }

    const nn_dense_layer_t *first = &model->layers[0];
    const nn_dense_layer_t *last = &model->layers[model->layer_count - 1U];

    if (input_count != first->input_count
        || output_count != last->output_count) {
        return NN_STATUS_DIMENSION_MISMATCH;
    }

    const float *layer_input = input;
    float *buffer_a = model->workspace;
    float *buffer_b = model->workspace + model->max_width;

    for (size_t layer = 0; layer < model->layer_count; ++layer) {
        const nn_dense_layer_t *current = &model->layers[layer];
        float *layer_output = (layer % 2U == 0U) ? buffer_a : buffer_b;

        for (size_t out = 0; out < current->output_count; ++out) {
            float sum = current->biases[out];
            const size_t row = out * current->input_count;

            for (size_t in = 0; in < current->input_count; ++in) {
                sum += current->weights[row + in] * layer_input[in];
            }

            layer_output[out] = activate(sum, current->activation);
        }

        layer_input = layer_output;
    }

    memcpy(output, layer_input, output_count * sizeof(*output));
    return NN_STATUS_OK;
}

const char *nn_status_string(nn_status_t status)
{
    switch (status) {
    case NN_STATUS_OK:
        return "ok";
    case NN_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case NN_STATUS_INVALID_MODEL:
        return "invalid model";
    case NN_STATUS_DIMENSION_MISMATCH:
        return "dimension mismatch";
    case NN_STATUS_WORKSPACE_TOO_SMALL:
        return "workspace too small";
    default:
        return "unknown status";
    }
}
