#ifndef NNC_NN_H
#define NNC_NN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NN_ACTIVATION_LINEAR = 0,
    NN_ACTIVATION_RELU,
    NN_ACTIVATION_SIGMOID,
    NN_ACTIVATION_TANH
} nn_activation_t;

typedef enum {
    NN_STATUS_OK = 0,
    NN_STATUS_INVALID_ARGUMENT,
    NN_STATUS_INVALID_MODEL,
    NN_STATUS_DIMENSION_MISMATCH,
    NN_STATUS_WORKSPACE_TOO_SMALL
} nn_status_t;

/*
 * Weights use output-major row order:
 * weights[output_neuron * input_count + input_neuron].
 */
typedef struct {
    size_t input_count;
    size_t output_count;
    const float *weights;
    const float *biases;
    nn_activation_t activation;
} nn_dense_layer_t;

typedef struct {
    const nn_dense_layer_t *layers;
    size_t layer_count;
    float *workspace;
    size_t workspace_count;
    size_t max_width;
} nn_model_t;

/*
 * Returns the number of float elements required for a model workspace.
 * Returns 0 when the layer description is invalid or the size overflows.
 */
size_t nn_workspace_floats(const nn_dense_layer_t *layers, size_t layer_count);

nn_status_t nn_model_init(
    nn_model_t *model,
    const nn_dense_layer_t *layers,
    size_t layer_count,
    float *workspace,
    size_t workspace_count
);

nn_status_t nn_predict(
    nn_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count
);

const char *nn_status_string(nn_status_t status);

#ifdef __cplusplus
}
#endif

#endif
