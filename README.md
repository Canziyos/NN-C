# NN-C

NN-C is a small, dependency-free neural-network inference library written in C99.
It is intended as an auditable foundation for resource-constrained and embedded
experiments, not as a replacement for mature runtimes such as TensorFlow Lite
Micro or CMSIS-NN.

## Current scope

- Dense, sequential feed-forward networks
- Caller-owned workspace with no heap allocation
- Externally supplied, read-only weights and biases
- Linear, ReLU, sigmoid and tanh activations
- Dimension and workspace validation
- Deterministic numerical tests
- Portable CMake build and Linux CI

The inference path uses `float` values. Weights are stored in output-major row
order:

```text
weights[output_neuron * input_count + input_neuron]
```

## Build and test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the example:

```sh
./build/nnc_basic
```

On multi-configuration Windows generators, the executable may be under
`build/Debug` or `build/Release`.

## Minimal model

```c
#include "nnc/nn.h"

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

nn_model_init(&model, layers, 1, workspace, 4);
```

Use `nn_workspace_floats()` when the required workspace size should be
calculated from a model description. Call `nn_predict()` with explicit input
and output lengths; it returns a status code rather than terminating the host
application.

## Design boundaries

NN-C currently performs inference only. Training belongs in a host-side tool,
with learned parameters exported into C arrays or a compact model format.
Quantization, convolution and microcontroller benchmarks are possible later
milestones, but they will be added only with tests and a concrete use case.

## License

[MIT](LICENSE)
