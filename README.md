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
- Dependency-free Python-to-C model exporter
- Portable CMake build and Linux CI

The inference path uses `float` values. Weights are stored in output-major row
order:

```text
weights[output_neuron * input_count + input_neuron]
```

## Build and test

For single-configuration generators:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/nnc_basic
```

Visual Studio uses a multi-configuration generator. In PowerShell, select the
configuration for both the build and CTest:

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
.\build\Debug\nnc_basic.exe
```

Replace `Debug` with `Release` for an optimized build.

## Export a model

The exporter accepts a versioned JSON description and writes a deterministic C
header containing weights, biases, layer descriptors and the required workspace
size:

```powershell
python tools\export_model.py models\example_model.json `
    --output examples\generated_model.h `
    --symbol example_model
```

Check that a committed generated header is current without rewriting it:

```powershell
python tools\export_model.py models\example_model.json `
    --output examples\generated_model.h `
    --symbol example_model `
    --check
```

The JSON format is deliberately small:

```json
{
  "format": "nnc-dense-v1",
  "layers": [
    {
      "input_count": 2,
      "output_count": 2,
      "activation": "sigmoid",
      "weights": [[0.5, 0.25], [-0.4, 0.8]],
      "biases": [0.1, -0.2]
    }
  ]
}
```

Each weight row belongs to one output neuron. Supported activation names are
`linear`, `relu`, `sigmoid` and `tanh`. The exporter rejects malformed
shapes, disconnected layers, unsupported activations, non-finite values and
invalid C symbols.

Run the generated-model example after building:

```powershell
.\build\Debug\nnc_generated.exe
```

Training code can write this JSON directly. The embedded C build consumes only
the generated header and retains no Python runtime dependency.

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
with learned parameters exported through the JSON interchange format.
Quantization, convolution and microcontroller benchmarks are possible later
milestones, but they will be added only with tests and a concrete use case.

## License

[MIT](LICENSE)
