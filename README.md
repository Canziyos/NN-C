# NN-C

NN-C is a small C99 neural-network inference library for auditable embedded
experiments. It is not intended to replace TensorFlow Lite Micro or CMSIS-NN.

## Features

- Sequential dense networks using caller-owned workspace
- No heap allocation during inference
- Read-only external weights and biases
- Linear, ReLU, sigmoid and tanh activations
- Shape and workspace validation with explicit status codes
- Dependency-free Python training and JSON-to-C export examples
- Deterministic C/Python tests and GitHub Actions CI

Weights use output-major row order:

```text
weights[output_neuron * input_count + input_neuron]
```

## Build and test

Single-configuration generators:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Visual Studio PowerShell:

```powershell
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
.\build\Debug\nnc_basic.exe
```

Replace `Debug` with `Release` for an optimized build.

## Train and export

`train_xor.py` is a deterministic, dependency-free host-training example. It
learns a 2-4-1 nonlinear XOR model and writes the versioned JSON interchange
format:

```powershell
python tools\train_xor.py --output models\trained_xor.json
python tools\export_model.py models\trained_xor.json `
    --output examples\trained_xor_model.h --symbol trained_xor
```

Build and run the exported model:

```powershell
cmake --build build --config Debug
.\build\Debug\nnc_trained_xor.exe
```

The example prints all four XOR probabilities and classifications. CTest also
retrains the model, checks byte-for-byte reproducibility, verifies that the
generated header is current, and runs the learned parameters through the C
inference engine.

## JSON model format

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

Each row belongs to one output neuron. Supported activations are `linear`,
`relu`, `sigmoid` and `tanh`. The exporter rejects invalid shapes,
disconnected layers, unsupported activations, non-finite values and invalid C
symbols.

Check a committed header without rewriting it:

```powershell
python tools\export_model.py models\example_model.json `
    -o examples\generated_model.h --symbol example_model --check
```

The embedded C build consumes only the generated header; it has no Python
runtime dependency. Training remains a host-side responsibility.

## Scope

NN-C currently supports dense `float` inference. Quantization, convolution and
microcontroller benchmarks are later milestones only when backed by tests and a
concrete use case.

## License

[MIT](LICENSE)
