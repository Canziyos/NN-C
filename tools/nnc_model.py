"""Load and validate the versioned NN-C JSON model format."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT_NAME = "nnc-dense-v1"
ACTIVATIONS = {"linear", "relu", "sigmoid", "tanh"}


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class DenseLayer:
    inputs: int
    outputs: int
    activation: str
    weights: tuple[float, ...]
    biases: tuple[float, ...]


def _size(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{field} must be a positive integer")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ModelError(f"{field} must be finite")
    return result


def _layer(raw: Any, index: int) -> DenseLayer:
    name = f"layers[{index}]"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")

    inputs = _size(raw.get("input_count"), f"{name}.input_count")
    outputs = _size(raw.get("output_count"), f"{name}.output_count")
    activation = raw.get("activation")
    if activation not in ACTIVATIONS:
        raise ModelError(f"{name}.activation is unsupported")

    rows = raw.get("weights")
    if not isinstance(rows, list) or len(rows) != outputs:
        raise ModelError(f"{name}.weights must contain {outputs} rows")

    weights: list[float] = []
    for output, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != inputs:
            raise ModelError(
                f"{name}.weights[{output}] must contain {inputs} values"
            )
        weights.extend(
            _number(value, f"{name}.weights[{output}][{position}]")
            for position, value in enumerate(row)
        )

    raw_biases = raw.get("biases")
    if not isinstance(raw_biases, list) or len(raw_biases) != outputs:
        raise ModelError(f"{name}.biases must contain {outputs} values")
    biases = tuple(
        _number(value, f"{name}.biases[{position}]")
        for position, value in enumerate(raw_biases)
    )
    return DenseLayer(inputs, outputs, activation, tuple(weights), biases)


def load_model(path: Path) -> tuple[DenseLayer, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot parse {path}: {error}") from error

    if not isinstance(raw, dict) or raw.get("format") != FORMAT_NAME:
        raise ModelError(f"format must be {FORMAT_NAME!r}")
    raw_layers = raw.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ModelError("layers must be a non-empty array")

    layers = tuple(_layer(value, index) for index, value in enumerate(raw_layers))
    for index in range(1, len(layers)):
        if layers[index - 1].outputs != layers[index].inputs:
            raise ModelError(
                f"layers[{index - 1}].output_count does not match "
                f"layers[{index}].input_count"
            )
    return layers
