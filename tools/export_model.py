#!/usr/bin/env python3
"""Export a validated dense NN-C model from JSON to a C header."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


FORMAT_NAME = "nnc-dense-v1"
ACTIVATIONS = {
    "linear": "NN_ACTIVATION_LINEAR",
    "relu": "NN_ACTIVATION_RELU",
    "sigmoid": "NN_ACTIVATION_SIGMOID",
    "tanh": "NN_ACTIVATION_TANH",
}
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExportError(ValueError):
    """Raised when an input model cannot be exported safely."""


@dataclass(frozen=True)
class DenseLayer:
    input_count: int
    output_count: int
    activation: str
    weights: tuple[float, ...]
    biases: tuple[float, ...]


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExportError(f"{field} must be a positive integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{field} must be numeric")

    result = float(value)
    if not math.isfinite(result):
        raise ExportError(f"{field} must be finite")
    return result


def _parse_layer(raw: Any, index: int) -> DenseLayer:
    prefix = f"layers[{index}]"
    if not isinstance(raw, dict):
        raise ExportError(f"{prefix} must be an object")

    input_count = _positive_int(raw.get("input_count"), f"{prefix}.input_count")
    output_count = _positive_int(raw.get("output_count"), f"{prefix}.output_count")

    activation = raw.get("activation")
    if activation not in ACTIVATIONS:
        supported = ", ".join(sorted(ACTIVATIONS))
        raise ExportError(
            f"{prefix}.activation must be one of: {supported}"
        )

    raw_weights = raw.get("weights")
    if not isinstance(raw_weights, list) or len(raw_weights) != output_count:
        raise ExportError(
            f"{prefix}.weights must contain {output_count} output rows"
        )

    weights: list[float] = []
    for out, row in enumerate(raw_weights):
        if not isinstance(row, list) or len(row) != input_count:
            raise ExportError(
                f"{prefix}.weights[{out}] must contain {input_count} values"
            )
        for inp, value in enumerate(row):
            weights.append(
                _finite_float(value, f"{prefix}.weights[{out}][{inp}]")
            )

    raw_biases = raw.get("biases")
    if not isinstance(raw_biases, list) or len(raw_biases) != output_count:
        raise ExportError(
            f"{prefix}.biases must contain {output_count} values"
        )

    biases = tuple(
        _finite_float(value, f"{prefix}.biases[{position}]")
        for position, value in enumerate(raw_biases)
    )

    return DenseLayer(
        input_count=input_count,
        output_count=output_count,
        activation=activation,
        weights=tuple(weights),
        biases=biases,
    )


def load_model(path: Path) -> tuple[DenseLayer, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ExportError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ExportError(
            f"{path}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        ) from error

    if not isinstance(raw, dict):
        raise ExportError("model root must be an object")
    if raw.get("format") != FORMAT_NAME:
        raise ExportError(f"format must be {FORMAT_NAME!r}")

    raw_layers = raw.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ExportError("layers must be a non-empty array")

    layers = tuple(
        _parse_layer(layer, index) for index, layer in enumerate(raw_layers)
    )

    for index in range(1, len(layers)):
        previous = layers[index - 1]
        current = layers[index]
        if previous.output_count != current.input_count:
            raise ExportError(
                f"layers[{index - 1}].output_count "
                f"({previous.output_count}) does not match "
                f"layers[{index}].input_count ({current.input_count})"
            )

    return layers


def _c_float(value: float) -> str:
    if value == 0.0:
        value = 0.0
    rendered = format(value, ".9g")
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "F"


def _array_lines(values: Sequence[float], indent: str = "    ") -> list[str]:
    lines: list[str] = []
    for start in range(0, len(values), 6):
        chunk = ", ".join(_c_float(value) for value in values[start : start + 6])
        suffix = "," if start + 6 < len(values) else ""
        lines.append(f"{indent}{chunk}{suffix}")
    return lines


def render_header(layers: Sequence[DenseLayer], symbol: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(symbol):
        raise ExportError(
            "symbol must be a valid C identifier "
            "(letters, digits and underscores; no leading digit)"
        )

    guard = f"NNC_GENERATED_{symbol.upper()}_H"
    prefix = symbol.lower()
    workspace_count = 2 * max(layer.output_count for layer in layers)

    lines = [
        "/* Generated by tools/export_model.py; do not edit. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "nnc/nn.h"',
        "",
        f"#define {symbol.upper()}_LAYER_COUNT {len(layers)}U",
        f"#define {symbol.upper()}_WORKSPACE_FLOATS {workspace_count}U",
        "",
    ]

    for index, layer in enumerate(layers):
        lines.append(f"static const float {prefix}_layer_{index}_weights[] = {{")
        lines.extend(_array_lines(layer.weights))
        lines.extend(
            [
                "};",
                "",
                f"static const float {prefix}_layer_{index}_biases[] = {{",
            ]
        )
        lines.extend(_array_lines(layer.biases))
        lines.extend(["};", ""])

    lines.append(
        f"static const nn_dense_layer_t {prefix}_layers"
        f"[{symbol.upper()}_LAYER_COUNT] = {{"
    )
    for index, layer in enumerate(layers):
        lines.extend(
            [
                "    {",
                f"        .input_count = {layer.input_count}U,",
                f"        .output_count = {layer.output_count}U,",
                f"        .weights = {prefix}_layer_{index}_weights,",
                f"        .biases = {prefix}_layer_{index}_biases,",
                f"        .activation = {ACTIVATIONS[layer.activation]}",
                "    },",
            ]
        )
    lines.extend(["};", "", f"#endif /* {guard} */", ""])
    return "\n".join(lines)


def export_model(
    model_path: Path,
    output_path: Path,
    symbol: str,
    check: bool,
) -> int:
    layers = load_model(model_path)
    rendered = render_header(layers, symbol)

    if check:
        try:
            existing = output_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ExportError(f"cannot read {output_path}: {error}") from error
        if existing != rendered:
            print(
                f"{output_path} is stale; regenerate it with export_model.py",
                file=sys.stderr,
            )
            return 1
        print(f"{output_path} is up to date")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(
        f"exported {len(layers)} layer(s) to {output_path} "
        f"as {symbol}"
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an nnc-dense-v1 JSON model to a C header."
    )
    parser.add_argument("model", type=Path, help="input JSON model")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--symbol",
        default="nnc_model",
        help="C identifier prefix for generated declarations",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the output is missing or differs from generated content",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return export_model(args.model, args.output, args.symbol, args.check)
    except ExportError as error:
        print(f"export_model: error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"export_model: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
