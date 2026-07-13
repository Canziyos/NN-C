#!/usr/bin/env python3
"""Train a deterministic 2-4-1 XOR network without external dependencies."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from nnc_model import canonicalize_numbers

SAMPLES = (
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def initialize(rows: int, columns: int, rng: random.Random) -> list[list[float]]:
    limit = math.sqrt(6.0 / (rows + columns))
    return [
        [rng.uniform(-limit, limit) for _ in range(columns)]
        for _ in range(rows)
    ]


def train(epochs: int, learning_rate: float) -> tuple[dict, float]:
    rng = random.Random(7)
    hidden_weights = initialize(4, 2, rng)
    hidden_biases = [0.0] * 4
    output_weights = initialize(1, 4, rng)
    output_bias = 0.0
    loss = 0.0
    for _ in range(epochs):
        hidden_weight_grad = [[0.0] * 2 for _ in range(4)]
        hidden_bias_grad = [0.0] * 4
        output_weight_grad = [[0.0] * 4]
        output_bias_grad = 0.0
        loss = 0.0

        for inputs, target in SAMPLES:
            hidden = [
                math.tanh(
                    sum(weight * value for weight, value in zip(row, inputs))
                    + bias
                )
                for row, bias in zip(hidden_weights, hidden_biases)
            ]
            prediction = sigmoid(
                sum(weight * value for weight, value in zip(output_weights[0], hidden))
                + output_bias
            )
            loss -= target * math.log(prediction) + (1.0 - target) * math.log(
                1.0 - prediction
            )
            output_delta = prediction - target
            for neuron, value in enumerate(hidden):
                output_weight_grad[0][neuron] += output_delta * value
                hidden_delta = (
                    output_delta
                    * output_weights[0][neuron]
                    * (1.0 - value * value)
                )
                hidden_bias_grad[neuron] += hidden_delta
                for feature, input_value in enumerate(inputs):
                    hidden_weight_grad[neuron][feature] += (
                        hidden_delta * input_value
                    )
            output_bias_grad += output_delta

        scale = learning_rate / len(SAMPLES)
        for neuron in range(4):
            for feature in range(2):
                hidden_weights[neuron][feature] -= (
                    scale * hidden_weight_grad[neuron][feature]
                )
            hidden_biases[neuron] -= scale * hidden_bias_grad[neuron]
            output_weights[0][neuron] -= (
                scale * output_weight_grad[0][neuron]
            )
        output_bias -= scale * output_bias_grad

    model = {
        "format": "nnc-dense-v1",
        "layers": [
            {
                "input_count": 2,
                "output_count": 4,
                "activation": "tanh",
                "weights": hidden_weights,
                "biases": hidden_biases,
            },
            {
                "input_count": 4,
                "output_count": 1,
                "activation": "sigmoid",
                "weights": output_weights,
                "biases": [output_bias],
            },
        ],
    }
    return canonicalize_numbers(model), loss / len(SAMPLES)


def render(model: dict) -> str:
    return json.dumps(model, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.learning_rate <= 0.0:
        parser.error("epochs and learning rate must be positive")
    model, loss = train(args.epochs, args.learning_rate)
    trained = render(model)

    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError as error:
            print(f"train_xor: error: {error}", file=sys.stderr)
            return 2
        if current != trained:
            print(f"{args.output} is stale; retrain it", file=sys.stderr)
            return 1
        print(f"{args.output} is reproducible; loss={loss:.6f}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(trained, encoding="utf-8", newline="\n")
    print(f"trained XOR model: loss={loss:.6f}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
