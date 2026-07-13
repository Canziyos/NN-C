#!/usr/bin/env python3
"""Command-line JSON-to-C exporter for NN-C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nnc_header import render_header
from nnc_model import ModelError, load_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an NN-C JSON model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--symbol", default="nnc_model")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def run(model: Path, output: Path, symbol: str, check: bool) -> int:
    rendered = render_header(load_model(model), symbol)
    if check:
        try:
            current = output.read_text(encoding="utf-8")
        except OSError as error:
            raise ModelError(f"cannot read {output}: {error}") from error
        if current != rendered:
            print(f"{output} is stale; regenerate it", file=sys.stderr)
            return 1
        print(f"{output} is up to date")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"exported {len(load_model(model))} layer(s) to {output}")
    return 0


def main() -> int:
    args = arguments()
    try:
        return run(args.model, args.output, args.symbol, args.check)
    except (ModelError, OSError) as error:
        print(f"export_model: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
