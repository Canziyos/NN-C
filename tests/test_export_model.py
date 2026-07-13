import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nnc_header  # noqa: E402
import nnc_model  # noqa: E402


class ExportModelTests(unittest.TestCase):
    def test_canonicalizes_platform_float_noise(self) -> None:
        model = {"weights": [[0.123456789123, -0.0]]}
        self.assertEqual(
            nnc_model.canonicalize_numbers(model),
            {"weights": [[0.12345679, -0.0]]},
        )

    def test_committed_header_is_current(self) -> None:
        layers = nnc_model.load_model(ROOT / "models" / "example_model.json")
        expected = nnc_header.render_header(layers, "example_model")
        actual = (ROOT / "examples" / "generated_model.h").read_text(
            encoding="utf-8"
        )
        self.assertEqual(actual, expected)

    def test_rejects_disconnected_layers(self) -> None:
        model = {
            "format": "nnc-dense-v1",
            "layers": [
                {
                    "input_count": 2,
                    "output_count": 2,
                    "activation": "relu",
                    "weights": [[1, 0], [0, 1]],
                    "biases": [0, 0],
                },
                {
                    "input_count": 3,
                    "output_count": 1,
                    "activation": "linear",
                    "weights": [[1, 1, 1]],
                    "biases": [0],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(
                nnc_model.ModelError,
                "does not match",
            ):
                nnc_model.load_model(path)

    def test_rejects_invalid_c_symbol(self) -> None:
        layers = nnc_model.load_model(ROOT / "models" / "example_model.json")
        with self.assertRaisesRegex(nnc_model.ModelError, "C identifier"):
            nnc_header.render_header(layers, "9-invalid")


if __name__ == "__main__":
    unittest.main()
