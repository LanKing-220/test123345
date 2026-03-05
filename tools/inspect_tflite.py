import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


def load_interpreter(model_path: str):
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            return Interpreter(model_path=model_path)
        except ImportError as exc:
            raise RuntimeError(
                "Neither tensorflow nor tflite_runtime is available. "
                "Please install one of them in your conda env."
            ) from exc

    return tf.lite.Interpreter(model_path=model_path)


def normalize_detail(detail: dict) -> dict:
    d = {}
    for k, v in detail.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, (tuple, set)):
            d[k] = list(v)
        elif isinstance(v, type):
            d[k] = v.__name__
        else:
            d[k] = v
    return d


def to_ascii_loadable_path(src: Path) -> Path:
    src_resolved = src.resolve()
    try:
        str(src_resolved).encode("ascii")
        return src_resolved
    except UnicodeEncodeError:
        temp_dir = Path(tempfile.gettempdir()) / "fomo_tflite_ascii"
        temp_dir.mkdir(parents=True, exist_ok=True)
        dst = temp_dir / src_resolved.name
        shutil.copyfile(src_resolved, dst)
        return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect .tflite model IO specs")
    parser.add_argument("model", type=str, help="Path to .tflite model")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    load_path = to_ascii_loadable_path(model_path)
    interpreter = load_interpreter(str(load_path))
    interpreter.allocate_tensors()

    inputs = [normalize_detail(x) for x in interpreter.get_input_details()]
    outputs = [normalize_detail(x) for x in interpreter.get_output_details()]

    payload = {
        "model_path": str(model_path.resolve()),
        "load_path": str(load_path),
        "inputs": inputs,
        "outputs": outputs,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Model: {model_path}")
    print("=" * 80)
    print("Inputs:")
    for i, detail in enumerate(inputs):
        print(f"[{i}] name={detail.get('name')}")
        print(f"    shape={detail.get('shape')}")
        print(f"    dtype={detail.get('dtype')}")
        print(f"    quantization={detail.get('quantization')}")
        print(f"    quantization_parameters={detail.get('quantization_parameters')}")

    print("Outputs:")
    for i, detail in enumerate(outputs):
        print(f"[{i}] name={detail.get('name')}")
        print(f"    shape={detail.get('shape')}")
        print(f"    dtype={detail.get('dtype')}")
        print(f"    quantization={detail.get('quantization')}")
        print(f"    quantization_parameters={detail.get('quantization_parameters')}")


if __name__ == "__main__":
    main()
