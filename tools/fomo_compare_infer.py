import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from fomo_infer_utils import load_labels, run_model


def print_detections(title: str, labels: List[str], dets: Dict[int, List[Tuple[int, int, int, int, float]]]):
    print(f"\n=== {title} ===")
    for class_idx, class_name in enumerate(labels):
        if class_idx == 0:
            continue
        items = dets.get(class_idx, [])
        if not items:
            continue
        print(f"{class_name}:")
        for x, y, w, h, score in items:
            cx = x + (w // 2)
            cy = y + (h // 2)
            print(f"  x {cx}\ty {cy}\tscore {score:.4f}\tbbox [{x},{y},{w},{h}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original/new FOMO model inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default="ei-tennis-openmv-v6/labels.txt")
    parser.add_argument("--orig-model", default="ei-tennis-openmv-v6/trained.tflite")
    parser.add_argument("--new-model", required=True)
    parser.add_argument("--min-conf", type=float, default=0.5)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    labels = load_labels(Path(args.labels).resolve())

    orig_dets, orig_out_shape = run_model(Path(args.orig_model).resolve(), image_path, args.min_conf)
    new_dets, new_out_shape = run_model(Path(args.new_model).resolve(), image_path, args.min_conf)

    print(f"Image: {image_path}")
    print(f"Original output shape: {orig_out_shape}")
    print(f"New output shape: {new_out_shape}")

    print_detections("Original model", labels, orig_dets)
    print_detections("New model", labels, new_dets)


if __name__ == "__main__":
    main()
