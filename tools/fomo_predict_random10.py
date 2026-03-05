import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

from fomo_infer_utils import load_labels, run_model


def format_items(items: List[Tuple[int, int, int, int, float]]) -> str:
    if not items:
        return "none"
    parts = []
    for x, y, w, h, score in items:
        cx = x + (w // 2)
        cy = y + (h // 2)
        parts.append(f"(cx={cx}, cy={cy}, score={score:.3f}, bbox=[{x},{y},{w},{h}])")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction on random 10 images")
    parser.add_argument("--image-dir", default="data/yolo_dataset/images/val")
    parser.add_argument("--model", default="outputs/fomo_short_run/fomo_int8.tflite")
    parser.add_argument("--labels", default="outputs/fomo_short_run/fomo_labels.txt")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--save", default="outputs/fomo_short_run/random10_predictions.md")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    model_path = Path(args.model).resolve()
    labels = load_labels(Path(args.labels).resolve())

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        all_images.extend(image_dir.glob(ext))
    all_images = sorted(all_images)

    if len(all_images) < args.count:
        raise ValueError(f"Not enough images in {image_dir}. found={len(all_images)}, need={args.count}")

    rng = random.Random(args.seed)
    picks = rng.sample(all_images, args.count)

    lines = ["# FOMO Random 10 Predictions", ""]
    lines.append(f"model: `{model_path}`")
    lines.append(f"image_dir: `{image_dir}`")
    lines.append(f"min_conf: `{args.min_conf}`")
    lines.append("")

    print(f"Using model: {model_path}")
    print(f"Sampling {args.count} images from: {image_dir}")

    for i, image_path in enumerate(picks, start=1):
        dets, out_shape = run_model(model_path, image_path, args.min_conf)
        print(f"\n[{i}] {image_path.name} | output_shape={out_shape}")
        lines.append(f"## {i}. {image_path.name}")
        lines.append(f"- path: `{image_path}`")
        lines.append(f"- output_shape: `{out_shape}`")

        for class_idx, class_name in enumerate(labels):
            if class_idx == 0:
                continue
            items = dets.get(class_idx, [])
            formatted = format_items(items)
            print(f"  - {class_name}: {formatted}")
            lines.append(f"- {class_name}: {formatted}")

        lines.append("")

    save_path = Path(args.save).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report: {save_path}")


if __name__ == "__main__":
    main()
