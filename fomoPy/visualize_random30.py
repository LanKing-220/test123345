import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import yaml

from eval_testset import imread_unicode, run_tflite
from train import parse_yolo_file


COLORS = [
    (255, 80, 80),
    (80, 220, 120),
    (80, 160, 255),
    (255, 200, 80),
    (220, 120, 255),
]


def load_and_resize(image_path: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    img_bgr = imread_unicode(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    resized_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img_bgr, rgb


def gt_points_from_label(label_path: Path, image_size: int, class_names: List[str]) -> List[Tuple[str, int, int]]:
    points = []
    rows = parse_yolo_file(label_path)
    for cid, cx, cy, _, _ in rows:
        x = int(np.clip(cx, 0.0, 0.9999) * image_size)
        y = int(np.clip(cy, 0.0, 0.9999) * image_size)
        if 0 <= cid < len(class_names):
            points.append((class_names[cid], x, y))
    return points


def pred_points_from_output(output: np.ndarray, threshold: float, image_size: int, class_names: List[str]) -> List[Tuple[str, int, int, float]]:
    grid_h, grid_w, channels = output.shape
    points = []
    fg = output[..., 1:]
    for gy in range(grid_h):
        for gx in range(grid_w):
            for cid in range(min(len(class_names), channels - 1)):
                score = float(fg[gy, gx, cid])
                if score >= threshold:
                    x = int((gx + 0.5) * image_size / grid_w)
                    y = int((gy + 0.5) * image_size / grid_h)
                    points.append((class_names[cid], x, y, score))
    return points


def draw_overlay(
    base_bgr: np.ndarray,
    gt_points: List[Tuple[str, int, int]],
    pred_points: List[Tuple[str, int, int, float]],
    class_names: List[str],
) -> np.ndarray:
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]

    # GT: circle
    for cls_name, x, y in gt_points:
        color = COLORS[class_names.index(cls_name) % len(COLORS)]
        cv2.circle(canvas, (x, y), 10, color, 2)
        cv2.putText(canvas, f"GT:{cls_name}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Pred: cross
    for cls_name, x, y, score in pred_points:
        color = COLORS[class_names.index(cls_name) % len(COLORS)]
        cv2.line(canvas, (x - 8, y), (x + 8, y), color, 2)
        cv2.line(canvas, (x, y - 8), (x, y + 8), color, 2)
        cv2.putText(
            canvas,
            f"P:{cls_name} {score:.2f}",
            (x + 8, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    legend_y = 22
    cv2.putText(canvas, "Circle = GT, Cross = Prediction", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Circle = GT, Cross = Prediction", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 30 random predictions from train/test split")
    parser.add_argument("--model", type=str, default="fomoPy/outputs/fomo_local/fomo_like_int8.tflite")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/yolo_labels")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="fomoPy/outputs/fomo_local/random30_test")
    args = parser.parse_args()

    workspace = Path.cwd()
    model_path = workspace / args.model
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir
    out_dir = workspace / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() else (workspace / cfg["path"]).resolve()
    class_names = list(cfg["names"])

    if args.split == "train":
        image_dir = dataset_root / cfg["train"]
        label_dir = labels_root / "training"
    else:
        image_dir = dataset_root / cfg["val"]
        label_dir = labels_root / "testing"

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(ext))
    image_paths = sorted([p for p in image_paths if (label_dir / f"{p.stem}.txt").exists()])

    rng = random.Random(args.seed)
    sample_paths = rng.sample(image_paths, min(args.count, len(image_paths)))

    interpreter = tf.lite.Interpreter(model_content=model_path.read_bytes())
    interpreter.allocate_tensors()

    for idx, image_path in enumerate(sample_paths, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        orig_bgr, resized_rgb = load_and_resize(image_path, args.image_size)
        output = run_tflite(interpreter, resized_rgb)
        gt_points = gt_points_from_label(label_path, args.image_size, class_names)
        pred_points = pred_points_from_output(output, args.threshold, args.image_size, class_names)

        resized_bgr = cv2.resize(orig_bgr, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        vis = draw_overlay(resized_bgr, gt_points, pred_points, class_names)
        out_path = out_dir / f"{idx:02d}_{image_path.stem}.jpg"
        ok, encoded = cv2.imencode(".jpg", vis)
        if ok:
            out_path.write_bytes(encoded.tobytes())

    print(f"saved {len(sample_paths)} images to: {out_dir}")


if __name__ == "__main__":
    main()
