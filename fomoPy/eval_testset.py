import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import yaml

from train import parse_yolo_file


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def windows_short_path(path: Path) -> str:
    path_str = str(path)
    if os.name != "nt":
        return path_str

    try:
        import ctypes

        buffer_len = 4096
        buffer = ctypes.create_unicode_buffer(buffer_len)
        result = ctypes.windll.kernel32.GetShortPathNameW(path_str, buffer, buffer_len)
        if result > 0:
            return buffer.value
    except Exception:
        pass

    return path_str


def safe_model_path(path: Path) -> str:
    path_str = windows_short_path(path)
    if all(ord(ch) < 128 for ch in path_str):
        return path_str

    temp_dir = Path(tempfile.gettempdir()) / "fomo_eval_ascii"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / path.name
    shutil.copy2(path, temp_path)
    return str(temp_path)


def load_image(image_path: Path, image_size: int) -> np.ndarray:
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def build_gt_cells(label_path: Path, grid_size: int, num_classes: int) -> np.ndarray:
    gt = np.zeros((grid_size, grid_size, num_classes), dtype=np.uint8)
    for cid, cx, cy, _, _ in parse_yolo_file(label_path):
        cx = float(np.clip(cx, 0.0, 0.9999))
        cy = float(np.clip(cy, 0.0, 0.9999))
        gx = min(int(cx * grid_size), grid_size - 1)
        gy = min(int(cy * grid_size), grid_size - 1)
        if 0 <= cid < num_classes:
            gt[gy, gx, cid] = 1
    return gt


def run_tflite(interpreter: tf.lite.Interpreter, image: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    batch = np.expand_dims(image, axis=0)
    input_dtype = input_detail["dtype"]
    if input_dtype == np.int8:
        scale, zero_point = input_detail["quantization"]
        batch = np.round(batch / scale + zero_point).astype(np.int8)
    else:
        batch = batch.astype(input_dtype)

    interpreter.set_tensor(input_detail["index"], batch)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]

    if output_detail["dtype"] == np.int8:
        scale, zero_point = output_detail["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
    else:
        output = output.astype(np.float32)

    return output


def run_keras(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    output = model(batch, training=False).numpy()[0]
    return output.astype(np.float32)


def load_model_runner(model_path: Path) -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    suffix = model_path.suffix.lower()

    if suffix == ".tflite":
        model_content = model_path.read_bytes()
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        return "tflite", lambda image: run_tflite(interpreter, image)

    if suffix in {".keras", ".h5", ".hdf5"}:
        model = tf.keras.models.load_model(safe_model_path(model_path), compile=False)
        return "keras", lambda image: run_keras(model, image)

    raise ValueError(f"unsupported model format: {model_path}")


def logits_to_pred_cells(output: np.ndarray, threshold: float) -> np.ndarray:
    # Output layout: [grid_h, grid_w, num_classes + 1], channel 0 is background.
    fg = output[..., 1:]
    pred = (fg >= threshold).astype(np.uint8)
    return pred


def compute_metrics(gt_all: np.ndarray, pred_all: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((gt_all == 1) & (pred_all == 1)))
    fp = int(np.sum((gt_all == 0) & (pred_all == 1)))
    fn = int(np.sum((gt_all == 1) & (pred_all == 0)))
    tn = int(np.sum((gt_all == 0) & (pred_all == 0)))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def per_class_metrics(gt_all: np.ndarray, pred_all: np.ndarray, class_names: List[str]) -> List[Dict[str, float]]:
    rows = []
    for i, name in enumerate(class_names):
        m = compute_metrics(gt_all[..., i], pred_all[..., i])
        m["class_name"] = name
        rows.append(m)
    return rows


def evaluate_split(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    image_dir: Path,
    label_dir: Path,
    class_names: List[str],
    num_classes: int,
    image_size: int,
    grid_size: int,
    threshold: float,
) -> Dict[str, object]:
    gt_list = []
    pred_list = []
    image_rows = []

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(ext))
    image_paths = sorted(image_paths)

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = load_image(image_path, image_size)
        output = predict_fn(image)
        pred = logits_to_pred_cells(output, threshold)
        gt = build_gt_cells(label_path, grid_size, num_classes)

        gt_list.append(gt)
        pred_list.append(pred)

        row = compute_metrics(gt, pred)
        row["image_name"] = image_path.name
        row["gt_objects"] = int(np.sum(gt))
        row["pred_objects"] = int(np.sum(pred))
        image_rows.append(row)

    gt_all = np.stack(gt_list)
    pred_all = np.stack(pred_list)

    return {
        "num_images": len(image_rows),
        "overall": compute_metrics(gt_all, pred_all),
        "per_class": per_class_metrics(gt_all, pred_all, class_names),
    }


def print_split_result(split_name: str, result: Dict[str, object]) -> None:
    overall = result["overall"]
    per_class = result["per_class"]
    print(f"{split_name} images: {result['num_images']}")
    print(f"{split_name} cell accuracy: {overall['accuracy']:.4f}")
    print(f"{split_name} precision: {overall['precision']:.4f}")
    print(f"{split_name} recall: {overall['recall']:.4f}")
    print(f"{split_name} f1: {overall['f1']:.4f}")
    print(f"{split_name} tp: {overall['tp']}  fp: {overall['fp']}  fn: {overall['fn']}  tn: {overall['tn']}")
    print("")
    print(f"{split_name} per class:")
    for row in per_class:
        print(
            f"  {row['class_name']}: "
            f"acc={row['accuracy']:.4f} "
            f"prec={row['precision']:.4f} "
            f"rec={row['recall']:.4f} "
            f"f1={row['f1']:.4f} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']}"
        )
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FOMO TFLite model on the testing split")
    parser.add_argument("--model", type=str, default="fomoPy/outputs/fomo_local/fomo_like_int8.tflite")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--report", type=str, default="fomoPy/outputs/fomo_local/eval_metrics.json")
    args = parser.parse_args()

    workspace = Path.cwd()
    model_path = workspace / args.model
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir
    report_path = workspace / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() else (workspace / cfg["path"]).resolve()
    train_img_dir = dataset_root / cfg["train"]
    test_img_dir = dataset_root / cfg["val"]
    train_label_dir = labels_root / "training"
    test_label_dir = labels_root / "testing"
    class_names = list(cfg["names"])
    num_classes = int(cfg["nc"])

    model_format, predict_fn = load_model_runner(model_path)

    train_result = evaluate_split(
        predict_fn=predict_fn,
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        class_names=class_names,
        num_classes=num_classes,
        image_size=args.image_size,
        grid_size=args.grid_size,
        threshold=args.threshold,
    )
    test_result = evaluate_split(
        predict_fn=predict_fn,
        image_dir=test_img_dir,
        label_dir=test_label_dir,
        class_names=class_names,
        num_classes=num_classes,
        image_size=args.image_size,
        grid_size=args.grid_size,
        threshold=args.threshold,
    )

    report = {
        "model": str(model_path),
        "model_format": model_format,
        "threshold": args.threshold,
        "metric_note": "accuracy is grid-cell accuracy; precision/recall/f1 are usually more meaningful for FOMO-style detection",
        "train": train_result,
        "test": test_result,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_split_result("train", train_result)
    print_split_result("test", test_result)
    print(f"report saved to: {report_path}")


if __name__ == "__main__":
    main()
