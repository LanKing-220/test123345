import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    from fomoPy.local_fomo_common import collect_image_paths, parse_yolo_file, resolve_dataset_layout, resize_rgb_image
except ImportError:
    from local_fomo_common import collect_image_paths, parse_yolo_file, resolve_dataset_layout, resize_rgb_image
try:
    from fomoPy.local_train_common import assign_object_to_target
except ImportError:
    from local_train_common import assign_object_to_target


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


def build_gt_cells(
    label_path: Path,
    grid_size: int,
    num_classes: int,
    label_mode: str,
    bbox_radius_scale: float,
    min_target_radius: int,
    max_target_radius: int,
    gt_positive_threshold: float,
) -> np.ndarray:
    target = np.zeros((grid_size, grid_size, num_classes + 1), dtype=np.float32)
    for cid, cx, cy, w, h in parse_yolo_file(label_path):
        assign_object_to_target(
            target=target,
            cid=cid,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            grid_size=grid_size,
            num_classes=num_classes,
            label_mode=label_mode,
            bbox_radius_scale=bbox_radius_scale,
            min_target_radius=min_target_radius,
            max_target_radius=max_target_radius,
        )
    return (target[..., 1:] >= gt_positive_threshold).astype(np.uint8)


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


def parse_class_thresholds(text: Optional[str], num_classes: int) -> Optional[np.ndarray]:
    if not text:
        return None
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != num_classes:
        raise ValueError(f"class thresholds count mismatch: expected {num_classes}, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def logits_to_pred_cells(output: np.ndarray, threshold: float, class_thresholds: Optional[np.ndarray]) -> np.ndarray:
    # Output layout: [grid_h, grid_w, num_classes + 1], channel 0 is background.
    fg = output[..., 1:]
    if class_thresholds is None:
        pred = (fg >= threshold).astype(np.uint8)
    else:
        thr = class_thresholds.reshape((1, 1, -1))
        pred = (fg >= thr).astype(np.uint8)
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
    class_thresholds: Optional[np.ndarray],
    label_mode: str,
    bbox_radius_scale: float,
    min_target_radius: int,
    max_target_radius: int,
    gt_positive_threshold: float,
) -> Dict[str, object]:
    gt_list = []
    pred_list = []
    image_rows = []

    for image_path in collect_image_paths(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = resize_rgb_image(image_path, image_size=image_size, normalize=True)
        output = predict_fn(image)
        pred = logits_to_pred_cells(output, threshold, class_thresholds)
        gt = build_gt_cells(
            label_path,
            grid_size,
            num_classes,
            label_mode=label_mode,
            bbox_radius_scale=bbox_radius_scale,
            min_target_radius=min_target_radius,
            max_target_radius=max_target_radius,
            gt_positive_threshold=gt_positive_threshold,
        )

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
    parser.add_argument("--class-thresholds", type=str, default="")
    parser.add_argument("--label-mode", type=str, choices=["point", "soft-box"], default="point")
    parser.add_argument("--bbox-radius-scale", type=float, default=0.25)
    parser.add_argument("--min-target-radius", type=int, default=0)
    parser.add_argument("--max-target-radius", type=int, default=3)
    parser.add_argument("--gt-positive-threshold", type=float, default=0.05)
    parser.add_argument("--report", type=str, default="fomoPy/outputs/fomo_local/eval_metrics.json")
    args = parser.parse_args()

    workspace = Path.cwd()

    # Resolve model, data_yaml, labels_root and report paths robustly to
    # avoid duplicated segments when running from inside `fomoPy`.
    def resolve_candidates(p: str) -> Path:
        p_path = Path(p)
        if p_path.is_absolute():
            return p_path
        candidates = [
            workspace / p_path,
            workspace.parent / p_path,
            Path.cwd() / p_path,
            Path(__file__).resolve().parent / p_path,
        ]
        for c in candidates:
            if c.exists():
                return c
        return (workspace / p_path)

    model_path = resolve_candidates(args.model)
    data_yaml = resolve_candidates(args.data_yaml)
    labels_root = resolve_candidates(args.labels_dir)
    report_path = resolve_candidates(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = resolve_dataset_layout(workspace, data_yaml, labels_root)

    train_img_dir = dataset.train_img_dir
    test_img_dir = dataset.val_img_dir
    train_label_dir = dataset.train_label_dir
    test_label_dir = dataset.val_label_dir
    class_names = dataset.class_names
    num_classes = dataset.num_classes
    class_thresholds = parse_class_thresholds(args.class_thresholds, num_classes)

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
        class_thresholds=class_thresholds,
        label_mode=args.label_mode,
        bbox_radius_scale=args.bbox_radius_scale,
        min_target_radius=args.min_target_radius,
        max_target_radius=args.max_target_radius,
        gt_positive_threshold=args.gt_positive_threshold,
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
        class_thresholds=class_thresholds,
        label_mode=args.label_mode,
        bbox_radius_scale=args.bbox_radius_scale,
        min_target_radius=args.min_target_radius,
        max_target_radius=args.max_target_radius,
        gt_positive_threshold=args.gt_positive_threshold,
    )

    report = {
        "model": str(model_path),
        "model_format": model_format,
        "threshold": args.threshold,
        "class_thresholds": class_thresholds.tolist() if class_thresholds is not None else None,
        "label_mode": args.label_mode,
        "bbox_radius_scale": args.bbox_radius_scale,
        "gt_positive_threshold": args.gt_positive_threshold,
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
