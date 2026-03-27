import json
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import tensorflow as tf


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_yolo_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    if (not path.exists()) or path.stat().st_size == 0:
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cid, cx, cy, w, h = line.split()
        rows.append((int(cid), float(cx), float(cy), float(w), float(h)))
    return rows


def collect_image_paths(image_dir: Path) -> List[Path]:
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(ext))
    return sorted(image_paths)


def resize_rgb_image(image_path: Path, image_size: int) -> np.ndarray:
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def yolo_box_to_pixel_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    width: int,
    height: int,
    expand_ratio: float = 0.0,
) -> Tuple[int, int, int, int]:
    half_w = (w * width) / 2.0
    half_h = (h * height) / 2.0
    expand_w = half_w * expand_ratio
    expand_h = half_h * expand_ratio
    x0 = int(np.floor(max(0.0, (cx * width) - half_w - expand_w)))
    y0 = int(np.floor(max(0.0, (cy * height) - half_h - expand_h)))
    x1 = int(np.ceil(min(float(width), (cx * width) + half_w + expand_w)))
    y1 = int(np.ceil(min(float(height), (cy * height) + half_h + expand_h)))
    return x0, y0, x1, y1


def box_overlap_area(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    return int(inter_w * inter_h)


def crop_is_background(
    crop_box: Tuple[int, int, int, int],
    gt_boxes: Sequence[Tuple[int, int, int, int]],
) -> bool:
    return all(box_overlap_area(crop_box, gt_box) == 0 for gt_box in gt_boxes)


def oversample_focus_samples(
    x: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    focus_classes_text: str,
    focus_multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if (not focus_classes_text.strip()) or focus_multiplier <= 1:
        return x, y, 0

    name_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    focus_ids = []
    for name in [s.strip() for s in focus_classes_text.split(",") if s.strip()]:
        if name in name_to_idx:
            focus_ids.append(name_to_idx[name])

    if not focus_ids:
        return x, y, 0

    mask = np.zeros((y.shape[0],), dtype=bool)
    for cls_id in focus_ids:
        mask |= np.any(y == cls_id, axis=(1, 2))

    if not np.any(mask):
        return x, y, 0

    focus_x = x[mask]
    focus_y = y[mask]
    x_parts = [x]
    y_parts = [y]
    for _ in range(focus_multiplier - 1):
        x_parts.append(focus_x.copy())
        y_parts.append(focus_y.copy())

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x_all.shape[0])
    return x_all[idx], y_all[idx], int(focus_x.shape[0] * (focus_multiplier - 1))


def propose_hard_negative_crop(
    box: Tuple[int, int, int, int],
    side: int,
    width: int,
    height: int,
    rng: np.random.Generator,
    gap_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    gap = max(4, int(round(side * gap_ratio)))
    candidates = []

    left_x = x0 - side - gap
    right_x = x1 + gap
    top_y = y0 - side - gap
    bottom_y = y1 + gap

    if left_x >= 0:
        low_y = max(0, min(y0, height - side))
        high_y = max(0, min(y1 - side, height - side))
        if high_y >= low_y:
            y = int(rng.integers(low_y, high_y + 1))
            candidates.append((left_x, y, left_x + side, y + side))

    if right_x + side <= width:
        low_y = max(0, min(y0, height - side))
        high_y = max(0, min(y1 - side, height - side))
        if high_y >= low_y:
            y = int(rng.integers(low_y, high_y + 1))
            candidates.append((right_x, y, right_x + side, y + side))

    if top_y >= 0:
        low_x = max(0, min(x0, width - side))
        high_x = max(0, min(x1 - side, width - side))
        if high_x >= low_x:
            x = int(rng.integers(low_x, high_x + 1))
            candidates.append((x, top_y, x + side, top_y + side))

    if bottom_y + side <= height:
        low_x = max(0, min(x0, width - side))
        high_x = max(0, min(x1 - side, width - side))
        if high_x >= low_x:
            x = int(rng.integers(low_x, high_x + 1))
            candidates.append((x, bottom_y, x + side, bottom_y + side))

    if not candidates:
        return None

    return candidates[int(rng.integers(0, len(candidates)))]


def synthesize_negative_crops(
    image_dir: Path,
    label_dir: Path,
    image_size: int,
    grid_size: int,
    negative_crops_per_image: int,
    seed: int,
    min_crop_scale: float,
    max_crop_scale: float,
    avoid_box_expand_ratio: float,
    hard_negative_ratio: float,
    hard_negative_gap_ratio: float,
    max_attempts_per_crop: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    if negative_crops_per_image <= 0:
        return (
            np.zeros((0, image_size, image_size, 3), dtype=np.float32),
            np.zeros((0, grid_size, grid_size), dtype=np.int32),
        )

    rng = np.random.default_rng(seed)
    x_neg = []
    y_neg = []

    for image_path in collect_image_paths(image_dir):
        img_bgr = imread_unicode(image_path)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        short_side = min(width, height)
        if short_side < 16:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        gt_boxes = [
            yolo_box_to_pixel_xyxy(
                cx,
                cy,
                w,
                h,
                width=width,
                height=height,
                expand_ratio=avoid_box_expand_ratio,
            )
            for _, cx, cy, w, h in parse_yolo_file(label_path)
        ]

        min_side = max(image_size, int(round(short_side * min_crop_scale)))
        max_side = max(min_side, int(round(short_side * max_crop_scale)))

        created = 0
        for _ in range(negative_crops_per_image):
            crop_box = None
            for _attempt in range(max_attempts_per_crop):
                side = int(rng.integers(min_side, max_side + 1))
                side = min(side, width, height)
                if side < image_size:
                    continue

                candidate = None
                if gt_boxes and rng.random() < hard_negative_ratio:
                    ref_box = gt_boxes[int(rng.integers(0, len(gt_boxes)))]
                    candidate = propose_hard_negative_crop(
                        ref_box,
                        side=side,
                        width=width,
                        height=height,
                        rng=rng,
                        gap_ratio=hard_negative_gap_ratio,
                    )

                if candidate is None:
                    max_x = max(0, width - side)
                    max_y = max(0, height - side)
                    x = int(rng.integers(0, max_x + 1))
                    y = int(rng.integers(0, max_y + 1))
                    candidate = (x, y, x + side, y + side)

                if crop_is_background(candidate, gt_boxes):
                    crop_box = candidate
                    break

            if crop_box is None:
                continue

            x0, y0, x1, y1 = crop_box
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
            x_neg.append(crop.astype(np.float32))
            y_neg.append(np.zeros((grid_size, grid_size), dtype=np.int32))
            created += 1

        if created == 0 and not gt_boxes:
            fallback = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            x_neg.append(fallback.astype(np.float32))
            y_neg.append(np.zeros((grid_size, grid_size), dtype=np.int32))

    if not x_neg:
        return (
            np.zeros((0, image_size, image_size, 3), dtype=np.float32),
            np.zeros((0, grid_size, grid_size), dtype=np.int32),
        )

    return np.stack(x_neg), np.stack(y_neg)


def bbox_to_grid_range(
    cx: float,
    cy: float,
    w: float,
    h: float,
    grid_size: int,
) -> Tuple[int, int, int, int]:
    x0 = float(np.clip(cx - (w / 2.0), 0.0, 0.9999))
    y0 = float(np.clip(cy - (h / 2.0), 0.0, 0.9999))
    x1 = float(np.clip(cx + (w / 2.0), 0.0, 0.9999))
    y1 = float(np.clip(cy + (h / 2.0), 0.0, 0.9999))

    gx0 = min(int(np.floor(x0 * grid_size)), grid_size - 1)
    gy0 = min(int(np.floor(y0 * grid_size)), grid_size - 1)
    gx1 = min(int(np.floor(x1 * grid_size)), grid_size - 1)
    gy1 = min(int(np.floor(y1 * grid_size)), grid_size - 1)
    return gx0, gy0, gx1, gy1


def bbox_to_grid_cells(
    cx: float,
    cy: float,
    w: float,
    h: float,
    grid_size: int,
) -> Set[Tuple[int, int]]:
    gx0, gy0, gx1, gy1 = bbox_to_grid_range(cx, cy, w, h, grid_size)
    cells = set()
    for gy in range(gy0, gy1 + 1):
        for gx in range(gx0, gx1 + 1):
            cells.add((gx, gy))

    center_gx = min(int(np.clip(cx, 0.0, 0.9999) * grid_size), grid_size - 1)
    center_gy = min(int(np.clip(cy, 0.0, 0.9999) * grid_size), grid_size - 1)
    cells.add((center_gx, center_gy))
    return cells


def build_point_target_map(label_path: Path, grid_size: int, num_classes: int) -> np.ndarray:
    target = np.zeros((grid_size, grid_size), dtype=np.int32)
    for cid, cx, cy, _, _ in parse_yolo_file(label_path):
        if not (0 <= cid < num_classes):
            continue
        gx = min(int(np.clip(cx, 0.0, 0.9999) * grid_size), grid_size - 1)
        gy = min(int(np.clip(cy, 0.0, 0.9999) * grid_size), grid_size - 1)
        target[gy, gx] = cid + 1
    return target


def load_training_arrays(
    image_dir: Path,
    label_dir: Path,
    image_size: int,
    grid_size: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for image_path in collect_image_paths(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        xs.append(resize_rgb_image(image_path, image_size))
        ys.append(build_point_target_map(label_path, grid_size, num_classes))
    return np.stack(xs), np.stack(ys)


def augment_images_and_targets(
    x: np.ndarray,
    y: np.ndarray,
    aug_multiplier: int,
    seed: int,
    hflip_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if aug_multiplier <= 0:
        return x, y

    rng = np.random.default_rng(seed)
    x_aug = [x]
    y_aug = [y]

    for _ in range(aug_multiplier):
        out_x = x.copy()
        out_y = y.copy()
        for i in range(out_x.shape[0]):
            img = out_x[i]
            target = out_y[i]

            if hflip_prob > 0.0 and rng.random() < hflip_prob:
                img = np.ascontiguousarray(img[:, ::-1, :])
                target = np.ascontiguousarray(target[:, ::-1])

            alpha = float(rng.uniform(0.85, 1.20))
            beta = float(rng.uniform(-20.0, 20.0))
            img = np.clip((img * alpha) + beta, 0.0, 255.0)

            gamma = float(rng.uniform(0.85, 1.20))
            img = np.power(img / 255.0, gamma) * 255.0

            noise_std = float(rng.uniform(0.0, 8.0))
            if noise_std > 0.0:
                noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 255.0)

            if rng.random() < 0.30:
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
                img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)
                img = img_u8.astype(np.float32)

            out_x[i] = img.astype(np.float32)
            out_y[i] = target.astype(np.int32)

        x_aug.append(out_x)
        y_aug.append(out_y)

    x_all = np.concatenate(x_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)
    idx = rng.permutation(x_all.shape[0])
    return x_all[idx], y_all[idx]


def representative_dataset(xs: np.ndarray, count: int = 100):
    limit = min(int(count), int(xs.shape[0]))
    for i in range(limit):
        yield [np.expand_dims(xs[i].astype(np.float32), axis=0)]


def export_tflite_models(model: tf.keras.Model, x_train: np.ndarray, out_dir: Path) -> Tuple[Path, Path]:
    float32_path = out_dir / "fomo_official_float32.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float32_path.write_bytes(converter.convert())

    int8_path = out_dir / "fomo_official_int8.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_path.write_bytes(converter.convert())

    return float32_path, int8_path


def safe_model_path(path: Path) -> str:
    path_str = str(path)
    if all(ord(ch) < 128 for ch in path_str):
        return path_str

    temp_dir = Path(tempfile.gettempdir()) / "fomo_official_eval"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / path.name
    temp_path.write_bytes(path.read_bytes())
    return str(temp_path)


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
    return model(batch, training=False).numpy()[0].astype(np.float32)


def load_model_runner(model_path: Path) -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    suffix = model_path.suffix.lower()
    if suffix == ".tflite":
        interpreter = tf.lite.Interpreter(model_content=model_path.read_bytes())
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


def extract_fomo_detections(
    output: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    class_thresholds: Optional[np.ndarray],
    image_size: int,
) -> List[Dict[str, object]]:
    grid_h, grid_w, channels = output.shape
    if channels != len(class_names) + 1:
        raise ValueError(f"output channel mismatch: got {channels}, expected {len(class_names) + 1}")

    cell_w = image_size / float(grid_w)
    cell_h = image_size / float(grid_h)
    bg = output[..., 0]
    detections: List[Dict[str, object]] = []

    for cls_idx, class_name in enumerate(class_names):
        thr = threshold if class_thresholds is None else float(class_thresholds[cls_idx])
        channel = output[..., cls_idx + 1]
        mask = np.logical_and(channel >= thr, channel >= bg).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            continue

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp_id in range(1, count):
            comp_mask = labels == comp_id
            ys, xs = np.where(comp_mask)
            if xs.size == 0:
                continue

            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max())
            y1 = int(ys.max())
            score = float(np.mean(channel[comp_mask]))

            weights = channel[comp_mask]
            if float(np.sum(weights)) > 1e-6:
                cx_grid = float(np.average(xs + 0.5, weights=weights))
                cy_grid = float(np.average(ys + 0.5, weights=weights))
            else:
                cx_grid = float(np.mean(xs + 0.5))
                cy_grid = float(np.mean(ys + 0.5))

            grid_cells = {(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())}
            detections.append(
                {
                    "class_idx": cls_idx + 1,
                    "class_name": class_name,
                    "score": score,
                    "grid_cells": grid_cells,
                    "bbox_grid": [x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)],
                    "center_grid": [cx_grid, cy_grid],
                    "bbox_image": [
                        int(round(x0 * cell_w)),
                        int(round(y0 * cell_h)),
                        int(round((x1 - x0 + 1) * cell_w)),
                        int(round((y1 - y0 + 1) * cell_h)),
                    ],
                    "center_image": [float(cx_grid * cell_w), float(cy_grid * cell_h)],
                }
            )

    detections.sort(key=lambda row: float(row["score"]), reverse=True)
    return detections


def collect_gt_objects(
    label_path: Path,
    class_names: Sequence[str],
    grid_size: int,
    image_size: int,
) -> List[Dict[str, object]]:
    rows = []
    for cid, cx, cy, w, h in parse_yolo_file(label_path):
        if not (0 <= cid < len(class_names)):
            continue
        x = (cx - (w / 2.0)) * image_size
        y = (cy - (h / 2.0)) * image_size
        bbox = [
            int(round(max(0.0, x))),
            int(round(max(0.0, y))),
            int(round(max(1.0, w * image_size))),
            int(round(max(1.0, h * image_size))),
        ]
        rows.append(
            {
                "class_idx": cid + 1,
                "class_name": class_names[cid],
                "grid_cells": bbox_to_grid_cells(cx, cy, w, h, grid_size),
                "center_grid": [
                    float(np.clip(cx, 0.0, 0.9999) * grid_size),
                    float(np.clip(cy, 0.0, 0.9999) * grid_size),
                ],
                "bbox_image": bbox,
                "center_image": [float(cx * image_size), float(cy * image_size)],
            }
        )
    return rows


def build_match_candidates(
    gt_objects: Sequence[Dict[str, object]],
    pred_objects: Sequence[Dict[str, object]],
) -> List[Tuple[float, int, int]]:
    candidates = []
    for gt_idx, gt in enumerate(gt_objects):
        gt_cells = gt["grid_cells"]
        for pred_idx, pred in enumerate(pred_objects):
            pred_cells = pred["grid_cells"]
            overlap = len(gt_cells & pred_cells)
            if overlap == 0:
                continue
            union = len(gt_cells | pred_cells)
            score = overlap / max(union, 1)
            score += float(pred["score"]) * 1e-3
            candidates.append((score, gt_idx, pred_idx))
    candidates.sort(reverse=True)
    return candidates


def confusion_matrix_from_objects(
    gt_objects: Sequence[Dict[str, object]],
    pred_objects: Sequence[Dict[str, object]],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    matched_gt = set()
    matched_pred = set()

    for _, gt_idx, pred_idx in build_match_candidates(gt_objects, pred_objects):
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        gt_class = int(gt_objects[gt_idx]["class_idx"])
        pred_class = int(pred_objects[pred_idx]["class_idx"])
        matrix[gt_class, pred_class] += 1

    for gt_idx, gt in enumerate(gt_objects):
        if gt_idx not in matched_gt:
            matrix[int(gt["class_idx"]), 0] += 1

    for pred_idx, pred in enumerate(pred_objects):
        if pred_idx not in matched_pred:
            matrix[0, int(pred["class_idx"])] += 1

    return matrix


def summarize_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    row_labels = ["background"] + list(class_names)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix.astype(np.float32),
        np.maximum(row_totals, 1),
        out=np.zeros_like(matrix, dtype=np.float32),
        where=np.maximum(row_totals, 1) > 0,
    )

    per_class = []
    for class_idx, class_name in enumerate(class_names, start=1):
        tp = int(matrix[class_idx, class_idx])
        fp = int(np.sum(matrix[:, class_idx]) - tp)
        fn = int(np.sum(matrix[class_idx, :]) - tp)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        per_class.append(
            {
                "class_name": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    tp_non_background = int(np.trace(matrix[1:, 1:]))
    fp_non_background = int(np.sum(matrix[:, 1:]) - tp_non_background)
    fn_non_background = int(np.sum(matrix[1:, :]) - tp_non_background)
    precision_non_background = tp_non_background / (tp_non_background + fp_non_background + 1e-9)
    recall_non_background = tp_non_background / (tp_non_background + fn_non_background + 1e-9)
    f1_non_background = (
        2 * precision_non_background * recall_non_background /
        (precision_non_background + recall_non_background + 1e-9)
    )

    return {
        "row_labels": row_labels,
        "matrix_counts": matrix.tolist(),
        "matrix_percent": np.round(row_percentages * 100.0, 2).tolist(),
        "per_class": per_class,
        "metrics_non_background": {
            "precision": precision_non_background,
            "recall": recall_non_background,
            "f1": f1_non_background,
            "tp": tp_non_background,
            "fp": fp_non_background,
            "fn": fn_non_background,
        },
    }


def print_confusion_summary(split_name: str, summary: Dict[str, object]) -> None:
    print(f"{split_name} confusion matrix (% by actual class):")
    row_labels = summary["row_labels"]
    percent = summary["matrix_percent"]
    header = "actual/pred".ljust(18) + "".join(label.ljust(18) for label in row_labels)
    print(header)
    for label, row in zip(row_labels, percent):
        values = "".join(f"{value:.2f}%".ljust(18) for value in row)
        print(label.ljust(18) + values)
    print("")
    nb = summary["metrics_non_background"]
    print(
        f"{split_name} non-background: "
        f"precision={nb['precision']:.4f} "
        f"recall={nb['recall']:.4f} "
        f"f1={nb['f1']:.4f} "
        f"tp={nb['tp']} fp={nb['fp']} fn={nb['fn']}"
    )
    for row in summary["per_class"]:
        print(
            f"  {row['class_name']}: "
            f"precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} "
            f"f1={row['f1']:.4f} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']}"
        )
    print("")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
