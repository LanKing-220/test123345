import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    from fomoPy.local_fomo_common import collect_image_paths, parse_yolo_file, resize_rgb_image
except ImportError:
    from local_fomo_common import collect_image_paths, parse_yolo_file, resize_rgb_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def validate_grid_layout(image_size: int, grid_size: int) -> int:
    if image_size % grid_size != 0:
        raise ValueError(f"image_size ({image_size}) must be divisible by grid_size ({grid_size})")

    downsample_factor = image_size // grid_size
    if downsample_factor < 2 or (downsample_factor & (downsample_factor - 1)) != 0:
        raise ValueError(
            "image_size / grid_size must be a power of two and at least 2 "
            f"(got {image_size} / {grid_size} = {downsample_factor})"
        )

    return int(np.log2(downsample_factor))


def derive_stage_filters(downsample_steps: int, width_scale: float = 1.0) -> List[int]:
    base_filters = [8, 16, 24, 32, 48]
    if downsample_steps > len(base_filters):
        raise ValueError(
            f"unsupported grid layout: need {downsample_steps} downsampling stages, "
            f"but only {len(base_filters)} are configured"
        )
    scaled = []
    for value in base_filters[:downsample_steps]:
        scaled_value = max(8, int(round(value * float(width_scale))))
        scaled.append(scaled_value)
    return scaled


def compute_target_radius(
    object_span_in_cells: float,
    radius_scale: float,
    min_target_radius: int,
    max_target_radius: int,
) -> int:
    raw_radius = max(0.0, object_span_in_cells - 1.0) * float(radius_scale)
    radius = int(np.ceil(raw_radius))
    return int(np.clip(radius, min_target_radius, max_target_radius))


def assign_object_to_target(
    target: np.ndarray,
    cid: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    grid_size: int,
    num_classes: int,
    label_mode: str,
    bbox_radius_scale: float,
    min_target_radius: int,
    max_target_radius: int,
) -> None:
    if not (0 <= cid < num_classes):
        return

    cx = float(np.clip(cx, 0.0, 0.9999))
    cy = float(np.clip(cy, 0.0, 0.9999))
    gx = min(int(cx * grid_size), grid_size - 1)
    gy = min(int(cy * grid_size), grid_size - 1)
    channel = cid + 1

    if label_mode == "point":
        target[gy, gx, channel] = 1.0
        return

    span_x = max(float(w) * grid_size, 1.0)
    span_y = max(float(h) * grid_size, 1.0)
    rx = compute_target_radius(span_x, bbox_radius_scale, min_target_radius, max_target_radius)
    ry = compute_target_radius(span_y, bbox_radius_scale, min_target_radius, max_target_radius)

    x0 = max(0, gx - rx)
    x1 = min(grid_size - 1, gx + rx)
    y0 = max(0, gy - ry)
    y1 = min(grid_size - 1, gy + ry)

    for yy in range(y0, y1 + 1):
        wy = 1.0 if ry == 0 else max(0.0, 1.0 - (abs(yy - gy) / float(ry + 1)))
        for xx in range(x0, x1 + 1):
            wx = 1.0 if rx == 0 else max(0.0, 1.0 - (abs(xx - gx) / float(rx + 1)))
            score = float(wx * wy)
            if score > target[yy, xx, channel]:
                target[yy, xx, channel] = score

    target[gy, gx, channel] = 1.0


def load_dataset(
    image_dir: Path,
    label_dir: Path,
    image_size: int,
    grid_size: int,
    num_classes: int,
    label_mode: str,
    bbox_radius_scale: float,
    min_target_radius: int,
    max_target_radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for image_path in tqdm(collect_image_paths(image_dir), desc=f"Loading {image_dir.name}"):
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = parse_yolo_file(label_path)

        img = resize_rgb_image(image_path, image_size=image_size, normalize=True)
        target = np.zeros((grid_size, grid_size, num_classes + 1), dtype=np.float32)

        for cid, cx, cy, w, h in labels:
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

        fg = np.clip(np.max(target[:, :, 1:], axis=-1), 0.0, 1.0)
        target[:, :, 0] = 1.0 - fg

        xs.append(img)
        ys.append(target)

    return np.stack(xs), np.stack(ys)


def build_fomo_like_model(
    image_size: int,
    num_classes: int,
    stage_filters: Iterable[int],
    head_filters: int,
    refine_blocks: int,
    dropout_rate: float,
) -> tf.keras.Model:
    def squeeze_excite(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        reduced = max(4, filters // 4)
        scale = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
        scale = tf.keras.layers.Conv2D(reduced, 1, activation="relu", padding="same", name=f"{name}_reduce")(scale)
        scale = tf.keras.layers.Conv2D(filters, 1, activation="sigmoid", padding="same", name=f"{name}_expand")(scale)
        return tf.keras.layers.Multiply(name=f"{name}_mul")([x, scale])

    def residual_refine_block(x: tf.Tensor, filters: int, block_name: str) -> tf.Tensor:
        shortcut = x
        y = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, name=f"{block_name}_sep1")(x)
        y = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(y)
        y = tf.keras.layers.ReLU(name=f"{block_name}_relu1")(y)
        y = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False, name=f"{block_name}_sep2")(y)
        y = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(y)
        y = squeeze_excite(y, filters=filters, name=f"{block_name}_se")
        if dropout_rate > 0.0:
            y = tf.keras.layers.SpatialDropout2D(dropout_rate, name=f"{block_name}_drop")(y)
        y = tf.keras.layers.Add(name=f"{block_name}_add")([shortcut, y])
        return tf.keras.layers.ReLU(name=f"{block_name}_out")(y)

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    stage_filters = list(stage_filters)
    if not stage_filters:
        raise ValueError("stage_filters must not be empty")

    first_filters = int(stage_filters[0])
    x = tf.keras.layers.Conv2D(first_filters, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for block_idx in range(refine_blocks):
        x = residual_refine_block(x, filters=first_filters, block_name=f"stage1_refine{block_idx + 1}")

    for stage_idx, filters in enumerate(stage_filters[1:], start=2):
        x = tf.keras.layers.SeparableConv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        for block_idx in range(refine_blocks):
            x = residual_refine_block(x, filters=int(filters), block_name=f"stage{stage_idx}_refine{block_idx + 1}")

    x = tf.keras.layers.SeparableConv2D(head_filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for block_idx in range(refine_blocks):
        x = residual_refine_block(x, filters=int(head_filters), block_name=f"head_refine{block_idx + 1}")

    outputs = tf.keras.layers.Conv2D(num_classes + 1, 1, activation="sigmoid", name="fomo_head")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="fomo_like")


def representative_dataset(xs: np.ndarray, count: int = 100):
    limit = min(int(count), int(xs.shape[0]))
    for i in range(limit):
        yield [np.expand_dims(xs[i].astype(np.float32), axis=0)]


def export_tflite_models(model: tf.keras.Model, x_train: np.ndarray, out_dir: Path) -> Tuple[Path, Path]:
    float32_path = out_dir / "fomo_like_float32.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float32_path.write_bytes(converter.convert())

    int8_path = out_dir / "fomo_like_int8.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_path.write_bytes(converter.convert())
    return float32_path, int8_path


def build_fomo_focal_loss(
    bg_weight: float = 0.25,
    fg_weight: float = 2.0,
    gamma: float = 2.0,
    fg_channel_weights: Optional[np.ndarray] = None,
    dice_weight: float = 0.0,
    dice_smooth: float = 1e-4,
):
    fg_channel_weights_np = None
    if fg_channel_weights is not None:
        fg_channel_weights_np = np.asarray(fg_channel_weights, dtype=np.float32)

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal = tf.pow(1.0 - pt, gamma)

        if fg_channel_weights_np is None:
            ch = tf.shape(y_true)[-1]
            fg = tf.fill([ch - 1], tf.cast(fg_weight, tf.float32))
        else:
            fg = tf.convert_to_tensor(fg_channel_weights_np * np.float32(fg_weight), dtype=tf.float32)

        weights = tf.concat([[tf.cast(bg_weight, tf.float32)], fg], axis=0)
        weights = tf.reshape(weights, [1, 1, 1, -1])
        focal_bce = tf.reduce_mean(bce * focal * weights)

        if dice_weight <= 0.0:
            return focal_bce

        fg_true = y_true[..., 1:]
        fg_pred = y_pred[..., 1:]
        intersection = tf.reduce_sum(fg_true * fg_pred, axis=(0, 1, 2))
        denominator = tf.reduce_sum(fg_true, axis=(0, 1, 2)) + tf.reduce_sum(fg_pred, axis=(0, 1, 2))
        dice_score = (2.0 * intersection + dice_smooth) / (denominator + dice_smooth)
        dice_loss = 1.0 - dice_score

        if fg_channel_weights_np is not None:
            dice_channel_weights = tf.convert_to_tensor(fg_channel_weights_np, dtype=tf.float32)
            dice_channel_weights = dice_channel_weights / tf.reduce_mean(dice_channel_weights)
            dice_loss = tf.reduce_sum(dice_loss * dice_channel_weights) / tf.reduce_sum(dice_channel_weights)
        else:
            dice_loss = tf.reduce_mean(dice_loss)

        return focal_bce + tf.cast(dice_weight, tf.float32) * dice_loss

    return _loss


def _metric_name(prefix: str, threshold: float, gt_threshold: float) -> str:
    threshold_tag = int(round(float(threshold) * 100))
    gt_tag = int(round(float(gt_threshold) * 100))
    return f"{prefix}_t{threshold_tag:02d}_g{gt_tag:02d}"


@tf.keras.utils.register_keras_serializable(package="fomo")
class ForegroundPrecisionMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold: float, gt_threshold: float = 0.5, name: Optional[str] = None, **kwargs):
        metric_name = name or _metric_name("fg_precision", threshold, gt_threshold)
        super().__init__(name=metric_name, **kwargs)
        self.threshold = float(threshold)
        self.gt_threshold = float(gt_threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true_fg = tf.cast(y_true[..., 1:] >= self.gt_threshold, self.dtype)
        y_pred_fg = tf.cast(y_pred[..., 1:] >= self.threshold, self.dtype)
        self.tp.assign_add(tf.reduce_sum(y_pred_fg * y_true_fg))
        self.fp.assign_add(tf.reduce_sum(y_pred_fg * (1.0 - y_true_fg)))

    def result(self) -> tf.Tensor:
        return self.tp / (self.tp + self.fp + tf.cast(1e-6, self.dtype))

    def reset_state(self) -> None:
        self.tp.assign(0.0)
        self.fp.assign(0.0)

    def get_config(self) -> Dict[str, float]:
        config = super().get_config()
        config.update({"threshold": self.threshold, "gt_threshold": self.gt_threshold})
        return config


@tf.keras.utils.register_keras_serializable(package="fomo")
class ForegroundRecallMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold: float, gt_threshold: float = 0.5, name: Optional[str] = None, **kwargs):
        metric_name = name or _metric_name("fg_recall", threshold, gt_threshold)
        super().__init__(name=metric_name, **kwargs)
        self.threshold = float(threshold)
        self.gt_threshold = float(gt_threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true_fg = tf.cast(y_true[..., 1:] >= self.gt_threshold, self.dtype)
        y_pred_fg = tf.cast(y_pred[..., 1:] >= self.threshold, self.dtype)
        self.tp.assign_add(tf.reduce_sum(y_pred_fg * y_true_fg))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_pred_fg) * y_true_fg))

    def result(self) -> tf.Tensor:
        return self.tp / (self.tp + self.fn + tf.cast(1e-6, self.dtype))

    def reset_state(self) -> None:
        self.tp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self) -> Dict[str, float]:
        config = super().get_config()
        config.update({"threshold": self.threshold, "gt_threshold": self.gt_threshold})
        return config


@tf.keras.utils.register_keras_serializable(package="fomo")
class ForegroundF1Metric(tf.keras.metrics.Metric):
    def __init__(self, threshold: float, gt_threshold: float = 0.5, name: Optional[str] = None, **kwargs):
        metric_name = name or _metric_name("fg_f1", threshold, gt_threshold)
        super().__init__(name=metric_name, **kwargs)
        self.threshold = float(threshold)
        self.gt_threshold = float(gt_threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true_fg = tf.cast(y_true[..., 1:] >= self.gt_threshold, self.dtype)
        y_pred_fg = tf.cast(y_pred[..., 1:] >= self.threshold, self.dtype)
        self.tp.assign_add(tf.reduce_sum(y_pred_fg * y_true_fg))
        self.fp.assign_add(tf.reduce_sum(y_pred_fg * (1.0 - y_true_fg)))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_pred_fg) * y_true_fg))

    def result(self) -> tf.Tensor:
        precision = self.tp / (self.tp + self.fp + tf.cast(1e-6, self.dtype))
        recall = self.tp / (self.tp + self.fn + tf.cast(1e-6, self.dtype))
        return 2.0 * precision * recall / (precision + recall + tf.cast(1e-6, self.dtype))

    def reset_state(self) -> None:
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self) -> Dict[str, float]:
        config = super().get_config()
        config.update({"threshold": self.threshold, "gt_threshold": self.gt_threshold})
        return config


def make_fg_precision_metric(threshold: float, gt_threshold: float = 0.5):
    return ForegroundPrecisionMetric(threshold=threshold, gt_threshold=gt_threshold)


def make_fg_recall_metric(threshold: float, gt_threshold: float = 0.5):
    return ForegroundRecallMetric(threshold=threshold, gt_threshold=gt_threshold)


def make_fg_f1_metric(threshold: float, gt_threshold: float = 0.5):
    return ForegroundF1Metric(threshold=threshold, gt_threshold=gt_threshold)


def compute_fg_binary_metrics(y_true_fg: np.ndarray, y_pred_fg: np.ndarray) -> Dict[str, float]:
    y_true_fg = np.asarray(y_true_fg, dtype=np.uint8)
    y_pred_fg = np.asarray(y_pred_fg, dtype=np.uint8)

    tp = int(np.sum((y_true_fg == 1) & (y_pred_fg == 1)))
    fp = int(np.sum((y_true_fg == 0) & (y_pred_fg == 1)))
    fn = int(np.sum((y_true_fg == 1) & (y_pred_fg == 0)))
    tn = int(np.sum((y_true_fg == 0) & (y_pred_fg == 0)))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _is_better_metric(candidate: Dict[str, float], incumbent: Optional[Dict[str, float]]) -> bool:
    if incumbent is None:
        return True

    candidate_key = (candidate["f1"], candidate["precision"], candidate["recall"])
    incumbent_key = (incumbent["f1"], incumbent["precision"], incumbent["recall"])
    return candidate_key > incumbent_key


def recommend_fg_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Sequence[float],
    gt_threshold: float,
    class_names: Sequence[str],
) -> Dict[str, object]:
    threshold_values = np.asarray(sorted({float(v) for v in thresholds}), dtype=np.float32)
    if threshold_values.size == 0:
        raise ValueError("thresholds must contain at least one value")

    fg_true = (np.asarray(y_true[..., 1:]) >= float(gt_threshold)).astype(np.uint8)
    fg_scores = np.asarray(y_pred[..., 1:], dtype=np.float32)

    if fg_true.shape != fg_scores.shape:
        raise ValueError(f"shape mismatch between y_true fg {fg_true.shape} and y_pred fg {fg_scores.shape}")

    best_global_threshold = None
    best_global_metrics = None
    for thr in threshold_values:
        pred = (fg_scores >= thr).astype(np.uint8)
        metrics = compute_fg_binary_metrics(fg_true, pred)
        if _is_better_metric(metrics, best_global_metrics):
            best_global_threshold = float(thr)
            best_global_metrics = metrics

    per_class_thresholds: List[float] = []
    per_class_rows: List[Dict[str, object]] = []
    for class_idx, class_name in enumerate(class_names):
        best_threshold = None
        best_metrics = None
        class_true = fg_true[..., class_idx]
        class_scores = fg_scores[..., class_idx]
        for thr in threshold_values:
            pred = (class_scores >= thr).astype(np.uint8)
            metrics = compute_fg_binary_metrics(class_true, pred)
            if _is_better_metric(metrics, best_metrics):
                best_threshold = float(thr)
                best_metrics = metrics

        per_class_thresholds.append(best_threshold)
        row = dict(best_metrics)
        row["class_name"] = class_name
        row["threshold"] = best_threshold
        per_class_rows.append(row)

    per_class_thresholds_np = np.asarray(per_class_thresholds, dtype=np.float32).reshape((1, 1, 1, -1))
    per_class_pred = (fg_scores >= per_class_thresholds_np).astype(np.uint8)
    per_class_overall = compute_fg_binary_metrics(fg_true, per_class_pred)

    return {
        "metric_gt_threshold": float(gt_threshold),
        "search_thresholds": [float(v) for v in threshold_values.tolist()],
        "global": {
            "threshold": float(best_global_threshold),
            **best_global_metrics,
        },
        "per_class": {
            "thresholds": [float(v) for v in per_class_thresholds],
            "overall": per_class_overall,
            "rows": per_class_rows,
        },
    }


def augment_photometric(
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
                target = np.ascontiguousarray(target[:, ::-1, :])

            alpha = float(rng.uniform(0.85, 1.20))
            beta = float(rng.uniform(-0.10, 0.10))
            img = np.clip((img * alpha) + beta, 0.0, 1.0)

            gamma = float(rng.uniform(0.85, 1.20))
            img = np.power(img, gamma)

            noise_std = float(rng.uniform(0.0, 0.03))
            if noise_std > 0.0:
                noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

            if rng.random() < 0.30:
                img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)
                img = img_u8.astype(np.float32) / 255.0

            out_x[i] = img.astype(np.float32)
            out_y[i] = target.astype(np.float32)

        x_aug.append(out_x)
        y_aug.append(out_y)

    x_all = np.concatenate(x_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)
    idx = rng.permutation(x_all.shape[0])
    return x_all[idx], y_all[idx]


def compute_auto_class_weights(
    y_train: np.ndarray,
    min_weight: float = 0.75,
    max_weight: float = 4.0,
) -> np.ndarray:
    fg = y_train[..., 1:]
    class_counts = np.sum(fg, axis=(0, 1, 2)).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    inv = np.sum(class_counts) / class_counts
    inv = inv / np.mean(inv)
    inv = np.clip(inv, min_weight, max_weight)
    return inv.astype(np.float32)


def oversample_focus_classes(
    x: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    focus_classes_text: str,
    focus_multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if (not focus_classes_text.strip()) or focus_multiplier <= 1:
        return x, y

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    selected = [name_to_idx[name] for name in [s.strip() for s in focus_classes_text.split(",") if s.strip()] if name in name_to_idx]
    if not selected:
        return x, y

    mask = np.zeros((y.shape[0],), dtype=bool)
    for cls_idx in selected:
        mask |= np.any(y[..., cls_idx + 1] > 0.5, axis=(1, 2))
    if not np.any(mask):
        return x, y

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
    return x_all[idx], y_all[idx]


def save_training_artifacts(history: tf.keras.callbacks.History, out_dir: Path, figure_path: Path) -> None:
    hist = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_path = out_dir / "training_history.json"
    history_path.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    epochs = list(range(1, len(hist.get("loss", [])) + 1))
    if not epochs:
        return

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(epochs, hist["loss"], color="#2563eb", linewidth=2.0, label="Training loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], color="#dc2626", linewidth=2.0, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss curve")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
