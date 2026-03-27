import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        if ry == 0:
            wy = 1.0
        else:
            wy = max(0.0, 1.0 - (abs(yy - gy) / float(ry + 1)))

        for xx in range(x0, x1 + 1):
            if rx == 0:
                wx = 1.0
            else:
                wx = max(0.0, 1.0 - (abs(xx - gx) / float(rx + 1)))

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
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(ext))
    image_paths = sorted(image_paths)

    xs = []
    ys = []

    for image_path in tqdm(image_paths, desc=f"Loading {image_dir.name}"):
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = parse_yolo_file(label_path)

        img = imread_unicode(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

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
        y = tf.keras.layers.ReLU(name=f"{block_name}_out")(y)
        return y

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
        axes = (0, 1, 2)
        intersection = tf.reduce_sum(fg_true * fg_pred, axis=axes)
        denominator = tf.reduce_sum(fg_true, axis=axes) + tf.reduce_sum(fg_pred, axis=axes)
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


def make_fg_precision_metric(threshold: float):
    def _metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_fg = tf.cast(y_true[..., 1:] >= 0.5, tf.float32)
        y_pred_fg = tf.cast(y_pred[..., 1:] >= threshold, tf.float32)
        tp = tf.reduce_sum(y_pred_fg * y_true_fg)
        fp = tf.reduce_sum(y_pred_fg * (1.0 - y_true_fg))
        return tp / (tp + fp + 1e-6)

    metric_tag = int(round(threshold * 100))
    _metric.__name__ = f"fg_precision_t{metric_tag:02d}"
    return _metric


def make_fg_recall_metric(threshold: float):
    def _metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_fg = tf.cast(y_true[..., 1:] >= 0.5, tf.float32)
        y_pred_fg = tf.cast(y_pred[..., 1:] >= threshold, tf.float32)
        tp = tf.reduce_sum(y_pred_fg * y_true_fg)
        fn = tf.reduce_sum((1.0 - y_pred_fg) * y_true_fg)
        return tp / (tp + fn + 1e-6)

    metric_tag = int(round(threshold * 100))
    _metric.__name__ = f"fg_recall_t{metric_tag:02d}"
    return _metric


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
    # y_train shape: [N, grid_h, grid_w, num_classes + 1], channel 0 is background.
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
    selected = []
    for name in [s.strip() for s in focus_classes_text.split(",") if s.strip()]:
        if name in name_to_idx:
            selected.append(name_to_idx[name])

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


def save_training_artifacts(
    history: tf.keras.callbacks.History,
    out_dir: Path,
    figure_path: Path,
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local FOMO-like model from Edge Impulse export")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug-multiplier", type=int, default=1)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bg-weight", type=float, default=0.25)
    parser.add_argument("--fg-weight", type=float, default=2.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--metric-threshold", type=float, default=0.35)
    parser.add_argument("--label-mode", type=str, choices=["point", "soft-box"], default="soft-box")
    parser.add_argument("--bbox-radius-scale", type=float, default=0.25)
    parser.add_argument("--min-target-radius", type=int, default=0)
    parser.add_argument("--max-target-radius", type=int, default=3)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage-width-scale", type=float, default=1.0)
    parser.add_argument("--head-filters", type=int, default=32)
    parser.add_argument("--refine-blocks", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--dice-weight", type=float, default=0.15)
    parser.add_argument("--auto-class-weight", action="store_true")
    parser.add_argument("--min-class-weight", type=float, default=0.75)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--focus-classes", type=str, default="")
    parser.add_argument("--focus-multiplier", type=int, default=1)
    parser.add_argument("--save-h5", action="store_true")
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--out-dir", type=str, default="fomoPy/outputs/fomo_local")
    args = parser.parse_args()

    workspace = Path.cwd()
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg_path = cfg.get("path")
    dataset_root = Path(cfg_path) if Path(cfg_path).is_absolute() else (workspace / cfg_path).resolve()

    train_img_dir = dataset_root / cfg["train"]
    val_img_dir = dataset_root / cfg["val"]
    train_label_dir = (labels_root / "training").resolve()
    val_label_dir = (labels_root / "testing").resolve()

    class_names = cfg.get("names", [])
    num_classes = int(cfg["nc"])
    downsample_steps = validate_grid_layout(args.image_size, args.grid_size)

    x_train, y_train = load_dataset(
        train_img_dir,
        train_label_dir,
        args.image_size,
        args.grid_size,
        num_classes,
        label_mode=args.label_mode,
        bbox_radius_scale=args.bbox_radius_scale,
        min_target_radius=args.min_target_radius,
        max_target_radius=args.max_target_radius,
    )
    x_val, y_val = load_dataset(
        val_img_dir,
        val_label_dir,
        args.image_size,
        args.grid_size,
        num_classes,
        label_mode=args.label_mode,
        bbox_radius_scale=args.bbox_radius_scale,
        min_target_radius=args.min_target_radius,
        max_target_radius=args.max_target_radius,
    )
    x_train, y_train = augment_photometric(
        x_train,
        y_train,
        aug_multiplier=args.aug_multiplier,
        seed=args.seed,
        hflip_prob=args.hflip_prob,
    )
    x_train, y_train = oversample_focus_classes(
        x_train,
        y_train,
        class_names=class_names,
        focus_classes_text=args.focus_classes,
        focus_multiplier=args.focus_multiplier,
        seed=args.seed,
    )

    print(f"Train samples after augmentation: {x_train.shape[0]}")
    print(f"Grid layout: image={args.image_size}, grid={args.grid_size}, downsample_steps={downsample_steps}")
    print(f"Label mode: {args.label_mode}, bbox_radius_scale={args.bbox_radius_scale}")

    fg_channel_weights = None
    if args.auto_class_weight:
        fg_channel_weights = compute_auto_class_weights(
            y_train,
            min_weight=args.min_class_weight,
            max_weight=args.max_class_weight,
        )
        print("Auto class weights:")
        for cls_name, weight in zip(class_names, fg_channel_weights):
            print(f"  {cls_name}: {weight:.4f}")

    # Compact layout that scales with the requested grid while staying OpenMV-friendly.
    stage_filters = derive_stage_filters(downsample_steps, width_scale=args.stage_width_scale)
    head_filters = int(args.head_filters)

    if args.optimizer == "adamw":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model = build_fomo_like_model(
        args.image_size,
        num_classes,
        stage_filters=stage_filters,
        head_filters=head_filters,
        refine_blocks=args.refine_blocks,
        dropout_rate=args.dropout_rate,
    )
    model.compile(
        optimizer=optimizer,
        loss=build_fomo_focal_loss(
            bg_weight=args.bg_weight,
            fg_weight=args.fg_weight,
            gamma=args.focal_gamma,
            fg_channel_weights=fg_channel_weights,
            dice_weight=args.dice_weight,
        ),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            make_fg_precision_metric(args.metric_threshold),
            make_fg_recall_metric(args.metric_threshold),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    out_dir = (workspace / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_training_artifacts(
        history=history,
        out_dir=out_dir,
        figure_path=(workspace / "img" / "train_loss_curve.png").resolve(),
    )

    keras_path = out_dir / "fomo_like.keras"
    model.save(keras_path)

    h5_path = None
    if args.save_h5:
        h5_path = out_dir / "fomo_like.h5"
        model.save(h5_path, include_optimizer=False)

    float32_path = None
    int8_path = None
    if not args.skip_tflite:
        float32_path, int8_path = export_tflite_models(model, x_train, out_dir)

    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(["background"] + class_names), encoding="utf-8")

    config_payload = vars(args).copy()
    config_payload["stage_filters"] = [int(v) for v in stage_filters]
    config_payload["classes"] = list(class_names)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {keras_path}")
    if h5_path is not None:
        print(f"  {h5_path} ({h5_path.stat().st_size} bytes)")
    if float32_path is not None and int8_path is not None:
        print(f"  {float32_path} ({float32_path.stat().st_size} bytes)")
        print(f"  {int8_path} ({int8_path.stat().st_size} bytes)")
    else:
        print("  TFLite export skipped")
    print(f"  {labels_path}")
    print(f"  {out_dir / 'training_history.json'}")
    print(f"  {config_path}")
    print(f"  {(workspace / 'img' / 'train_loss_curve.png').resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
