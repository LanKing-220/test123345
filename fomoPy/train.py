import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm


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


def load_dataset(
    image_dir: Path,
    label_dir: Path,
    image_size: int,
    grid_size: int,
    num_classes: int,
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

        for cid, cx, cy, _, _ in labels:
            cx = float(np.clip(cx, 0.0, 0.9999))
            cy = float(np.clip(cy, 0.0, 0.9999))
            gx = min(int(cx * grid_size), grid_size - 1)
            gy = min(int(cy * grid_size), grid_size - 1)
            if 0 <= cid < num_classes:
                target[gy, gx, cid + 1] = 1.0

        fg = np.max(target[:, :, 1:], axis=-1)
        target[:, :, 0] = 1.0 - fg

        xs.append(img)
        ys.append(target)

    return np.stack(xs), np.stack(ys)


def build_fomo_like_model(
    image_size: int,
    num_classes: int,
    stage_filters: Iterable[int],
    head_filters: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs

    stage_filters = list(stage_filters)
    if not stage_filters:
        raise ValueError("stage_filters must not be empty")

    first_filters = int(stage_filters[0])
    x = tf.keras.layers.Conv2D(first_filters, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for filters in stage_filters[1:]:
        x = tf.keras.layers.SeparableConv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(head_filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

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


def build_fomo_focal_loss(bg_weight: float = 0.25, fg_weight: float = 2.0, gamma: float = 2.0):
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal = tf.pow(1.0 - pt, gamma)

        ch = tf.shape(y_true)[-1]
        fg = tf.fill([ch - 1], tf.cast(fg_weight, tf.float32))
        weights = tf.concat([[tf.cast(bg_weight, tf.float32)], fg], axis=0)
        weights = tf.reshape(weights, [1, 1, 1, -1])
        return tf.reduce_mean(bce * focal * weights)

    return _loss


def fg_precision_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_fg = y_true[..., 1:]
    y_pred_fg = tf.cast(y_pred[..., 1:] >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_pred_fg * y_true_fg)
    fp = tf.reduce_sum(y_pred_fg * (1.0 - y_true_fg))
    return tp / (tp + fp + 1e-6)


def fg_recall_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_fg = y_true[..., 1:]
    y_pred_fg = tf.cast(y_pred[..., 1:] >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_pred_fg * y_true_fg)
    fn = tf.reduce_sum((1.0 - y_pred_fg) * y_true_fg)
    return tp / (tp + fn + 1e-6)


def augment_photometric(
    x: np.ndarray,
    y: np.ndarray,
    aug_multiplier: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if aug_multiplier <= 0:
        return x, y

    rng = np.random.default_rng(seed)
    x_aug = [x]
    y_aug = [y]

    for _ in range(aug_multiplier):
        out = x.copy()
        for i in range(out.shape[0]):
            img = out[i]
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

            out[i] = img.astype(np.float32)

        x_aug.append(out)
        y_aug.append(y)

    x_all = np.concatenate(x_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)
    idx = rng.permutation(x_all.shape[0])
    return x_all[idx], y_all[idx]


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bg-weight", type=float, default=0.25)
    parser.add_argument("--fg-weight", type=float, default=2.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="Compactness preset for OpenMV-style deployment",
    )
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

    x_train, y_train = load_dataset(train_img_dir, train_label_dir, args.image_size, args.grid_size, num_classes)
    x_val, y_val = load_dataset(val_img_dir, val_label_dir, args.image_size, args.grid_size, num_classes)
    x_train, y_train = augment_photometric(x_train, y_train, aug_multiplier=args.aug_multiplier, seed=args.seed)

    print(f"Train samples after augmentation: {x_train.shape[0]}")

    size_presets = {
        "tiny": ([8, 16, 24], 32),
        "small": ([12, 24, 32], 48),
        "medium": ([16, 32, 48], 64),
    }
    stage_filters, head_filters = size_presets[args.model_size]

    model = build_fomo_like_model(
        args.image_size,
        num_classes,
        stage_filters=stage_filters,
        head_filters=head_filters,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=build_fomo_focal_loss(bg_weight=args.bg_weight, fg_weight=args.fg_weight, gamma=args.focal_gamma),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc"), fg_precision_metric, fg_recall_metric],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    out_dir = (workspace / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    keras_path = out_dir / "fomo_like.keras"
    model.save(keras_path)

    float32_path, int8_path = export_tflite_models(model, x_train, out_dir)

    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(["background"] + class_names), encoding="utf-8")

    print("Saved:")
    print(f"  {keras_path}")
    print(f"  {float32_path} ({float32_path.stat().st_size} bytes)")
    print(f"  {int8_path} ({int8_path.stat().st_size} bytes)")
    print(f"  {labels_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
