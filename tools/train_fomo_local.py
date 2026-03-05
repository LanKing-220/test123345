import argparse
import os
from pathlib import Path
from typing import List, Tuple

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


def resolve_label_path(image_path: Path, label_dir: Path) -> Path:
    """Resolve YOLO label path for images with possible ingestion-style suffixes."""
    stem = image_path.stem

    # 1) Exact stem match.
    p = label_dir / f"{stem}.txt"
    if p.exists():
        return p

    # 2) Common case: "name.jpg.<extra>.ingestion-..." -> "name"
    lower_name = image_path.name.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        marker = ext + "."
        idx = lower_name.find(marker)
        if idx != -1:
            base = image_path.name[:idx]
            p = label_dir / f"{base}.txt"
            if p.exists():
                return p

    # 3) Fallback: progressively trim dotted suffixes from stem.
    parts = stem.split(".")
    while len(parts) > 1:
        parts = parts[:-1]
        p = label_dir / f"{'.'.join(parts)}.txt"
        if p.exists():
            return p

    return label_dir / f"{stem}.txt"


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
        label_path = resolve_label_path(image_path, label_dir)
        labels = parse_yolo_file(label_path)

        img = imread_unicode(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        # Target tensor shape: [grid_h, grid_w, num_classes + 1(background)]
        target = np.zeros((grid_size, grid_size, num_classes + 1), dtype=np.float32)

        for cid, cx, cy, _, _ in labels:
            # Clamp noisy labels into [0, 1].
            cx = float(np.clip(cx, 0.0, 0.9999))
            cy = float(np.clip(cy, 0.0, 0.9999))
            gx = min(int(cx * grid_size), grid_size - 1)
            gy = min(int(cy * grid_size), grid_size - 1)
            if 0 <= cid < num_classes:
                target[gy, gx, cid + 1] = 1.0

        # Background channel: 1 for empty cells, 0 for cells with foreground.
        fg = np.max(target[:, :, 1:], axis=-1)
        target[:, :, 0] = 1.0 - fg

        xs.append(img)
        ys.append(target)

    return np.stack(xs), np.stack(ys)


def build_fomo_like_model(image_size: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs

    # Three stride-2 stages -> output stride 8, e.g. 96x96 -> 12x12 heatmap.
    for filters in [16, 32, 64]:
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2D(num_classes + 1, 1, activation="sigmoid", name="fomo_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="fomo_like")


def build_fomo_focal_loss(bg_weight: float = 0.25, fg_weight: float = 2.0, gamma: float = 2.0):
    """Foreground-aware focal BCE loss for FOMO grids.

    Channel 0 is background; channels 1..N are foreground classes.
    """

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
    """Create extra training samples with photometric augmentation only.

    This keeps object locations unchanged so FOMO grid targets remain valid.
    """
    if aug_multiplier <= 0:
        return x, y

    rng = np.random.default_rng(seed)
    x_aug = [x]
    y_aug = [y]

    for _ in range(aug_multiplier):
        out = x.copy()
        for i in range(out.shape[0]):
            img = out[i]

            # Brightness + contrast jitter
            alpha = float(rng.uniform(0.85, 1.20))
            beta = float(rng.uniform(-0.10, 0.10))
            img = np.clip((img * alpha) + beta, 0.0, 1.0)

            # Mild gamma jitter
            gamma = float(rng.uniform(0.85, 1.20))
            img = np.power(img, gamma)

            # Small gaussian noise
            noise_std = float(rng.uniform(0.0, 0.03))
            if noise_std > 0.0:
                noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

            # Occasional blur to simulate motion/focus changes.
            if rng.random() < 0.30:
                img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)
                img = img_u8.astype(np.float32) / 255.0

            out[i] = img.astype(np.float32)

        x_aug.append(out)
        y_aug.append(y)

    x_all = np.concatenate(x_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)

    # Shuffle after augmentation.
    idx = rng.permutation(x_all.shape[0])
    return x_all[idx], y_all[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local FOMO-like model from YOLO labels")
    parser.add_argument("--data-yaml", type=str, default="data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=98)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug-multiplier", type=int, default=1, help="How many extra augmented copies per training sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bg-weight", type=float, default=0.25, help="Loss weight for background channel")
    parser.add_argument("--fg-weight", type=float, default=2.0, help="Loss weight for foreground channels")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--out-dir", type=str, default="outputs/fomo_local")
    args = parser.parse_args()

    workspace = Path.cwd()
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg_path = cfg.get("path")
    if cfg_path:
        cfg_path_obj = Path(cfg_path)
        if cfg_path_obj.is_absolute():
            dataset_root = cfg_path_obj
        else:
            dataset_root = (workspace / cfg_path_obj).resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    train_img_dir = dataset_root / cfg["train"]
    val_img_dir = dataset_root / cfg["val"]

    train_label_dir = (labels_root / "training").resolve()
    val_label_dir = (labels_root / "testing").resolve()

    class_names = cfg.get("names", [])
    num_classes = int(cfg["nc"])

    x_train, y_train = load_dataset(
        train_img_dir,
        train_label_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        num_classes=num_classes,
    )
    x_val, y_val = load_dataset(
        val_img_dir,
        val_label_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        num_classes=num_classes,
    )

    x_train, y_train = augment_photometric(
        x_train,
        y_train,
        aug_multiplier=args.aug_multiplier,
        seed=args.seed,
    )

    print(f"Train samples after augmentation: {x_train.shape[0]}")

    model = build_fomo_like_model(args.image_size, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=build_fomo_focal_loss(bg_weight=args.bg_weight, fg_weight=args.fg_weight, gamma=args.focal_gamma),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            fg_precision_metric,
            fg_recall_metric,
        ],
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

    tflite_path = out_dir / "fomo_like_float32.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    labels_path = out_dir / "labels.txt"
    labels = ["background"] + class_names
    labels_path.write_text("\n".join(labels), encoding="utf-8")

    print("Saved:")
    print(f"  {keras_path}")
    print(f"  {tflite_path}")
    print(f"  {labels_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
