import argparse
import os
from pathlib import Path

import tensorflow as tf

from train import (
    augment_photometric,
    build_fomo_like_model,
    fg_precision_metric,
    fg_recall_metric,
    load_dataset,
)
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke-train a local FOMO-like model")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--aug-multiplier", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--es-patience", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="fomoPy/outputs/fomo_quick")
    args = parser.parse_args()

    workspace = Path.cwd()
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() else (workspace / cfg["path"]).resolve()
    train_img_dir = dataset_root / cfg["train"]
    val_img_dir = dataset_root / cfg["val"]
    train_label_dir = (labels_root / "training").resolve()
    val_label_dir = (labels_root / "testing").resolve()

    class_names = cfg.get("names", [])
    num_classes = int(cfg["nc"])

    x_train, y_train = load_dataset(train_img_dir, train_label_dir, args.image_size, args.grid_size, num_classes)
    x_val, y_val = load_dataset(val_img_dir, val_label_dir, args.image_size, args.grid_size, num_classes)
    x_train, y_train = augment_photometric(x_train, y_train, aug_multiplier=args.aug_multiplier, seed=args.seed)

    model = build_fomo_like_model(args.image_size, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc"), fg_precision_metric, fg_recall_metric],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
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
    (out_dir / "labels.txt").write_text("\n".join(["background"] + class_names), encoding="utf-8")
    model.save(out_dir / "fomo.keras")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
