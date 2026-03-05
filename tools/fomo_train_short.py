import argparse
import os
from pathlib import Path

import tensorflow as tf
import yaml

from train_fomo_local import (
    augment_photometric,
    build_fomo_like_model,
    fg_precision_metric,
    fg_recall_metric,
    load_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Short training entry for fomo")
    parser.add_argument("--data-yaml", type=str, default="data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=98)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--aug-multiplier", type=int, default=1, help="How many extra augmented copies per training sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--es-patience", type=int, default=4, help="Early stopping patience on val_loss")
    parser.add_argument("--out-dir", type=str, default="outputs/fomo_short_run")
    args = parser.parse_args()

    workspace = Path.cwd()
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg_path = cfg.get("path")
    if cfg_path:
        cfg_path_obj = Path(cfg_path)
        dataset_root = cfg_path_obj if cfg_path_obj.is_absolute() else (workspace / cfg_path_obj).resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    train_img_dir = dataset_root / cfg["train"]
    val_img_dir = dataset_root / cfg["val"]
    train_label_dir = (labels_root / "training").resolve()
    val_label_dir = (labels_root / "testing").resolve()

    class_names = cfg.get("names", [])
    num_classes = int(cfg["nc"])

    x_train, y_train = load_dataset(train_img_dir, train_label_dir, args.image_size, args.grid_size, num_classes)
    x_val, y_val = load_dataset(val_img_dir, val_label_dir, args.image_size, args.grid_size, num_classes)

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
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            fg_precision_metric,
            fg_recall_metric,
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
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

    keras_path = out_dir / "fomo.keras"
    model.save(keras_path)

    float_tflite_path = out_dir / "fomo_float32.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_tflite_path.write_bytes(converter.convert())

    labels_path = out_dir / "fomo_labels.txt"
    labels_path.write_text("\n".join(["background"] + class_names), encoding="utf-8")

    print("Saved:")
    print(keras_path)
    print(float_tflite_path)
    print(labels_path)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
