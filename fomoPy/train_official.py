import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import yaml

from official_fomo import (
    augment_images_and_targets,
    export_tflite_models,
    load_training_arrays,
    oversample_focus_samples,
    save_json,
    synthesize_negative_crops,
)


def build_official_fomo_model(
    image_size: int,
    num_classes: int,
    alpha: float,
    head_filters: int,
    weights: str,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_rescale")(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights=weights,
        alpha=alpha,
    )
    features = backbone.get_layer("block_6_expand_relu").output

    x = tf.keras.layers.Conv2D(head_filters, 1, padding="same", use_bias=False, name="head_conv")(features)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.ReLU(name="head_relu")(x)
    logits = tf.keras.layers.Conv2D(num_classes + 1, 1, padding="same", name="head_logits")(x)
    outputs = tf.keras.layers.Softmax(axis=-1, name="fomo_probs")(logits)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fomo_official")
    return model, backbone


def make_weighted_sparse_ce(object_weight: float, background_weight: float):
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weights = tf.where(
            y_true > 0,
            tf.cast(object_weight, tf.float32),
            tf.cast(background_weight, tf.float32),
        )
        return tf.reduce_mean(loss * weights)

    return _loss


def non_background_precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true > 0, y_pred == y_true), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(y_pred > 0, y_pred != y_true), tf.float32))
    return tp / (tp + fp + 1e-6)


def non_background_recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(y_true > 0, y_pred == y_true), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(y_true > 0, y_pred != y_true), tf.float32))
    return tp / (tp + fn + 1e-6)


def merge_histories(*histories: tf.keras.callbacks.History) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, [])
            merged[key].extend(float(v) for v in values)
    return merged


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    object_weight: float,
    background_weight: float,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_weighted_sparse_ce(object_weight=object_weight, background_weight=background_weight),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_acc"),
            non_background_precision,
            non_background_recall,
        ],
    )


def set_backbone_trainable(backbone: tf.keras.Model, trainable: bool) -> None:
    backbone.trainable = trainable
    if trainable:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Edge Impulse official-style FOMO model locally")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--aug-multiplier", type=int, default=1)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--mobilenet-alpha", type=float, default=0.35)
    parser.add_argument("--head-filters", type=int, default=32)
    parser.add_argument("--object-weight", type=float, default=100.0)
    parser.add_argument("--background-weight", type=float, default=2.0)
    parser.add_argument("--focus-classes", type=str, default="Tennis player,Tennis racket")
    parser.add_argument("--focus-multiplier", type=int, default=2)
    parser.add_argument("--negative-crops-per-image", type=int, default=1)
    parser.add_argument("--negative-crop-min-scale", type=float, default=0.35)
    parser.add_argument("--negative-crop-max-scale", type=float, default=0.85)
    parser.add_argument("--negative-avoid-expand-ratio", type=float, default=0.20)
    parser.add_argument("--negative-hard-ratio", type=float, default=0.70)
    parser.add_argument("--negative-hard-gap-ratio", type=float, default=0.08)
    parser.add_argument("--pretrained", type=str, choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--allow-random-init-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-h5", action="store_true")
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--out-dir", type=str, default="fomoPy/outputs/fomo_official")
    args = parser.parse_args()

    if args.image_size // args.grid_size != 8:
        raise ValueError(
            "official FOMO uses the 1/8 MobileNetV2 cut point, so image_size / grid_size must equal 8 "
            f"(got {args.image_size} / {args.grid_size} = {args.image_size / args.grid_size:.3f})"
        )

    tf.keras.utils.set_random_seed(args.seed)

    workspace = Path.cwd()
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    dataset_root = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() else (workspace / cfg["path"]).resolve()
    train_img_dir = dataset_root / cfg["train"]
    val_img_dir = dataset_root / cfg["val"]
    train_label_dir = labels_root / "training"
    val_label_dir = labels_root / "testing"
    class_names = list(cfg["names"])
    num_classes = int(cfg["nc"])

    x_train, y_train = load_training_arrays(
        train_img_dir,
        train_label_dir,
        args.image_size,
        args.grid_size,
        num_classes,
    )
    x_val, y_val = load_training_arrays(
        val_img_dir,
        val_label_dir,
        args.image_size,
        args.grid_size,
        num_classes,
    )

    x_neg, y_neg = synthesize_negative_crops(
        train_img_dir,
        train_label_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        negative_crops_per_image=args.negative_crops_per_image,
        seed=args.seed,
        min_crop_scale=args.negative_crop_min_scale,
        max_crop_scale=args.negative_crop_max_scale,
        avoid_box_expand_ratio=args.negative_avoid_expand_ratio,
        hard_negative_ratio=args.negative_hard_ratio,
        hard_negative_gap_ratio=args.negative_hard_gap_ratio,
    )
    if x_neg.shape[0] > 0:
        x_train = np.concatenate([x_train, x_neg], axis=0)
        y_train = np.concatenate([y_train, y_neg], axis=0)

    x_train, y_train = augment_images_and_targets(
        x_train,
        y_train,
        aug_multiplier=args.aug_multiplier,
        seed=args.seed,
        hflip_prob=args.hflip_prob,
    )
    x_train, y_train, added_focus = oversample_focus_samples(
        x_train,
        y_train,
        class_names=class_names,
        focus_classes_text=args.focus_classes,
        focus_multiplier=args.focus_multiplier,
        seed=args.seed,
    )

    print(f"Train samples after augmentation: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Classes: {class_names}")
    print(f"Synthetic negative crops added: {x_neg.shape[0]}")
    print(f"Focus oversampling added: {added_focus}")
    print(
        f"Official-style config: alpha={args.mobilenet_alpha}, "
        f"object_weight={args.object_weight}, background_weight={args.background_weight}, "
        f"pretrained={args.pretrained}"
    )

    weights_arg = "imagenet" if args.pretrained == "imagenet" else None
    try:
        model, backbone = build_official_fomo_model(
            image_size=args.image_size,
            num_classes=num_classes,
            alpha=args.mobilenet_alpha,
            head_filters=args.head_filters,
            weights=weights_arg,
        )
    except Exception:
        if not args.allow_random_init_fallback or args.pretrained != "imagenet":
            raise
        print("Warning: failed to load ImageNet weights, falling back to random init.")
        model, backbone = build_official_fomo_model(
            image_size=args.image_size,
            num_classes=num_classes,
            alpha=args.mobilenet_alpha,
            head_filters=args.head_filters,
            weights=None,
        )
        args.pretrained = "none"

    output_shape = model.output_shape
    if output_shape[1] != args.grid_size or output_shape[2] != args.grid_size:
        raise ValueError(
            f"grid mismatch: model outputs {output_shape[1]}x{output_shape[2]}, expected {args.grid_size}x{args.grid_size}"
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    warmup_history = None
    finetune_history = None

    if args.pretrained == "imagenet" and args.warmup_epochs > 0:
        warmup_epochs = min(args.epochs, args.warmup_epochs)
        set_backbone_trainable(backbone, trainable=False)
        compile_model(
            model,
            learning_rate=args.head_lr,
            object_weight=args.object_weight,
            background_weight=args.background_weight,
        )
        warmup_history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=warmup_epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        remaining_epochs = max(0, args.epochs - warmup_epochs)
        if remaining_epochs > 0:
            set_backbone_trainable(backbone, trainable=True)
            compile_model(
                model,
                learning_rate=args.finetune_lr,
                object_weight=args.object_weight,
                background_weight=args.background_weight,
            )
            finetune_history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                initial_epoch=warmup_epochs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
                verbose=2,
            )
    else:
        set_backbone_trainable(backbone, trainable=True)
        compile_model(
            model,
            learning_rate=args.finetune_lr,
            object_weight=args.object_weight,
            background_weight=args.background_weight,
        )
        finetune_history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=2,
        )

    history_payload = merge_histories(warmup_history, finetune_history)
    out_dir = (workspace / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "training_history.json", history_payload)

    keras_path = out_dir / "fomo_official.keras"
    model.save(keras_path)

    h5_path = None
    if args.save_h5:
        h5_path = out_dir / "fomo_official.h5"
        model.save(h5_path, include_optimizer=False)

    float32_path = None
    int8_path = None
    if not args.skip_tflite:
        float32_path, int8_path = export_tflite_models(model, x_train, out_dir)

    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(["background"] + class_names), encoding="utf-8")

    config_payload = {
        "image_size": args.image_size,
        "grid_size": args.grid_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "mobilenet_alpha": args.mobilenet_alpha,
        "head_filters": args.head_filters,
        "object_weight": args.object_weight,
        "background_weight": args.background_weight,
        "focus_classes": args.focus_classes,
        "focus_multiplier": args.focus_multiplier,
        "negative_crops_per_image": args.negative_crops_per_image,
        "negative_crop_min_scale": args.negative_crop_min_scale,
        "negative_crop_max_scale": args.negative_crop_max_scale,
        "negative_avoid_expand_ratio": args.negative_avoid_expand_ratio,
        "negative_hard_ratio": args.negative_hard_ratio,
        "negative_hard_gap_ratio": args.negative_hard_gap_ratio,
        "pretrained": args.pretrained,
        "classes": class_names,
    }
    save_json(out_dir / "config.json", config_payload)

    print("Saved:")
    print(f"  {keras_path}")
    if h5_path is not None:
        print(f"  {h5_path}")
    if float32_path is not None and int8_path is not None:
        print(f"  {float32_path}")
        print(f"  {int8_path}")
    print(f"  {labels_path}")
    print(f"  {out_dir / 'training_history.json'}")
    print(f"  {out_dir / 'config.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
