import argparse
import json
import os
from pathlib import Path

import tensorflow as tf

try:
    from fomoPy.local_fomo_common import resolve_dataset_layout
except ImportError:
    from local_fomo_common import resolve_dataset_layout
try:
    from fomoPy.local_train_common import (
        augment_photometric,
        build_fomo_focal_loss,
        build_fomo_like_model,
        compute_auto_class_weights,
        derive_stage_filters,
        export_tflite_models,
        load_dataset,
        make_fg_precision_metric,
        make_fg_recall_metric,
        oversample_focus_classes,
        save_training_artifacts,
        validate_grid_layout,
    )
except ImportError:
    from local_train_common import (
        augment_photometric,
        build_fomo_focal_loss,
        build_fomo_like_model,
        compute_auto_class_weights,
        derive_stage_filters,
        export_tflite_models,
        load_dataset,
        make_fg_precision_metric,
        make_fg_recall_metric,
        oversample_focus_classes,
        save_training_artifacts,
        validate_grid_layout,
    )


def build_arg_parser() -> argparse.ArgumentParser:
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
    return parser


def load_training_data(args: argparse.Namespace, workspace: Path):
    dataset = resolve_dataset_layout(workspace, workspace / args.data_yaml, workspace / args.labels_dir)

    x_train, y_train = load_dataset(
        dataset.train_img_dir,
        dataset.train_label_dir,
        args.image_size,
        args.grid_size,
        dataset.num_classes,
        label_mode=args.label_mode,
        bbox_radius_scale=args.bbox_radius_scale,
        min_target_radius=args.min_target_radius,
        max_target_radius=args.max_target_radius,
    )
    x_val, y_val = load_dataset(
        dataset.val_img_dir,
        dataset.val_label_dir,
        args.image_size,
        args.grid_size,
        dataset.num_classes,
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
        class_names=dataset.class_names,
        focus_classes_text=args.focus_classes,
        focus_multiplier=args.focus_multiplier,
        seed=args.seed,
    )
    return dataset, x_train, y_train, x_val, y_val


def build_optimizer(args: argparse.Namespace):
    if args.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=args.lr)


def save_outputs(
    args: argparse.Namespace,
    workspace: Path,
    out_dir: Path,
    model: tf.keras.Model,
    x_train,
    history,
    class_names,
    stage_filters,
) -> None:
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
    labels_path.write_text("\n".join(["background"] + list(class_names)), encoding="utf-8")

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


def main() -> None:
    args = build_arg_parser().parse_args()

    workspace = Path.cwd()
    dataset, x_train, y_train, x_val, y_val = load_training_data(args, workspace)
    downsample_steps = validate_grid_layout(args.image_size, args.grid_size)

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
        for cls_name, weight in zip(dataset.class_names, fg_channel_weights):
            print(f"  {cls_name}: {weight:.4f}")

    stage_filters = derive_stage_filters(downsample_steps, width_scale=args.stage_width_scale)
    model = build_fomo_like_model(
        args.image_size,
        dataset.num_classes,
        stage_filters=stage_filters,
        head_filters=int(args.head_filters),
        refine_blocks=args.refine_blocks,
        dropout_rate=args.dropout_rate,
    )
    model.compile(
        optimizer=build_optimizer(args),
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

    save_outputs(
        args=args,
        workspace=workspace,
        out_dir=(workspace / args.out_dir).resolve(),
        model=model,
        x_train=x_train,
        history=history,
        class_names=dataset.class_names,
        stage_filters=stage_filters,
    )


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
