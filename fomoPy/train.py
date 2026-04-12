import argparse
import json
import os
from pathlib import Path

import numpy as np
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
        make_fg_f1_metric,
        make_fg_precision_metric,
        make_fg_recall_metric,
        oversample_focus_classes,
        recommend_fg_thresholds,
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
        make_fg_f1_metric,
        make_fg_precision_metric,
        make_fg_recall_metric,
        oversample_focus_classes,
        recommend_fg_thresholds,
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
    parser.add_argument("--metric-threshold", type=float, default=0.50)
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
    parser.add_argument("--metric-gt-threshold", type=float, default=None)
    parser.add_argument(
        "--monitor",
        type=str,
        choices=["val_loss", "val_fg_precision", "val_fg_recall", "val_fg_f1"],
        default="val_fg_f1",
    )
    parser.add_argument("--early-stop-patience", type=int, default=16)
    parser.add_argument("--early-stop-start-epoch", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--reduce-lr-patience", type=int, default=5)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--threshold-search-min", type=float, default=0.20)
    parser.add_argument("--threshold-search-max", type=float, default=0.80)
    parser.add_argument("--threshold-search-step", type=float, default=0.05)
    parser.add_argument("--focus-classes", type=str, default="")
    parser.add_argument("--focus-multiplier", type=int, default=1)
    parser.add_argument("--save-h5", action="store_true")
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--out-dir", type=str, default="fomoPy/outputs/fomo_local")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable EarlyStopping callback")
    return parser


def infer_metric_gt_threshold(args: argparse.Namespace) -> float:
    if args.metric_gt_threshold is not None:
        return float(args.metric_gt_threshold)
    return 0.99 if args.label_mode == "soft-box" else 0.5


def build_threshold_search_values(args: argparse.Namespace) -> np.ndarray:
    if args.threshold_search_step <= 0:
        raise ValueError("threshold_search_step must be > 0")
    if args.threshold_search_max < args.threshold_search_min:
        raise ValueError("threshold_search_max must be >= threshold_search_min")

    values = np.arange(
        args.threshold_search_min,
        args.threshold_search_max + args.threshold_search_step * 0.5,
        args.threshold_search_step,
        dtype=np.float32,
    )
    values = np.unique(np.clip(values, 1e-3, 0.999))
    if values.size == 0:
        raise ValueError("threshold search range produced no values")
    return values


def summarize_empty_images(y: np.ndarray, gt_threshold: float) -> tuple[int, int]:
    has_fg = np.any(y[..., 1:] >= float(gt_threshold), axis=(1, 2, 3))
    empty = int(np.sum(~has_fg))
    return empty, int(y.shape[0])


def resolve_output_dir(workspace: Path, out_dir_text: str) -> Path:
    out_dir = Path(out_dir_text)
    if out_dir.is_absolute():
        return out_dir
    if workspace.name == "fomoPy" and out_dir.parts and out_dir.parts[0] == workspace.name:
        return (workspace.parent / out_dir).resolve()
    return (workspace / out_dir).resolve()


def load_training_data(args: argparse.Namespace, workspace: Path):
    # Resolve paths for data_yaml and labels_dir robustly to avoid duplicating
    # the `fomoPy` segment when running from inside the `fomoPy` folder.
    data_yaml_path = Path(args.data_yaml)
    labels_dir_path = Path(args.labels_dir)

    if not data_yaml_path.is_absolute():
        candidates = [
            workspace / data_yaml_path,
            workspace.parent / data_yaml_path,
            Path.cwd() / data_yaml_path,
            Path(__file__).resolve().parent / data_yaml_path,
        ]
        for c in candidates:
            if c.exists():
                data_yaml_path = c
                break
        else:
            data_yaml_path = (workspace / data_yaml_path)

    if not labels_dir_path.is_absolute():
        candidates = [
            workspace / labels_dir_path,
            workspace.parent / labels_dir_path,
            Path.cwd() / labels_dir_path,
            Path(__file__).resolve().parent / labels_dir_path,
        ]
        for c in candidates:
            if c.exists():
                labels_dir_path = c
                break
        else:
            labels_dir_path = (workspace / labels_dir_path)

    dataset = resolve_dataset_layout(workspace, data_yaml_path, labels_dir_path)

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
    metric_gt_threshold: float,
    threshold_report,
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
    config_payload["metric_gt_threshold"] = float(metric_gt_threshold)
    config_payload["recommended_threshold"] = float(threshold_report["global"]["threshold"])
    config_payload["recommended_class_thresholds"] = [
        float(v) for v in threshold_report["per_class"]["thresholds"]
    ]
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    threshold_path = out_dir / "threshold_recommendations.json"
    threshold_path.write_text(json.dumps(threshold_report, ensure_ascii=False, indent=2), encoding="utf-8")

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
    print(f"  {threshold_path}")
    print(f"  {(workspace / 'img' / 'train_loss_curve.png').resolve()}")


def main() -> None:
    args = build_arg_parser().parse_args()

    workspace = Path.cwd()
    dataset, x_train, y_train, x_val, y_val = load_training_data(args, workspace)
    downsample_steps = validate_grid_layout(args.image_size, args.grid_size)

    print(f"Train samples after augmentation: {x_train.shape[0]}")
    print(f"Grid layout: image={args.image_size}, grid={args.grid_size}, downsample_steps={downsample_steps}")
    print(f"Label mode: {args.label_mode}, bbox_radius_scale={args.bbox_radius_scale}")

    metric_gt_threshold = infer_metric_gt_threshold(args)
    train_empty, train_total = summarize_empty_images(y_train, gt_threshold=metric_gt_threshold)
    val_empty, val_total = summarize_empty_images(y_val, gt_threshold=metric_gt_threshold)
    print(f"Metric ground-truth threshold: {metric_gt_threshold:.2f}")
    print(f"Empty images under metric view: train={train_empty}/{train_total}, val={val_empty}/{val_total}")
    if train_empty == 0 and val_empty > 0:
        print(
            "Warning: validation/test contains empty frames but training does not. "
            "Precision will stay hard to improve until negative-only images are added to training."
        )

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
    precision_metric = make_fg_precision_metric(args.metric_threshold, gt_threshold=metric_gt_threshold)
    recall_metric = make_fg_recall_metric(args.metric_threshold, gt_threshold=metric_gt_threshold)
    f1_metric = make_fg_f1_metric(args.metric_threshold, gt_threshold=metric_gt_threshold)
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
            precision_metric,
            recall_metric,
            f1_metric,
        ],
    )

    monitor_lookup = {
        "val_loss": "val_loss",
        "val_fg_precision": f"val_{precision_metric.name}",
        "val_fg_recall": f"val_{recall_metric.name}",
        "val_fg_f1": f"val_{f1_metric.name}",
    }
    monitor_name = monitor_lookup[args.monitor]
    monitor_mode = "min" if args.monitor == "val_loss" else "max"
    print(f"Callbacks monitor: {monitor_name} ({monitor_mode})")

    callbacks = []
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_name,
            mode=monitor_mode,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_delta=args.early_stop_min_delta,
            min_lr=args.min_lr,
            verbose=1,
        )
    )
    if not args.no_early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_name,
                mode=monitor_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                start_from_epoch=args.early_stop_start_epoch,
                restore_best_weights=True,
                verbose=1,
            )
        )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    threshold_report = recommend_fg_thresholds(
        y_true=y_val,
        y_pred=model.predict(x_val, batch_size=args.batch_size, verbose=0),
        thresholds=build_threshold_search_values(args),
        gt_threshold=metric_gt_threshold,
        class_names=dataset.class_names,
    )

    print("Validation threshold calibration:")
    print(
        "  Global threshold="
        f"{threshold_report['global']['threshold']:.2f} "
        f"prec={threshold_report['global']['precision']:.4f} "
        f"rec={threshold_report['global']['recall']:.4f} "
        f"f1={threshold_report['global']['f1']:.4f}"
    )
    print("  Per-class thresholds:")
    for row in threshold_report["per_class"]["rows"]:
        print(
            f"    {row['class_name']}: thr={row['threshold']:.2f} "
            f"prec={row['precision']:.4f} rec={row['recall']:.4f} f1={row['f1']:.4f}"
        )
    print(
        "  Combined per-class:"
        f" prec={threshold_report['per_class']['overall']['precision']:.4f}"
        f" rec={threshold_report['per_class']['overall']['recall']:.4f}"
        f" f1={threshold_report['per_class']['overall']['f1']:.4f}"
    )

    save_outputs(
        args=args,
        workspace=workspace,
        out_dir=resolve_output_dir(workspace, args.out_dir),
        model=model,
        x_train=x_train,
        history=history,
        class_names=dataset.class_names,
        stage_filters=stage_filters,
        metric_gt_threshold=metric_gt_threshold,
        threshold_report=threshold_report,
    )


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
