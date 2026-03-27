import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml

from official_fomo import (
    collect_gt_objects,
    collect_image_paths,
    extract_fomo_detections,
    load_model_runner,
    parse_class_thresholds,
    print_confusion_summary,
    resize_rgb_image,
    confusion_matrix_from_objects,
    save_json,
    summarize_confusion_matrix,
)


def evaluate_split(
    predict_fn,
    image_dir: Path,
    label_dir: Path,
    class_names,
    image_size: int,
    grid_size: int,
    threshold: float,
    class_thresholds: Optional[object],
) -> Dict[str, object]:
    import numpy as np

    total_matrix = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=np.int32)
    image_rows = []

    for image_path in collect_image_paths(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        image = resize_rgb_image(image_path, image_size)
        output = predict_fn(image)
        gt_objects = collect_gt_objects(label_path, class_names, grid_size=grid_size, image_size=image_size)
        pred_objects = extract_fomo_detections(
            output,
            class_names=class_names,
            threshold=threshold,
            class_thresholds=class_thresholds,
            image_size=image_size,
        )
        matrix = confusion_matrix_from_objects(gt_objects, pred_objects, num_classes=len(class_names))
        total_matrix += matrix
        image_rows.append(
            {
                "image_name": image_path.name,
                "gt_count": len(gt_objects),
                "pred_count": len(pred_objects),
            }
        )

    summary = summarize_confusion_matrix(total_matrix, class_names)
    summary["num_images"] = len(image_rows)
    summary["images"] = image_rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an official-style FOMO model with object-level metrics")
    parser.add_argument("--model", type=str, default="fomoPy/outputs/fomo_official/fomo_official_int8.tflite")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/yolo_dataset/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/yolo_labels")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--class-thresholds", type=str, default="")
    parser.add_argument("--report", type=str, default="fomoPy/outputs/fomo_official/eval_official.json")
    args = parser.parse_args()

    workspace = Path.cwd()
    model_path = workspace / args.model
    data_yaml = workspace / args.data_yaml
    labels_root = workspace / args.labels_dir
    report_path = workspace / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() else (workspace / cfg["path"]).resolve()
    train_img_dir = dataset_root / cfg["train"]
    test_img_dir = dataset_root / cfg["val"]
    train_label_dir = labels_root / "training"
    test_label_dir = labels_root / "testing"
    class_names = list(cfg["names"])
    class_thresholds = parse_class_thresholds(args.class_thresholds, len(class_names))

    model_format, predict_fn = load_model_runner(model_path)

    train_summary = evaluate_split(
        predict_fn=predict_fn,
        image_dir=train_img_dir,
        label_dir=train_label_dir,
        class_names=class_names,
        image_size=args.image_size,
        grid_size=args.grid_size,
        threshold=args.threshold,
        class_thresholds=class_thresholds,
    )
    test_summary = evaluate_split(
        predict_fn=predict_fn,
        image_dir=test_img_dir,
        label_dir=test_label_dir,
        class_names=class_names,
        image_size=args.image_size,
        grid_size=args.grid_size,
        threshold=args.threshold,
        class_thresholds=class_thresholds,
    )

    payload = {
        "model": str(model_path),
        "model_format": model_format,
        "image_size": args.image_size,
        "grid_size": args.grid_size,
        "threshold": args.threshold,
        "class_thresholds": class_thresholds.tolist() if class_thresholds is not None else None,
        "metric_note": "Edge Impulse-style object-level confusion matrix built from FOMO heatmap blobs matched against GT coverage cells",
        "train": train_summary,
        "test": test_summary,
    }
    save_json(report_path, payload)

    print_confusion_summary("train", train_summary)
    print_confusion_summary("test", test_summary)
    print(f"report saved to: {report_path}")


if __name__ == "__main__":
    main()
