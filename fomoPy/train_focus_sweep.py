import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_CLASSES = ["Tennis", "Tennis player", "Tennis racket"]


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def slugify(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def run_command(command: List[str], workdir: Path) -> None:
    print("")
    print(subprocess.list2cmdline(command))
    print("")
    subprocess.run(command, cwd=str(workdir), check=True)


def load_eval_summary(report_path: Path) -> Dict[str, object]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    test_summary = payload["test"]
    per_class = {
        row["class_name"]: {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
        }
        for row in test_summary["per_class"]
    }
    overall = test_summary["metrics_non_background"]
    return {
        "overall": {
            "precision": float(overall["precision"]),
            "recall": float(overall["recall"]),
            "f1": float(overall["f1"]),
            "tp": int(overall["tp"]),
            "fp": int(overall["fp"]),
            "fn": int(overall["fn"]),
        },
        "per_class": per_class,
    }


def save_summary_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_summary_markdown(path: Path, rows: List[Dict[str, object]], class_names: List[str]) -> None:
    header = [
        "Focus",
        "Overall P",
        "Overall R",
        "Overall F1",
    ]
    for class_name in class_names:
        header.append(f"{class_name} F1")

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [
            str(row["focus_class"]),
            f"{row['overall']['precision']:.3f}",
            f"{row['overall']['recall']:.3f}",
            f"{row['overall']['f1']:.3f}",
        ]
        for class_name in class_names:
            values.append(f"{row['per_class'][class_name]['f1']:.3f}")
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three official-style FOMO experiments focused on each class")
    parser.add_argument("--data-yaml", type=str, default="fomoPy/data/augmented_rf_smoke/data.yaml")
    parser.add_argument("--labels-dir", type=str, default="fomoPy/data/augmented_rf_smoke/labels")
    parser.add_argument("--classes", type=str, default="Tennis,Tennis player,Tennis racket")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--mobilenet-alpha", type=float, default=0.35)
    parser.add_argument("--head-filters", type=int, default=32)
    parser.add_argument("--object-weight", type=float, default=80.0)
    parser.add_argument("--background-weight", type=float, default=2.5)
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
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--class-thresholds", type=str, default="0.50,0.60,0.60")
    parser.add_argument("--base-out-dir", type=str, default="fomoPy/outputs/focus_sweep_100")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    classes = parse_csv(args.classes)
    if not classes:
        classes = list(DEFAULT_CLASSES)

    base_out_dir = (repo_root / args.base_out_dir).resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for focus_class in classes:
        class_slug = slugify(focus_class)
        experiment_dir = base_out_dir / class_slug
        report_path = experiment_dir / "eval_official.json"

        if args.skip_existing and report_path.exists():
            print(f"Skipping existing experiment: {focus_class}")
        else:
            train_command = [
                sys.executable,
                "fomoPy/train_official.py",
                "--data-yaml",
                args.data_yaml,
                "--labels-dir",
                args.labels_dir,
                "--epochs",
                str(args.epochs),
                "--warmup-epochs",
                str(args.warmup_epochs),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.image_size),
                "--grid-size",
                str(args.grid_size),
                "--mobilenet-alpha",
                str(args.mobilenet_alpha),
                "--head-filters",
                str(args.head_filters),
                "--object-weight",
                str(args.object_weight),
                "--background-weight",
                str(args.background_weight),
                "--focus-classes",
                focus_class,
                "--focus-multiplier",
                str(args.focus_multiplier),
                "--negative-crops-per-image",
                str(args.negative_crops_per_image),
                "--negative-crop-min-scale",
                str(args.negative_crop_min_scale),
                "--negative-crop-max-scale",
                str(args.negative_crop_max_scale),
                "--negative-avoid-expand-ratio",
                str(args.negative_avoid_expand_ratio),
                "--negative-hard-ratio",
                str(args.negative_hard_ratio),
                "--negative-hard-gap-ratio",
                str(args.negative_hard_gap_ratio),
                "--pretrained",
                args.pretrained,
                "--seed",
                str(args.seed),
                "--out-dir",
                str(experiment_dir),
            ]
            if args.allow_random_init_fallback:
                train_command.append("--allow-random-init-fallback")

            eval_command = [
                sys.executable,
                "fomoPy/eval_official.py",
                "--model",
                str(experiment_dir / "fomo_official_int8.tflite"),
                "--data-yaml",
                args.data_yaml,
                "--labels-dir",
                args.labels_dir,
                "--image-size",
                str(args.image_size),
                "--grid-size",
                str(args.grid_size),
                "--threshold",
                str(args.threshold),
                "--class-thresholds",
                args.class_thresholds,
                "--report",
                str(report_path),
            ]

            print("=" * 80)
            print(f"Running focus experiment: {focus_class}")
            print("=" * 80)
            run_command(train_command, workdir=repo_root)
            run_command(eval_command, workdir=repo_root)

        metrics = load_eval_summary(report_path)
        row = {
            "focus_class": focus_class,
            "out_dir": str(experiment_dir),
            "report_path": str(report_path),
            "overall": metrics["overall"],
            "per_class": metrics["per_class"],
        }
        summary_rows.append(row)

        print(
            f"[{focus_class}] "
            f"P={row['overall']['precision']:.3f}, "
            f"R={row['overall']['recall']:.3f}, "
            f"F1={row['overall']['f1']:.3f}"
        )

    summary_payload = {
        "data_yaml": args.data_yaml,
        "labels_dir": args.labels_dir,
        "epochs": args.epochs,
        "threshold": args.threshold,
        "class_thresholds": args.class_thresholds,
        "rows": summary_rows,
    }
    save_summary_json(base_out_dir / "summary.json", summary_payload)
    save_summary_markdown(base_out_dir / "summary.md", summary_rows, class_names=classes)

    print("")
    print("Saved:")
    print(f"  {base_out_dir / 'summary.json'}")
    print(f"  {base_out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
