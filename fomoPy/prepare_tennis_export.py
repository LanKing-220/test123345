import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import yaml


def load_boxes(path: Path) -> Dict[str, List[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["boundingBoxes"]


def collect_class_names(*box_maps: Dict[str, List[dict]]) -> List[str]:
    names = sorted({box["label"] for box_map in box_maps for boxes in box_map.values() for box in boxes})
    return names


def image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def clamp01(value: float) -> float:
    return min(0.999999, max(0.0, value))


def to_yolo_line(box: dict, class_to_id: Dict[str, int], width: int, height: int) -> str:
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["width"])
    h = float(box["height"])

    cx = clamp01((x + w / 2.0) / width)
    cy = clamp01((y + h / 2.0) / height)
    nw = clamp01(w / width)
    nh = clamp01(h / height)

    class_id = class_to_id[box["label"]]
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def write_split(
    split_name: str,
    image_dir: Path,
    labels_out_dir: Path,
    box_map: Dict[str, List[dict]],
    class_to_id: Dict[str, int],
) -> int:
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        stem = image_path.stem
        label_path = labels_out_dir / f"{stem}.txt"
        boxes = box_map.get(image_path.name, [])

        if boxes:
            width, height = image_size(image_path)
            lines = [to_yolo_line(box, class_to_id, width, height) for box in boxes]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")

        count += 1

    print(f"{split_name}: wrote {count} label files to {labels_out_dir}")
    return count


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    export_root = repo_root / "data" / "tennis-export"
    train_image_dir = export_root / "training"
    test_image_dir = export_root / "testing"

    train_boxes = load_boxes(train_image_dir / "bounding_boxes.labels")
    test_boxes = load_boxes(test_image_dir / "bounding_boxes.labels")
    class_names = collect_class_names(train_boxes, test_boxes)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    workspace_root = Path(__file__).resolve().parent
    yolo_root = workspace_root / "data"
    labels_root = yolo_root / "yolo_labels"
    dataset_root = yolo_root / "yolo_dataset"

    write_split("training", train_image_dir, labels_root / "training", train_boxes, class_to_id)
    write_split("testing", test_image_dir, labels_root / "testing", test_boxes, class_to_id)

    dataset_root.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "path": str(export_root.as_posix()),
        "train": "training",
        "val": "testing",
        "nc": len(class_names),
        "names": class_names,
    }
    (dataset_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print("classes:", class_names)
    print("saved:", dataset_root / "data.yaml")


if __name__ == "__main__":
    main()
