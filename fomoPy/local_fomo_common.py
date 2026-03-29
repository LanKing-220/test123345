from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: Path
    train_img_dir: Path
    val_img_dir: Path
    train_label_dir: Path
    val_label_dir: Path
    class_names: List[str]
    num_classes: int


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


def collect_image_paths(image_dir: Path) -> List[Path]:
    image_paths = []
    for pattern in IMAGE_GLOBS:
        image_paths.extend(image_dir.glob(pattern))
    return sorted(image_paths)


def resize_rgb_image(image_path: Path, image_size: int, normalize: bool) -> np.ndarray:
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    if normalize:
        img = img / 255.0
    return img


def resolve_dataset_layout(workspace: Path, data_yaml: Path, labels_root: Path) -> DatasetLayout:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg_path = cfg.get("path")
    dataset_root = Path(cfg_path) if Path(cfg_path).is_absolute() else (workspace / cfg_path).resolve()

    return DatasetLayout(
        dataset_root=dataset_root,
        train_img_dir=dataset_root / cfg["train"],
        val_img_dir=dataset_root / cfg["val"],
        train_label_dir=(labels_root / "training").resolve(),
        val_label_dir=(labels_root / "testing").resolve(),
        class_names=list(cfg.get("names", [])),
        num_classes=int(cfg["nc"]),
    )
