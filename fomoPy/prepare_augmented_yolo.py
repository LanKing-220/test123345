import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from prepare_tennis_export import collect_class_names, load_boxes, to_yolo_line


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTERNAL_CLASS_ALIASES = {
    "tennis": "Tennis",
    "tennis ball": "Tennis",
    "tennisball": "Tennis",
    "ball": "Tennis",
    "tennis player": "Tennis player",
    "player": "Tennis player",
    "player1": "Tennis player",
    "player 1": "Tennis player",
    "player2": "Tennis player",
    "player 2": "Tennis player",
    "person": "Tennis player",
    "tennis racket": "Tennis racket",
    "tennis racquet": "Tennis racket",
    "tennisracket": "Tennis racket",
    "tennisracquet": "Tennis racket",
    "racket": "Tennis racket",
    "racquet": "Tennis racket",
}


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", name.strip().lower())).strip()


def safe_rmtree(path: Path, workspace: Path) -> None:
    resolved = path.resolve()
    workspace_resolved = workspace.resolve()
    if workspace_resolved not in resolved.parents and resolved != workspace_resolved:
        raise ValueError(f"refusing to remove directory outside workspace: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def iter_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def read_yolo_rows(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO row in {path}: {line}")
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def write_yolo_rows(path: Path, rows: Sequence[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(path.parent)
    if rows:
        content = "\n".join(
            f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            for cid, cx, cy, w, h in rows
        ) + "\n"
    else:
        content = ""
    path.write_text(content, encoding="utf-8")


def load_yaml_names(data_yaml_path: Path) -> List[str]:
    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, dict):
        ordered = [name for _, name in sorted((int(k), v) for k, v in names.items())]
        return ordered
    if isinstance(names, list):
        return [str(v) for v in names]
    raise ValueError(f"unable to parse class names from {data_yaml_path}")


def parse_external_arg(value: str) -> Tuple[Path, int]:
    if "::" in value:
        raw_path, raw_limit = value.rsplit("::", 1)
        return Path(raw_path), int(raw_limit)
    return Path(value), 0


def unzip_if_needed(source: Path, cache_root: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"external dataset path does not exist: {source}")
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == ".zip":
        target = cache_root / source.stem
        if target.exists():
            return target
        ensure_dir(cache_root)
        shutil.unpack_archive(str(source), str(target))
        return target
    raise ValueError(f"external dataset must be a directory or .zip file: {source}")


def locate_data_yaml(root: Path) -> Path:
    direct = root / "data.yaml"
    if direct.exists():
        return direct
    matches = sorted(root.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(f"could not find data.yaml under {root}")
    return matches[0]


def locate_split_dirs(root: Path, split: str) -> Optional[Tuple[Path, Path]]:
    candidates = [
        (root / split / "images", root / split / "labels"),
        (root / split, root / "labels" / split),
        (root / "images" / split, root / "labels" / split),
    ]
    for image_dir, label_dir in candidates:
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir
    return None


def canonical_target_name(external_name: str, canonical_names: Sequence[str]) -> Optional[str]:
    normalized = normalize_name(external_name)
    canonical_lookup = {normalize_name(name): name for name in canonical_names}
    if normalized in canonical_lookup:
        return canonical_lookup[normalized]
    alias_target = EXTERNAL_CLASS_ALIASES.get(normalized)
    if alias_target and alias_target in canonical_names:
        return alias_target
    return None


def remap_rows(
    rows: Sequence[Tuple[int, float, float, float, float]],
    external_names: Sequence[str],
    canonical_names: Sequence[str],
    canonical_to_id: Dict[str, int],
) -> Tuple[List[Tuple[int, float, float, float, float]], List[str]]:
    remapped: List[Tuple[int, float, float, float, float]] = []
    unmapped: List[str] = []
    for cid, cx, cy, w, h in rows:
        if cid < 0 or cid >= len(external_names):
            raise ValueError(f"class id {cid} out of range for external dataset with {len(external_names)} classes")
        external_name = external_names[cid]
        target_name = canonical_target_name(external_name, canonical_names)
        if target_name is None:
            unmapped.append(external_name)
            continue
        remapped.append((canonical_to_id[target_name], cx, cy, w, h))
    return remapped, unmapped


def copy_base_split(
    split_name: str,
    image_dir: Path,
    box_map: Dict[str, List[dict]],
    image_out_dir: Path,
    label_out_dir: Path,
    class_to_id: Dict[str, int],
) -> Dict[str, object]:
    images_written = 0
    object_counts = {name: 0 for name in class_to_id}

    for image_path in iter_images(image_dir):
        out_name = image_path.name
        copy_file(image_path, image_out_dir / out_name)

        boxes = box_map.get(image_path.name, [])
        width = None
        height = None
        if boxes:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
        yolo_rows: List[str] = []
        if boxes:
            assert width is not None and height is not None
            yolo_rows = [to_yolo_line(box, class_to_id, width, height) for box in boxes]
            for box in boxes:
                object_counts[box["label"]] += 1

        label_path = label_out_dir / f"{image_path.stem}.txt"
        ensure_dir(label_path.parent)
        label_path.write_text(("\n".join(yolo_rows) + "\n") if yolo_rows else "", encoding="utf-8")
        images_written += 1

    return {
        "source": f"base_{split_name}",
        "images": images_written,
        "objects": object_counts,
    }


def sanitize_source_name(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower() or "external"


def choose_paths(paths: List[Path], limit: int, seed: int) -> List[Path]:
    if limit <= 0 or limit >= len(paths):
        return paths
    rng = random.Random(seed)
    chosen = sorted(rng.sample(paths, limit), key=lambda p: p.name.lower())
    return chosen


def merge_external_dataset(
    dataset_path: Path,
    image_out_dir: Path,
    label_out_dir: Path,
    canonical_names: Sequence[str],
    canonical_to_id: Dict[str, int],
    include_test: bool,
    max_images: int,
    seed: int,
    cache_root: Path,
) -> Dict[str, object]:
    extracted_root = unzip_if_needed(dataset_path, cache_root=cache_root)
    data_yaml_path = locate_data_yaml(extracted_root)
    dataset_root = data_yaml_path.parent
    external_names = load_yaml_names(data_yaml_path)
    source_name = sanitize_source_name(dataset_path)
    split_names = ["train", "valid"]
    if include_test:
        split_names.append("test")

    stats = {
        "source": source_name,
        "root": str(dataset_root),
        "max_images": max_images,
        "external_classes": external_names,
        "images": 0,
        "objects": {name: 0 for name in canonical_names},
        "unmapped_classes": [],
    }

    image_jobs: List[Tuple[str, Path, Path]] = []
    for split_name in split_names:
        located = locate_split_dirs(dataset_root, split_name)
        if located is None:
            continue
        img_dir, label_dir = located
        for image_path in iter_images(img_dir):
            image_jobs.append((split_name, image_path, label_dir / f"{image_path.stem}.txt"))

    image_jobs = [(split_name, image_path, label_path) for split_name, image_path, label_path in image_jobs]
    selected_jobs = choose_paths([job[1] for job in image_jobs], max_images, seed)
    selected_lookup = {p.resolve() for p in selected_jobs}

    for idx, (split_name, image_path, label_path) in enumerate(image_jobs):
        if max_images > 0 and image_path.resolve() not in selected_lookup:
            continue

        rows = read_yolo_rows(label_path)
        remapped_rows, unmapped = remap_rows(rows, external_names, canonical_names, canonical_to_id)
        stats["unmapped_classes"].extend(unmapped)
        for cid, *_ in remapped_rows:
            stats["objects"][canonical_names[cid]] += 1

        out_stem = f"{source_name}_{split_name}_{idx:05d}_{image_path.stem}"
        copy_file(image_path, image_out_dir / f"{out_stem}{image_path.suffix.lower()}")
        write_yolo_rows(label_out_dir / f"{out_stem}.txt", remapped_rows)
        stats["images"] += 1

    stats["unmapped_classes"] = sorted(set(stats["unmapped_classes"]))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merged YOLO dataset from the base tennis export plus external Roboflow datasets")
    parser.add_argument("--base-export-root", type=str, default="data/tennis-export")
    parser.add_argument(
        "--external",
        action="append",
        default=[],
        help="Path to a Roboflow YOLO export directory or zip. Optional '::N' suffix limits imported images from that dataset.",
    )
    parser.add_argument("--include-external-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="fomoPy/data/augmented_rf")
    args = parser.parse_args()

    workspace = Path.cwd()
    base_export_root = (workspace / args.base_export_root).resolve()
    out_root = (workspace / args.out_dir).resolve()
    cache_root = (workspace / "fomoPy" / "data" / "_external_cache").resolve()

    train_image_dir = base_export_root / "training"
    test_image_dir = base_export_root / "testing"
    train_boxes = load_boxes(train_image_dir / "bounding_boxes.labels")
    test_boxes = load_boxes(test_image_dir / "bounding_boxes.labels")
    canonical_names = collect_class_names(train_boxes, test_boxes)
    canonical_to_id = {name: idx for idx, name in enumerate(canonical_names)}

    external_inputs: List[Tuple[Path, int]] = []
    for raw_external in args.external:
        dataset_path, max_images = parse_external_arg(raw_external)
        resolved_dataset_path = dataset_path if dataset_path.is_absolute() else (workspace / dataset_path).resolve()
        if not resolved_dataset_path.exists():
            raise FileNotFoundError(f"external dataset path does not exist: {resolved_dataset_path}")
        external_inputs.append((resolved_dataset_path, max_images))

    safe_rmtree(out_root, workspace=workspace)
    train_image_out = out_root / "images" / "training"
    test_image_out = out_root / "images" / "testing"
    train_label_out = out_root / "labels" / "training"
    test_label_out = out_root / "labels" / "testing"
    ensure_dir(train_image_out)
    ensure_dir(test_image_out)
    ensure_dir(train_label_out)
    ensure_dir(test_label_out)

    summary = {
        "classes": canonical_names,
        "sources": [],
    }
    summary["sources"].append(
        copy_base_split(
            split_name="training",
            image_dir=train_image_dir,
            box_map=train_boxes,
            image_out_dir=train_image_out,
            label_out_dir=train_label_out,
            class_to_id=canonical_to_id,
        )
    )
    summary["sources"].append(
        copy_base_split(
            split_name="testing",
            image_dir=test_image_dir,
            box_map=test_boxes,
            image_out_dir=test_image_out,
            label_out_dir=test_label_out,
            class_to_id=canonical_to_id,
        )
    )

    for resolved_dataset_path, max_images in external_inputs:
        external_stats = merge_external_dataset(
            dataset_path=resolved_dataset_path,
            image_out_dir=train_image_out,
            label_out_dir=train_label_out,
            canonical_names=canonical_names,
            canonical_to_id=canonical_to_id,
            include_test=args.include_external_test,
            max_images=max_images,
            seed=args.seed,
            cache_root=cache_root,
        )
        summary["sources"].append(external_stats)

    data_yaml = {
        "path": str((out_root / "images").as_posix()),
        "train": "training",
        "val": "testing",
        "nc": len(canonical_names),
        "names": canonical_names,
    }
    (out_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    totals = {
        "train_images": len(iter_images(train_image_out)),
        "test_images": len(iter_images(test_image_out)),
        "train_objects": {name: 0 for name in canonical_names},
        "test_objects": {name: 0 for name in canonical_names},
    }
    for entry in summary["sources"]:
        source_name = str(entry["source"])
        target_key = "test_objects" if source_name == "base_testing" else "train_objects"
        for class_name, count in entry["objects"].items():
            totals[target_key][class_name] += int(count)
    summary["totals"] = totals

    summary_path = out_root / "manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("classes:", canonical_names)
    print("saved data yaml:", out_root / "data.yaml")
    print("saved manifest:", summary_path)
    print("train images:", totals["train_images"])
    print("test images:", totals["test_images"])
    print("train objects:", totals["train_objects"])
    print("test objects:", totals["test_objects"])
    for entry in summary["sources"]:
        if entry["source"] in {"base_training", "base_testing"}:
            continue
        print(
            f"external {entry['source']}: images={entry['images']}, "
            f"objects={entry['objects']}, unmapped={entry['unmapped_classes']}"
        )


if __name__ == "__main__":
    main()
