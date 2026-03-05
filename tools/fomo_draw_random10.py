import argparse
import random
from pathlib import Path

import cv2

from fomo_infer_utils import imread_unicode, load_labels, run_model


def is_racket(name: str) -> bool:
    s = name.lower()
    return "racket" in s


def is_tennis_ball(name: str) -> bool:
    s = name.lower()
    return ("tennis" in s) and ("racket" not in s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw detections on random 10 images")
    parser.add_argument("--image-dir", default="data/yolo_dataset/images/val")
    parser.add_argument("--model", default="ei-tennis-openmv-v6/trained.tflite")
    parser.add_argument("--labels", default="ei-tennis-openmv-v6/labels.txt")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=98)
    parser.add_argument("--save-dir", default="outputs/fomo_short_run/random10_drawn")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    model_path = Path(args.model).resolve()
    labels = load_labels(Path(args.labels).resolve())
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        all_images.extend(image_dir.glob(ext))
    all_images = sorted(all_images)

    if len(all_images) < args.count:
        raise ValueError(f"Not enough images in {image_dir}. found={len(all_images)}, need={args.count}")

    rng = random.Random(args.seed)
    picks = rng.sample(all_images, args.count)

    print(f"Using model: {model_path}")
    print(f"Drawing on {args.count} sampled images from: {image_dir}")

    for i, image_path in enumerate(picks, start=1):
        src = imread_unicode(image_path)
        if src is None:
            print(f"[skip] {image_path.name} failed to read")
            continue

        canvas = cv2.resize(src, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        detections, out_shape = run_model(model_path, image_path, args.min_conf)

        for class_idx, class_name in enumerate(labels):
            if class_idx == 0:
                continue
            items = detections.get(class_idx, [])
            if not items:
                continue

            for x, y, w, h, score in items:
                cx = x + (w // 2)
                cy = y + (h // 2)

                if is_tennis_ball(class_name):
                    cv2.circle(canvas, (cx, cy), 8, (0, 255, 0), 2)
                    cv2.putText(
                        canvas,
                        f"T:{score:.2f}",
                        (max(0, cx - 18), max(12, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                elif is_racket(class_name):
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        canvas,
                        f"R:{score:.2f}",
                        (max(0, x), max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        out_name = f"{i:02d}_{image_path.stem}_drawn.jpg"
        out_path = save_dir / out_name
        ok = cv2.imencode(".jpg", canvas)[1]
        ok.tofile(str(out_path))
        print(f"[{i}] {image_path.name} | output_shape={out_shape} -> {out_path.name}")

    print(f"Saved drawn images to: {save_dir}")


if __name__ == "__main__":
    main()
