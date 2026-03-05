import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def iter_images(image_dir: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for p in image_dir.glob(ext):
            yield p


def representative_dataset(image_dir: Path, image_size: int, limit: int):
    image_paths = sorted(iter_images(image_dir))[:limit]

    def _gen():
        for image_path in image_paths:
            img = imread_unicode(image_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            yield [np.expand_dims(img, axis=0)]

    return _gen


def main() -> None:
    parser = argparse.ArgumentParser(description="Export int8 TFLite model for fomo")
    parser.add_argument("--keras", required=True, help="Path to .keras model")
    parser.add_argument("--train-images", default="data/yolo_dataset/images/train", help="Representative dataset images dir")
    parser.add_argument("--image-size", type=int, default=98)
    parser.add_argument("--rep-limit", type=int, default=300)
    parser.add_argument("--out", default="outputs/fomo_short_run/fomo_int8.tflite")
    args = parser.parse_args()

    keras_path = Path(args.keras).resolve()
    train_images = Path(args.train_images).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export does not require training compile state; skip custom loss deserialization.
    model = tf.keras.models.load_model(keras_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(train_images, args.image_size, args.rep_limit)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)

    print(f"Saved int8 model: {out_path}")


if __name__ == "__main__":
    main()
