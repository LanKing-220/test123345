import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_labels(labels_path: Path) -> List[str]:
    return [x.strip() for x in labels_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def to_ascii_loadable_path(src: Path) -> Path:
    src = src.resolve()
    try:
        str(src).encode("ascii")
        return src
    except UnicodeEncodeError:
        temp_dir = Path(tempfile.gettempdir()) / "fomo_tflite_ascii"
        temp_dir.mkdir(parents=True, exist_ok=True)
        dst = temp_dir / src.name
        shutil.copyfile(src, dst)
        return dst


def load_interpreter(model_path: Path):
    model_path = to_ascii_loadable_path(model_path)
    try:
        import tensorflow as tf  # type: ignore

        return tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))


def preprocess_rgb_image(image_path: Path, input_h: int, input_w: int) -> np.ndarray:
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_AREA)
    return img


def quantize_input(img_rgb: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        x = img_rgb.astype(np.float32) / 255.0
        return np.expand_dims(x, axis=0)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        raise ValueError("Quantized model has invalid input quantization scale")

    x = (img_rgb.astype(np.float32) / 255.0) / scale + zero_point
    x = np.clip(x, -128, 127).astype(np.int8)
    return np.expand_dims(x, axis=0)


def dequantize_output(y: np.ndarray, output_detail: dict) -> np.ndarray:
    if y.dtype == np.float32:
        return y

    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if not scale:
        raise ValueError("Quantized model has invalid output quantization scale")

    return (y.astype(np.float32) - zero_point) * scale


def connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int, np.ndarray]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for comp_id in range(1, n):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        roi_mask = labels[y : y + h, x : x + w] == comp_id
        out.append((x, y, w, h, roi_mask))
    return out


def fomo_like_postprocess(
    output_grid: np.ndarray,
    min_confidence: float,
    roi: Tuple[int, int, int, int],
) -> Dict[int, List[Tuple[int, int, int, int, float]]]:
    oh, ow, oc = output_grid.shape

    x_scale = roi[2] / ow
    y_scale = roi[3] / oh
    scale = min(x_scale, y_scale)

    # Keep aligned with ei_object_detection.py offset logic.
    x_offset = ((roi[2] - (ow * scale)) / 2.0) + roi[0]
    y_offset = ((roi[3] - (ow * scale)) / 2.0) + roi[1]

    threshold = float(min_confidence)
    detections: Dict[int, List[Tuple[int, int, int, int, float]]] = {i: [] for i in range(oc)}

    for class_idx in range(oc):
        class_map = output_grid[:, :, class_idx]
        mask = class_map >= threshold
        for x, y, w, h, roi_mask in connected_components(mask):
            roi_values = class_map[y : y + h, x : x + w]
            score = float(np.mean(roi_values[roi_mask]))

            px = int((x * scale) + x_offset)
            py = int((y * scale) + y_offset)
            pw = int(w * scale)
            ph = int(h * scale)
            detections[class_idx].append((px, py, pw, ph, score))

    return detections


def run_model(model_path: Path, image_path: Path, min_conf: float):
    interpreter = load_interpreter(model_path)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    _, h, w, _ = input_detail["shape"]
    img_rgb = preprocess_rgb_image(image_path, int(h), int(w))

    x = quantize_input(img_rgb, input_detail)
    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_detail["index"])
    y = dequantize_output(y, output_detail)
    grid = y[0]

    roi = (0, 0, img_rgb.shape[1], img_rgb.shape[0])
    detections = fomo_like_postprocess(grid, min_conf, roi)
    return detections, tuple(output_detail["shape"])
