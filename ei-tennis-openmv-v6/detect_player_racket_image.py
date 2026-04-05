import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = Path(__file__).with_name("trained.tflite")
LABELS_PATH = Path(__file__).with_name("labels.txt")

THRESH_HEATMAP = 0.20
THRESH_PLAYER = 0.30
THRESH_RACKET = 0.35

BLUE = (255, 0, 0)   # OpenCV uses BGR
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_labels(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_interpreter(model_path: Path):
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    except ValueError:
        # TensorFlow Lite on Windows can fail to open models under non-ASCII paths.
        if not model_path.exists():
            raise
        interpreter = tf.lite.Interpreter(model_content=model_path.read_bytes())
    interpreter.allocate_tensors()
    return interpreter


def resolve_labels_path(model_path: Path, labels_arg):
    if labels_arg:
        return Path(labels_arg)

    model_labels = model_path.with_name("labels.txt")
    if model_labels.exists():
        return model_labels

    return LABELS_PATH


def preprocess_image(image_bgr: np.ndarray, input_width: int, input_height: int):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_width, input_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def run_tflite(interpreter, image_input: np.ndarray):
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    batch = np.expand_dims(image_input, axis=0)
    if input_detail["dtype"] == np.int8:
        scale, zero_point = input_detail["quantization"]
        batch = np.round(batch / scale + zero_point).astype(np.int8)
    else:
        batch = batch.astype(input_detail["dtype"])

    interpreter.set_tensor(input_detail["index"], batch)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]

    if output_detail["dtype"] == np.int8:
        scale, zero_point = output_detail["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
    else:
        output = output.astype(np.float32)

    return output


def score_region(roi: np.ndarray, score_mode: str):
    flat = roi.reshape(-1)
    if flat.size == 0:
        return 0.0
    if score_mode == "max":
        return float(np.max(flat))
    if score_mode == "top3":
        k = min(3, flat.size)
        return float(np.mean(np.sort(flat)[-k:]))
    if score_mode == "top5":
        k = min(5, flat.size)
        return float(np.mean(np.sort(flat)[-k:]))
    return float(np.mean(flat))


def decode_fomo(
    output: np.ndarray,
    image_width: int,
    image_height: int,
    heatmap_thresh: float = THRESH_HEATMAP,
    score_mode: str = "mean",
):
    grid_h, grid_w, num_channels = output.shape
    results = [[] for _ in range(num_channels)]

    x_scale = image_width / grid_w
    y_scale = image_height / grid_h

    for class_id in range(num_channels):
        channel = output[:, :, class_id]
        mask = (channel >= heatmap_thresh).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            if w <= 0 or h <= 0:
                continue

            roi = channel[y : y + h, x : x + w]
            score = score_region(roi, score_mode)

            results[class_id].append(
                (
                    int(x * x_scale),
                    int(y * y_scale),
                    max(1, int(w * x_scale)),
                    max(1, int(h * y_scale)),
                    score,
                )
            )

    return results


def choose_best_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda item: (item[4], (item[2] * item[3])))


def print_heatmap_stats(output: np.ndarray, labels):
    for class_id in range(output.shape[2]):
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        channel = output[:, :, class_id]
        print(
            f"heatmap[{class_id}] {label}: "
            f"max={float(channel.max()):.6f} mean={float(channel.mean()):.6f}"
        )


def draw_player(image_bgr: np.ndarray, det):
    x, y, w, h, score = det
    cx = x + (w // 2)
    cy = y + (h // 2)
    radius = max(20, int(max(w, h) * 0.45))
    cv2.circle(image_bgr, (cx, cy), radius, BLUE, 2)
    cv2.drawMarker(
        image_bgr,
        (cx, cy),
        WHITE,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    cv2.putText(
        image_bgr,
        f"Player {score:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        YELLOW,
        2,
        cv2.LINE_AA,
    )


def draw_racket(image_bgr: np.ndarray, det):
    x, y, w, h, score = det
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), RED, 2)
    cv2.putText(
        image_bgr,
        f"Racket {score:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        YELLOW,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="识别网球运动员和网球拍，并输出标注结果图。")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--output", help="输出图片路径，默认保存在输入图片同目录")
    parser.add_argument("--model", help="模型路径，默认使用脚本目录下的 trained.tflite")
    parser.add_argument("--labels", help="标签文件路径，默认优先跟随模型目录下的 labels.txt")
    parser.add_argument("--player-thresh", type=float, default=THRESH_PLAYER, help="人物类别阈值")
    parser.add_argument("--racket-thresh", type=float, default=THRESH_RACKET, help="球拍类别阈值")
    parser.add_argument("--heatmap-thresh", type=float, default=THRESH_HEATMAP, help="热力图连通域阈值")
    parser.add_argument(
        "--score-mode",
        choices=("mean", "max", "top3", "top5"),
        default="mean",
        help="连通域置信度计算方式",
    )
    parser.add_argument("--debug", action="store_true", help="打印各类别热力图统计")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_player_racket.jpg")
    model_path = Path(args.model) if args.model else MODEL_PATH
    labels_path = resolve_labels_path(model_path, args.labels)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels not found: {labels_path}")

    labels = load_labels(labels_path)
    interpreter = load_interpreter(model_path)

    input_detail = interpreter.get_input_details()[0]
    _, input_h, input_w, _ = input_detail["shape"]

    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    image_input = preprocess_image(image_bgr, input_w, input_h)
    output = run_tflite(interpreter, image_input)
    if args.debug:
        print_heatmap_stats(output, labels)
    predictions = decode_fomo(
        output,
        image_bgr.shape[1],
        image_bgr.shape[0],
        heatmap_thresh=args.heatmap_thresh,
        score_mode=args.score_mode,
    )

    player_detections = []
    racket_detections = []

    for class_id, detection_list in enumerate(predictions):
        if class_id >= len(labels):
            continue
        label = labels[class_id].lower()
        if "player" in label:
            player_detections.extend([d for d in detection_list if d[4] >= args.player_thresh])
        elif "racket" in label:
            racket_detections.extend([d for d in detection_list if d[4] >= args.racket_thresh])

    best_player = choose_best_detection(player_detections)
    best_racket = choose_best_detection(racket_detections)

    if best_player is not None:
        draw_player(image_bgr, best_player)
    if best_racket is not None:
        draw_racket(image_bgr, best_racket)

    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("failed to encode output image")
    output_path.write_bytes(encoded.tobytes())

    print("model:", model_path)
    print("labels:", labels)
    print("best_player:", best_player)
    print("best_racket:", best_racket)
    print("saved:", output_path)


if __name__ == "__main__":
    main()
