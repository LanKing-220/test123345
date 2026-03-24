# OpenMV FOMO runtime for the tennis model.
#
# Expected files on the OpenMV device:
# - trained.tflite
# - labels.txt
# - main_openmv.py (rename to main.py if you want it to auto-run)
# - pid.py

import gc
import math
import time
import uos

import image
import ml
import sensor
from pyb import Pin, Timer

from pid import PID


MODEL_PATH = "trained.tflite"
LABELS_PATH = "labels.txt"

FRAME_SIZE = sensor.QVGA
WINDOW_SIZE = (240, 240)
MIN_CONFIDENCE = 0.5

TARGET_LABEL = "Tennis"
DRAW_RADIUS = 12

# Servo limits.
PAN_LIMIT = (30.0, 150.0)
TILT_LIMIT = (80.0, 150.0)
PAN_START = 90.0
TILT_START = 130.0

# PID gains. Tune on-device if needed.
PAN_PID = PID(p=0.08, i=0.0, d=0.015, imax=10)
TILT_PID = PID(p=0.08, i=0.0, d=0.015, imax=10)

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]


pan_angle = PAN_START
tilt_angle = TILT_START


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def setup_sensor():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(FRAME_SIZE)
    sensor.set_windowing(WINDOW_SIZE)
    sensor.set_auto_whitebal(False)
    sensor.skip_frames(time=2000)


def setup_servos():
    p1 = Pin("P1", Pin.OUT_PP)
    p9 = Pin("P9", Pin.OUT_PP)

    p1_pulse = Timer(12)
    p1_main = Timer(13, freq=50)
    p9_pulse = Timer(14)
    p9_main = Timer(15, freq=50)

    def p1_isr0(_):
        p1.low()
        p1_pulse.deinit()

    def p1_isr(_):
        psr = int(pan_angle * 1000 / 9 + 5000) - 1
        p1.high()
        p1_pulse.init(prescaler=psr, period=23)
        p1_pulse.callback(p1_isr0)

    def p9_isr0(_):
        p9.low()
        p9_pulse.deinit()

    def p9_isr(_):
        psr = int(tilt_angle * 1000 / 9 + 5000) - 1
        p9.high()
        p9_pulse.init(prescaler=psr, period=23)
        p9_pulse.callback(p9_isr0)

    p1_main.callback(p1_isr)
    p9_main.callback(p9_isr)


def load_model():
    try:
        return ml.Model(
            MODEL_PATH,
            load_to_fb=uos.stat(MODEL_PATH)[6] > (gc.mem_free() - (64 * 1024)),
        )
    except Exception as e:
        raise Exception('Failed to load "trained.tflite": ' + str(e))


def load_labels():
    try:
        return [line.rstrip("\n") for line in open(LABELS_PATH)]
    except Exception as e:
        raise Exception('Failed to load "labels.txt": ' + str(e))


threshold_list = [(math.ceil(MIN_CONFIDENCE * 255), 255)]


def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh
    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (oh * scale)) / 2) + inputs[0].roi[1]

    detections = [[] for _ in range(oc)]
    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list,
            x_stride=1,
            y_stride=1,
            area_threshold=1,
            pixels_threshold=1,
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            detections[i].append((x, y, w, h, score))
    return detections


def find_best_target(predictions, labels):
    if TARGET_LABEL not in labels:
        return None

    target_index = labels.index(TARGET_LABEL)
    if target_index >= len(predictions):
        return None

    best = None
    best_score = None
    for x, y, w, h, score in predictions[target_index]:
        if (best_score is None) or (score > best_score):
            best = (x, y, w, h, score)
            best_score = score
    return best


def update_servos(target_cx, target_cy, width, height):
    global pan_angle, tilt_angle

    err_x = (width / 2) - target_cx
    err_y = target_cy - (height / 2)

    pan_angle += PAN_PID.get_pid(err_x, 1.0)
    tilt_angle += TILT_PID.get_pid(err_y, 1.0)

    pan_angle = clamp(pan_angle, PAN_LIMIT[0], PAN_LIMIT[1])
    tilt_angle = clamp(tilt_angle, TILT_LIMIT[0], TILT_LIMIT[1])


def draw_all_detections(img, predictions, labels):
    for i, detection_list in enumerate(predictions):
        if i == 0:
            continue
        if len(detection_list) == 0:
            continue

        color = COLORS[i % len(COLORS)]
        for x, y, w, h, score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            img.draw_circle((center_x, center_y, DRAW_RADIUS), color=color)
            img.draw_string(
                x,
                max(0, y - 12),
                "%s %.2f" % (labels[i], score),
                color=color,
                mono_space=False,
            )


def main():
    setup_sensor()
    setup_servos()

    net = load_model()
    labels = load_labels()
    clock = time.clock()

    print("OpenMV FOMO runtime started")
    print("Target label:", TARGET_LABEL)

    while True:
        clock.tick()
        img = sensor.snapshot()

        predictions = net.predict([img], callback=fomo_post_process)
        draw_all_detections(img, predictions, labels)

        best = find_best_target(predictions, labels)
        if best is not None:
            x, y, w, h, score = best
            cx = math.floor(x + (w / 2))
            cy = math.floor(y + (h / 2))

            update_servos(cx, cy, img.width(), img.height())

            img.draw_cross(cx, cy, color=(0, 255, 0), size=12, thickness=2)
            img.draw_string(
                2,
                2,
                "track %s %.2f" % (TARGET_LABEL, score),
                color=(0, 255, 0),
                mono_space=False,
            )

            print("target", TARGET_LABEL, "x", cx, "y", cy, "score", score)
        else:
            img.draw_string(2, 2, "target lost", color=(255, 0, 0), mono_space=False)
            print("target lost")

        img.draw_string(
            2,
            20,
            "pan %.1f tilt %.1f fps %.2f" % (pan_angle, tilt_angle, clock.fps()),
            color=(255, 255, 0),
            mono_space=False,
        )


main()
