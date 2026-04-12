# 能追着网球跑的


import gc
import math
import sys
import time
import uos

import display
import image
import ml
import sensor
from pyb import Pin, Timer
from pid import PID


MODEL_PATH = "trained.tflite"
LABELS_PATH = "labels.txt"
LCD_HINT = image.ROTATE_270

# 需要舵机追踪时打开；如果 4.8.1 下再次出现白屏，可先改回 False 做隔离。
ENABLE_SERVOS = True
PAN_SERVO_PIN = "P1"
TILT_SERVO_PIN = "P9"

THRESH_TENNIS = 0.35
THRESH_PLAYER = 0.40
THRESH_RACKET = 0.35

GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

lcd = None
labels = None
net = None
tracked_tennis = None
p1 = None
p9 = None
p1_tim_pluse = None
p1_tim_main = None
p9_tim_pluse = None
p9_tim_main = None

TRACK_MAX_MISS = 8
TRACK_GATE_MIN = 28
TRACK_SMOOTH_OLD_NUM = 7
TRACK_SMOOTH_NEW_NUM = 3

pan_angle = 90.0
tilt_angle = 120.0
pan_angle_limit = [30.0, 150.0]
tilt_angle_limit = [80.0, 150.0]
SERVO_DEADBAND = 6

pan_pid = PID(p=0.07, i=0, imax=90)
tilt_pid = PID(p=0.05, i=0, imax=90)


def init_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing((240, 240))
    sensor.set_auto_whitebal(False)
    sensor.skip_frames(time=2000)


def init_lcd():
    global lcd
    time.sleep_ms(300)
    lcd = display.SPIDisplay(width=240, height=320)


def P1_ISR0(t):
    p1.low()
    p1_tim_pluse.deinit()


def P1_ISR(t):
    psr = int(pan_angle * 1000 / 9 + 5000) - 1
    p1.high()
    p1_tim_pluse.init(prescaler=psr, period=23)
    p1_tim_pluse.callback(P1_ISR0)


def P9_ISR0(t):
    p9.low()
    p9_tim_pluse.deinit()


def P9_ISR(t):
    psr = int(tilt_angle * 1000 / 9 + 5000) - 1
    p9.high()
    p9_tim_pluse.init(prescaler=psr, period=23)
    p9_tim_pluse.callback(P9_ISR0)


def init_servos():
    global p1, p9, p1_tim_pluse, p1_tim_main, p9_tim_pluse, p9_tim_main

    if not ENABLE_SERVOS:
        return

    p1 = Pin(PAN_SERVO_PIN, Pin.OUT_PP)
    p9 = Pin(TILT_SERVO_PIN, Pin.OUT_PP)

    p1_tim_pluse = Timer(12)
    p1_tim_main = Timer(13, freq=50)
    p1_tim_main.callback(P1_ISR)

    p9_tim_pluse = Timer(14)
    p9_tim_main = Timer(15, freq=50)
    p9_tim_main.callback(P9_ISR)


def display_frame(img):
    if lcd is not None:
        lcd.write(img, hint=LCD_HINT)


def show_message(line1, line2=None, color=YELLOW):
    img = sensor.snapshot()
    img.draw_string(2, 2, line1, color=color, mono_space=False)
    if line2:
        img.draw_string(2, 20, line2, color=color, mono_space=False)
    display_frame(img)


def halt_with_error(title, err):
    try:
        img = sensor.snapshot()
        img.draw_string(2, 2, title, color=RED, mono_space=False)
        img.draw_string(2, 20, str(err), color=YELLOW, mono_space=False)
        display_frame(img)
    except Exception:
        pass
    sys.print_exception(err)
    while True:
        time.sleep_ms(1000)


def threshold_for_class(index):
    if index == 1:
        return THRESH_TENNIS
    if index == 2:
        return THRESH_PLAYER
    if index == 3:
        return THRESH_RACKET
    return 0.50


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def smooth_value(prev_v, curr_v):
    return ((prev_v * TRACK_SMOOTH_OLD_NUM) + (curr_v * TRACK_SMOOTH_NEW_NUM)) // (
        TRACK_SMOOTH_OLD_NUM + TRACK_SMOOTH_NEW_NUM
    )


def choose_tennis_target(candidates):
    global tracked_tennis

    if not candidates:
        if tracked_tennis is not None:
            tracked_tennis["miss"] += 1
            if tracked_tennis["miss"] > TRACK_MAX_MISS:
                tracked_tennis = None
        return tracked_tennis

    if tracked_tennis is None:
        best = None
        best_score = None
        for cand in candidates:
            score = (cand["w"] * cand["h"]) + int(cand["score"] * 100)
            if (best_score is None) or (score > best_score):
                best_score = score
                best = cand
        tracked_tennis = best.copy()
        tracked_tennis["miss"] = 0
        return tracked_tennis

    best = None
    best_cost = None
    gate = max(TRACK_GATE_MIN, tracked_tennis["radius"] * 3)
    gate2 = gate * gate

    for cand in candidates:
        dx = cand["cx"] - tracked_tennis["cx"]
        dy = cand["cy"] - tracked_tennis["cy"]
        d2 = (dx * dx) + (dy * dy)
        if d2 > gate2:
            continue
        cost = d2 - (cand["w"] * cand["h"]) - int(cand["score"] * 50)
        if (best_cost is None) or (cost < best_cost):
            best_cost = cost
            best = cand

    if best is None:
        tracked_tennis["miss"] += 1
        if tracked_tennis["miss"] > TRACK_MAX_MISS:
            tracked_tennis = None
        return tracked_tennis

    tracked_tennis["cx"] = smooth_value(tracked_tennis["cx"], best["cx"])
    tracked_tennis["cy"] = smooth_value(tracked_tennis["cy"], best["cy"])
    tracked_tennis["radius"] = smooth_value(tracked_tennis["radius"], best["radius"])
    tracked_tennis["score"] = best["score"]
    tracked_tennis["w"] = best["w"]
    tracked_tennis["h"] = best["h"]
    tracked_tennis["dist_cm"] = best["dist_cm"]
    tracked_tennis["miss"] = 0
    return tracked_tennis


def draw_tracked_tennis(img, target):
    if target is None:
        img.draw_string(2, 38, "TRACK SEARCH", color=YELLOW, mono_space=False)
        return

    cx = clamp(target["cx"], 0, img.width() - 1)
    cy = clamp(target["cy"], 0, img.height() - 1)
    radius = clamp(target["radius"], 8, 50)

    img.draw_circle((cx, cy, radius + 3), color=YELLOW)
    img.draw_cross(cx, cy, color=WHITE, size=10, thickness=2)
    img.draw_line((img.width() // 2, img.height() // 2, cx, cy), color=YELLOW)

    status = "TRACK LOCK"
    img.draw_string(2, 38, status, color=YELLOW, mono_space=False)
    img.draw_string(2, 56, "dx:%d dy:%d" % (cx - (img.width() // 2), cy - (img.height() // 2)), color=WHITE, mono_space=False)
    if target["dist_cm"] is not None:
        img.draw_string(2, 74, "dist:%.1fcm" % target["dist_cm"], color=WHITE, mono_space=False)


def update_servo_tracking(target, img):
    global pan_angle, tilt_angle

    if not ENABLE_SERVOS:
        return
    if target is None:
        return
    if p1 is None or p9 is None:
        return
    if target.get("miss", 0) > 0:
        return

    pan_error = target["cx"] - (img.width() / 2)
    tilt_error = target["cy"] - (img.height() / 2)

    if -SERVO_DEADBAND < pan_error < SERVO_DEADBAND:
        pan_error = 0
    if -SERVO_DEADBAND < tilt_error < SERVO_DEADBAND:
        tilt_error = 0

    pan_output = pan_pid.get_pid(pan_error, 1) / 2
    tilt_output = tilt_pid.get_pid(tilt_error, 1)

    pan_angle -= pan_output
    tilt_angle += tilt_output

    pan_angle = clamp(pan_angle, pan_angle_limit[0], pan_angle_limit[1])
    tilt_angle = clamp(tilt_angle, tilt_angle_limit[0], tilt_angle_limit[1])


def estimate_distance_cm(pixel_diameter):
    # 简化版距离估计，只用检测框尺度，避免额外的重图像处理。
    focal_length_mm = 2.8
    tennis_diameter_mm = 67
    sensor_width_mm = 4.896
    image_width = 240

    if pixel_diameter <= 0:
        return None

    mm_per_pixel = sensor_width_mm / image_width
    projected_mm = pixel_diameter * mm_per_pixel
    if projected_mm <= 0:
        return None

    distance_mm = (focal_length_mm * tennis_diameter_mm) / projected_mm
    return distance_mm / 10.0


def make_grayscale_image(channel, oh, ow):
    # 4.8.1 下优先走最稳的 bytearray 路线。
    buf = bytearray(oh * ow)
    for y in range(oh):
        row_offset = y * ow
        for x in range(ow):
            val = int(channel[y, x] * 255 + 0.5)
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            buf[row_offset + x] = val
    return image.Image(ow, oh, image.GRAYSCALE, buffer=buf, copy_to_fb=False)


def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh
    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (oh * scale)) / 2) + inputs[0].roi[1]

    threshold_list = [(math.ceil(0.20 * 255), 255)]
    results = [[] for _ in range(oc)]

    for i in range(oc):
        channel = outputs[0][0, :, :, i]
        heatmap = make_grayscale_image(channel, oh, ow)
        blobs = heatmap.find_blobs(
            threshold_list,
            x_stride=1,
            y_stride=1,
            area_threshold=1,
            pixels_threshold=1,
        )
        for blob in blobs:
            rect = blob.rect()
            x, y, w, h = rect
            score = heatmap.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            results[i].append((x, y, w, h, score))

    return results


def boot():
    global labels, net

    try:
        init_camera()
    except Exception as err:
        sys.print_exception(err)
        raise

    try:
        init_lcd()
    except Exception as err:
        sys.print_exception(err)
        raise

    show_message("Camera OK", "Checking files...")

    try:
        model_size = uos.stat(MODEL_PATH)[6]
        labels = [line.rstrip("\n") for line in open(LABELS_PATH)]
    except Exception as err:
        halt_with_error("FILE FAIL", err)

    try:
        show_message("Loading model...", MODEL_PATH)
        net = ml.Model(
            MODEL_PATH,
            load_to_fb=model_size > (gc.mem_free() - (64 * 1024)),
        )
    except Exception as err:
        halt_with_error("LOAD FAIL", err)

    print("Model input shape:", net.input_shape)
    print("Model output shape:", net.output_shape)
    print("Labels:", labels)
    if ENABLE_SERVOS:
        try:
            show_message("Model OK", "Init servos...")
            init_servos()
        except Exception as err:
            halt_with_error("SERVO FAIL", err)
    show_message("Model OK", "Running...")


def main_loop():
    clock = time.clock()

    while True:
        clock.tick()
        try:
            img = sensor.snapshot()
            predictions = net.predict([img], callback=fomo_post_process)
            tennis_candidates = []

            for i, detection_list in enumerate(predictions):
                if i == 0:
                    continue

                conf_th = threshold_for_class(i)
                detection_list = [d for d in detection_list if d[4] >= conf_th]
                if not detection_list:
                    continue

                label_name = labels[i] if i < len(labels) else ("class_%d" % i)
                label_l = label_name.lower()

                for x, y, w, h, score in detection_list:
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)

                    if ("tennis" in label_l) and ("player" not in label_l) and ("racket" not in label_l):
                        radius = max(8, min(40, int(max(w, h) * 0.7)))
                        img.draw_circle((center_x, center_y, radius), color=GREEN)

                        dist_cm = estimate_distance_cm(max(w, h))
                        img.draw_string(center_x + 4, max(0, center_y - 12), "T", color=YELLOW, mono_space=False)
                        if dist_cm is not None:
                            img.draw_string(center_x + 4, min(img.height() - 12, center_y + 4), "%.1fcm" % dist_cm, color=WHITE, mono_space=False)
                        tennis_candidates.append(
                            {
                                "cx": center_x,
                                "cy": center_y,
                                "radius": radius,
                                "w": w,
                                "h": h,
                                "score": score,
                                "dist_cm": dist_cm,
                            }
                        )

                    elif "player" in label_l:
                        img.draw_circle((center_x, center_y, 20), color=BLUE)
                        img.draw_string(center_x + 4, max(0, center_y - 12), "P", color=YELLOW, mono_space=False)

                    elif "racket" in label_l:
                        img.draw_rectangle((x, y, w, h), color=RED)
                        img.draw_string(x, max(0, y - 12), "R", color=YELLOW, mono_space=False)

                    else:
                        radius = max(8, min(24, max(w, h) // 2))
                        color = colors[i % len(colors)]
                        img.draw_circle((center_x, center_y, radius), color=color)

            target = choose_tennis_target(tennis_candidates)
            update_servo_tracking(target, img)
            draw_tracked_tennis(img, target)
            img.draw_string(2, 2, "FOMO OK", color=YELLOW, mono_space=False)
            img.draw_string(2, 20, "%.2f fps" % clock.fps(), color=WHITE, mono_space=False)
            if ENABLE_SERVOS:
                img.draw_string(2, 92, "pan:%d tilt:%d" % (int(pan_angle), int(tilt_angle)), color=WHITE, mono_space=False)
            display_frame(img)
        except Exception as err:
            img = sensor.snapshot()
            img.draw_string(2, 2, "RUN FAIL", color=RED, mono_space=False)
            img.draw_string(2, 20, str(err), color=YELLOW, mono_space=False)
            display_frame(img)
            sys.print_exception(err)
            time.sleep_ms(300)


try:
    boot()
    main_loop()
except Exception as err:
    sys.print_exception(err)
    while True:
        time.sleep_ms(1000)
