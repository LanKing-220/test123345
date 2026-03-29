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
tracked_player = None
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

GRID_ROWS = 9
GRID_COLS = 12
GRID_COLOR = (96, 96, 96)

MODE_PICK = 0
MODE_PLAY = 1
PICK_SCAN = 0
PICK_TRACK = 1
PLAY_TRACK_PLAYER = 0
PLAY_WAIT_SERVE = 1

AUTO_SWITCH_TO_PLAY = True
PLAY_ENTER_CONFIRM_FRAMES = 5
PICK_LOCK_CONFIRM_FRAMES = 3
CAPTURE_DISTANCE_CM = 10.0
CAPTURE_HOLD_FRAMES = 8

mode = MODE_PICK
pick_substate = PICK_SCAN
play_substate = PLAY_TRACK_PLAYER

capture_cmd = 0
capture_flash_frames = 0
player_locked = False
balls_served = 0
target_balls = 5
scan_direction = 1
scan_speed = 2.0
scan_lock_count = 0
play_ready_frames = 0
racket_seen_prev = False
best_scan_tennis = None


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


def draw_dashed_line(img, x0, y0, x1, y1, color, dash_len=8, gap_len=6):
    if x0 == x1:
        y = y0
        while y < y1:
            y_end = min(y + dash_len, y1)
            img.draw_line((x0, y, x1, y_end), color=color)
            y = y_end + gap_len
    elif y0 == y1:
        x = x0
        while x < x1:
            x_end = min(x + dash_len, x1)
            img.draw_line((x, y0, x_end, y1), color=color)
            x = x_end + gap_len


def draw_grid(img, rows, cols, color):
    w = img.width()
    h = img.height()
    for i in range(1, cols):
        x = (w * i) // cols
        draw_dashed_line(img, x, 0, x, h, color)
    for j in range(1, rows):
        y = (h * j) // rows
        draw_dashed_line(img, 0, y, w, y, color)


def get_grid_position(x, y, img_w, img_h, rows, cols):
    col = min(cols - 1, max(0, (x * cols) // img_w))
    row = min(rows - 1, max(0, (y * rows) // img_h))
    return row, col


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
    tracked_tennis["row"] = best["row"]
    tracked_tennis["col"] = best["col"]
    tracked_tennis["kind"] = best["kind"]
    tracked_tennis["miss"] = 0
    return tracked_tennis


def choose_player_target(candidates):
    global tracked_player

    if not candidates:
        if tracked_player is not None:
            tracked_player["miss"] += 1
            if tracked_player["miss"] > TRACK_MAX_MISS:
                tracked_player = None
        return tracked_player

    if tracked_player is None:
        best = None
        best_score = None
        for cand in candidates:
            score = (cand["w"] * cand["h"]) + int(cand["score"] * 100)
            if (best_score is None) or (score > best_score):
                best_score = score
                best = cand
        tracked_player = best.copy()
        tracked_player["miss"] = 0
        return tracked_player

    best = None
    best_cost = None
    gate = max(TRACK_GATE_MIN + 20, max(tracked_player["w"], tracked_player["h"]) * 2)
    gate2 = gate * gate

    for cand in candidates:
        dx = cand["cx"] - tracked_player["cx"]
        dy = cand["cy"] - tracked_player["cy"]
        d2 = (dx * dx) + (dy * dy)
        if d2 > gate2:
            continue
        cost = d2 - (cand["w"] * cand["h"]) - int(cand["score"] * 40)
        if (best_cost is None) or (cost < best_cost):
            best_cost = cost
            best = cand

    if best is None:
        tracked_player["miss"] += 1
        if tracked_player["miss"] > TRACK_MAX_MISS:
            tracked_player = None
        return tracked_player

    tracked_player["cx"] = smooth_value(tracked_player["cx"], best["cx"])
    tracked_player["cy"] = smooth_value(tracked_player["cy"], best["cy"])
    tracked_player["w"] = best["w"]
    tracked_player["h"] = best["h"]
    tracked_player["score"] = best["score"]
    tracked_player["row"] = best["row"]
    tracked_player["col"] = best["col"]
    tracked_player["kind"] = best["kind"]
    tracked_player["miss"] = 0
    return tracked_player


def draw_active_target(img, target):
    if target is None or target.get("miss", 0) > 0:
        img.draw_string(2, 74, "target:search", color=YELLOW, mono_space=False)
        return

    cx = clamp(target["cx"], 0, img.width() - 1)
    cy = clamp(target["cy"], 0, img.height() - 1)
    radius = clamp(target["radius"], 8, 50)

    img.draw_circle((cx, cy, radius + 3), color=YELLOW)
    img.draw_cross(cx, cy, color=WHITE, size=10, thickness=2)
    img.draw_line((img.width() // 2, img.height() // 2, cx, cy), color=YELLOW)

    kind = target.get("kind", "target").upper()
    img.draw_string(2, 74, "%s LOCK" % kind[:6], color=YELLOW, mono_space=False)
    img.draw_string(
        2,
        92,
        "dx:%d dy:%d" % (cx - (img.width() // 2), cy - (img.height() // 2)),
        color=WHITE,
        mono_space=False,
    )
    if target["dist_cm"] is not None:
        img.draw_string(2, 110, "dist:%.1fcm" % target["dist_cm"], color=WHITE, mono_space=False)
    else:
        img.draw_string(
            2,
            110,
            "grid:(%d,%d)" % (target["row"], target["col"]),
            color=WHITE,
            mono_space=False,
        )


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


def update_tilt_tracking(target, img):
    global tilt_angle

    if not ENABLE_SERVOS:
        return
    if target is None:
        return
    if p9 is None:
        return

    tilt_error = target["cy"] - (img.height() / 2)
    if -SERVO_DEADBAND < tilt_error < SERVO_DEADBAND:
        tilt_error = 0

    tilt_output = tilt_pid.get_pid(tilt_error, 1)
    tilt_angle += tilt_output
    tilt_angle = clamp(tilt_angle, tilt_angle_limit[0], tilt_angle_limit[1])


def choose_nearest_tennis(candidates):
    best = None
    best_key = None
    for cand in candidates:
        distance = cand["dist_cm"] if cand["dist_cm"] is not None else 9999.0
        key = (distance, -(cand["w"] * cand["h"]), -cand["score"])
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand
    return best


def update_scan_motion(img, nearest_tennis):
    global pan_angle, scan_direction

    if not ENABLE_SERVOS:
        return
    if p1 is None:
        return

    pan_angle += scan_direction * scan_speed
    if pan_angle >= pan_angle_limit[1]:
        pan_angle = pan_angle_limit[1]
        scan_direction = -1
    elif pan_angle <= pan_angle_limit[0]:
        pan_angle = pan_angle_limit[0]
        scan_direction = 1

    update_tilt_tracking(nearest_tennis, img)


def enter_pick_mode():
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, scan_lock_count, play_ready_frames
    global racket_seen_prev, best_scan_tennis, tracked_tennis, tracked_player

    mode = MODE_PICK
    pick_substate = PICK_SCAN
    play_substate = PLAY_TRACK_PLAYER
    capture_cmd = 0
    capture_flash_frames = 0
    player_locked = False
    balls_served = 0
    scan_lock_count = 0
    play_ready_frames = 0
    racket_seen_prev = False
    best_scan_tennis = None
    tracked_tennis = None
    tracked_player = None


def enter_play_mode():
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, scan_lock_count, play_ready_frames
    global racket_seen_prev, best_scan_tennis, tracked_tennis, tracked_player

    mode = MODE_PLAY
    pick_substate = PICK_SCAN
    play_substate = PLAY_TRACK_PLAYER
    capture_cmd = 0
    capture_flash_frames = 0
    player_locked = False
    balls_served = 0
    scan_lock_count = 0
    play_ready_frames = 0
    racket_seen_prev = False
    best_scan_tennis = None
    tracked_tennis = None
    tracked_player = None


def run_state_machine(img, tennis_target, tennis_candidates, player_target, racket_candidates):
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, play_ready_frames, racket_seen_prev
    global scan_lock_count, best_scan_tennis, tracked_tennis

    racket_present = len(racket_candidates) > 0
    nearest_tennis = choose_nearest_tennis(tennis_candidates)
    active_target = tennis_target

    if capture_flash_frames > 0:
        capture_cmd = 1
        capture_flash_frames -= 1
    else:
        capture_cmd = 0

    if mode == MODE_PICK:
        if (
            AUTO_SWITCH_TO_PLAY
            and player_target is not None
            and player_target.get("miss", 0) == 0
            and racket_present
        ):
            play_ready_frames += 1
            if play_ready_frames >= PLAY_ENTER_CONFIRM_FRAMES:
                enter_play_mode()
        else:
            play_ready_frames = 0

    if mode == MODE_PICK:
        if pick_substate == PICK_SCAN:
            update_scan_motion(img, nearest_tennis)
            active_target = tennis_target
            if tennis_target is not None and tennis_target.get("miss", 0) == 0:
                scan_lock_count += 1
                best_scan_tennis = tennis_target.copy()
                if scan_lock_count >= PICK_LOCK_CONFIRM_FRAMES:
                    pick_substate = PICK_TRACK
            else:
                scan_lock_count = 0
                best_scan_tennis = None
        else:
            if tennis_target is None or tennis_target.get("miss", 0) > 0:
                pick_substate = PICK_SCAN
                scan_lock_count = 0
                best_scan_tennis = None
                active_target = None
            else:
                active_target = tennis_target
                update_servo_tracking(tennis_target, img)
                if (
                    tennis_target["dist_cm"] is not None
                    and tennis_target["dist_cm"] <= CAPTURE_DISTANCE_CM
                ):
                    capture_flash_frames = CAPTURE_HOLD_FRAMES
                    capture_cmd = 1
                    tracked_tennis = None
                    pick_substate = PICK_SCAN
                    scan_lock_count = 0
                    best_scan_tennis = None

        return active_target, "PICK", "SCAN" if pick_substate == PICK_SCAN else "TRACK", racket_present

    active_target = player_target
    if player_target is not None and player_target.get("miss", 0) == 0:
        player_locked = True
        update_servo_tracking(player_target, img)
    else:
        player_locked = False

    if play_substate == PLAY_TRACK_PLAYER:
        if player_locked and racket_present:
            play_substate = PLAY_WAIT_SERVE
    else:
        if racket_present and (not racket_seen_prev):
            balls_served += 1
        if not player_locked:
            play_substate = PLAY_TRACK_PLAYER
        elif balls_served >= target_balls:
            enter_pick_mode()
            return tennis_target, "PICK", "SCAN", racket_present

    racket_seen_prev = racket_present
    return active_target, "PLAY", "TRACK_P" if play_substate == PLAY_TRACK_PLAYER else "WAIT_R", racket_present


def draw_status_panel(img, fps, mode_name, state_name, racket_present):
    img.draw_string(2, 2, "FOMO OK", color=YELLOW, mono_space=False)
    img.draw_string(2, 20, "%.2f fps" % fps, color=WHITE, mono_space=False)
    img.draw_string(2, 38, "mode:%s %s" % (mode_name, state_name), color=YELLOW, mono_space=False)
    img.draw_string(
        2,
        56,
        "cap:%d serve:%d/%d" % (capture_cmd, balls_served, target_balls),
        color=WHITE,
        mono_space=False,
    )
    if capture_cmd:
        img.draw_string(140, 2, "CAPTURE", color=RED, mono_space=False)
    if ENABLE_SERVOS:
        img.draw_string(
            2,
            128,
            "pan:%d tilt:%d" % (int(pan_angle), int(tilt_angle)),
            color=WHITE,
            mono_space=False,
        )
    if mode_name == "PLAY":
        img.draw_string(
            2,
            146,
            "player:%d racket:%d" % (1 if player_locked else 0, 1 if racket_present else 0),
            color=WHITE,
            mono_space=False,
        )
    elif best_scan_tennis is not None and best_scan_tennis.get("dist_cm") is not None:
        img.draw_string(
            2,
            146,
            "scan:%.1fcm" % best_scan_tennis["dist_cm"],
            color=WHITE,
            mono_space=False,
        )


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
            draw_grid(img, GRID_ROWS, GRID_COLS, GRID_COLOR)
            predictions = net.predict([img], callback=fomo_post_process)
            tennis_candidates = []
            player_candidates = []
            racket_candidates = []

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
                    row, col = get_grid_position(
                        center_x, center_y, img.width(), img.height(), GRID_ROWS, GRID_COLS
                    )

                    if ("tennis" in label_l) and ("player" not in label_l) and ("racket" not in label_l):
                        radius = max(8, min(40, int(max(w, h) * 0.7)))
                        img.draw_circle((center_x, center_y, radius), color=GREEN)

                        dist_cm = estimate_distance_cm(max(w, h))
                        img.draw_string(
                            center_x + 4,
                            max(0, center_y - 12),
                            "T[%d,%d]" % (row, col),
                            color=YELLOW,
                            mono_space=False,
                        )
                        if dist_cm is not None:
                            img.draw_string(
                                center_x + 4,
                                min(img.height() - 12, center_y + 4),
                                "%.1fcm" % dist_cm,
                                color=WHITE,
                                mono_space=False,
                            )
                        tennis_candidates.append(
                            {
                                "cx": center_x,
                                "cy": center_y,
                                "radius": radius,
                                "w": w,
                                "h": h,
                                "score": score,
                                "dist_cm": dist_cm,
                                "row": row,
                                "col": col,
                                "kind": "tennis",
                            }
                        )

                    elif "player" in label_l:
                        img.draw_circle((center_x, center_y, 20), color=BLUE)
                        img.draw_string(
                            center_x + 4,
                            max(0, center_y - 12),
                            "P[%d,%d]" % (row, col),
                            color=YELLOW,
                            mono_space=False,
                        )
                        player_candidates.append(
                            {
                                "cx": center_x,
                                "cy": center_y,
                                "radius": 20,
                                "w": w,
                                "h": h,
                                "score": score,
                                "dist_cm": None,
                                "row": row,
                                "col": col,
                                "kind": "player",
                            }
                        )

                    elif "racket" in label_l:
                        img.draw_rectangle((x, y, w, h), color=RED)
                        img.draw_string(
                            x,
                            max(0, y - 12),
                            "R[%d,%d]" % (row, col),
                            color=YELLOW,
                            mono_space=False,
                        )
                        racket_candidates.append(
                            {
                                "cx": center_x,
                                "cy": center_y,
                                "radius": max(8, min(24, max(w, h) // 2)),
                                "w": w,
                                "h": h,
                                "score": score,
                                "dist_cm": None,
                                "row": row,
                                "col": col,
                                "kind": "racket",
                            }
                        )

                    else:
                        radius = max(8, min(24, max(w, h) // 2))
                        color = colors[i % len(colors)]
                        img.draw_circle((center_x, center_y, radius), color=color)

            tennis_target = choose_tennis_target(tennis_candidates)
            player_target = choose_player_target(player_candidates)
            active_target, mode_name, state_name, racket_present = run_state_machine(
                img, tennis_target, tennis_candidates, player_target, racket_candidates
            )
            draw_active_target(img, active_target)
            draw_status_panel(img, clock.fps(), mode_name, state_name, racket_present)
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
