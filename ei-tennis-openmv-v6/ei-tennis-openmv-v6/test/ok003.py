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
frame_index = 0

TRACK_MAX_MISS = 8
TRACK_GATE_MIN = 28
TRACK_SMOOTH_OLD_NUM = 7
TRACK_SMOOTH_NEW_NUM = 3

pan_angle = 90.0
tilt_angle = 132.0
pan_angle_limit = [30.0, 150.0]
tilt_angle_limit = [80.0, 150.0]
SERVO_DEADBAND = 6
SCAN_TILT_TARGET = 132.0
SCAN_TILT_FLOOR = 128.0
SCAN_TILT_STEP = 1.5

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
NEAREST_SWITCH_MARGIN_CM = 6.0
NEAREST_SWITCH_CONFIRM_FRAMES = 2

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
nearest_switch_count = 0
play_ready_frames = 0
racket_seen_prev = False
best_scan_tennis = None

ENABLE_TENNIS_REFINEMENT = True
HOUGH_INTERVAL = 3
DISTANCE_SCALE = 2.15
REFINE_ALL_TENNIS_IN_PICK_SCAN = True

COLOR_TOL_L_BASE = 12
COLOR_TOL_A_BASE = 10
COLOR_TOL_B_BASE = 10
COLOR_TOL_L_EXTRA = 3
COLOR_TOL_A_EXTRA = 3
COLOR_TOL_B_EXTRA = 3.5

DRAW_RADIUS_SCALE = 1.15
ROI_NEAR_SWITCH = 26
FAR_ROI_PAD_MIN = 6
MAX_COLOR_BLOB_AREA_MULT = 6

EDGE_LOW_TH = 55
EDGE_HIGH_TH = 110

GLARE_L_TH = 88
GLARE_RATIO_TH_PCT = 18
SMALL_BALL_SIZE_TH = 24
GLARE_DIAMETER_CAP_PCT = 115


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


def correct_tennis_distance(distance_cm, radius):
    if radius >= 50:
        factor = 1.10
    elif radius >= 42:
        factor = 1.15
    elif radius >= 36:
        factor = 1.20
    else:
        factor = 1.25
    return distance_cm * factor


def build_tennis_color_threshold(img, x, y, w, h):
    seed_w = max(6, (w * 2) // 5)
    seed_h = max(6, (h * 2) // 5)
    seed_x = clamp(x + (w - seed_w) // 2, 0, img.width() - 1)
    seed_y = clamp(y + (h - seed_h) // 2, 0, img.height() - 1)
    seed_w = clamp(seed_w, 1, img.width() - seed_x)
    seed_h = clamp(seed_h, 1, img.height() - seed_y)

    s = img.get_statistics(roi=(seed_x, seed_y, seed_w, seed_h))
    l_mean = s.l_mean()
    a_mean = s.a_mean()
    b_mean = s.b_mean()
    l_std = s.l_stdev()
    a_std = s.a_stdev()
    b_std = s.b_stdev()

    tol_l = int(clamp(COLOR_TOL_L_BASE + COLOR_TOL_L_EXTRA + l_std, 8, 24))
    tol_a = int(clamp(COLOR_TOL_A_BASE + COLOR_TOL_A_EXTRA + a_std, 6, 18))
    tol_b = int(clamp(COLOR_TOL_B_BASE + COLOR_TOL_B_EXTRA + b_std, 6, 18))

    l_lo = int(clamp(l_mean - tol_l, 0, 100))
    l_hi = int(clamp(l_mean + tol_l, 0, 100))
    a_lo = int(clamp(a_mean - tol_a, -128, 127))
    a_hi = int(clamp(a_mean + tol_a, -128, 127))
    b_lo = int(clamp(b_mean - tol_b, -128, 127))
    b_hi = int(clamp(b_mean + tol_b, -128, 127))
    return (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi)


def estimate_color_blob(img, roi, ref_cx, ref_cy, bbox_d):
    rx, ry, rw, rh = roi
    thr = build_tennis_color_threshold(
        img, ref_cx - (rw // 6), ref_cy - (rh // 6), max(6, rw // 3), max(6, rh // 3)
    )
    blobs = img.find_blobs(
        [thr],
        roi=roi,
        x_stride=1,
        y_stride=1,
        area_threshold=20,
        pixels_threshold=20,
        merge=True,
        margin=3,
    )

    if not blobs:
        return None, ref_cx, ref_cy

    best = None
    best_score = None
    max_blob_area = max(36, bbox_d * bbox_d * MAX_COLOR_BLOB_AREA_MULT)
    for b in blobs:
        if b.pixels() > max_blob_area:
            continue

        dx = b.cx() - ref_cx
        dy = b.cy() - ref_cy
        inside_ref = (b.x() <= ref_cx <= (b.x() + b.w())) and (b.y() <= ref_cy <= (b.y() + b.h()))

        long_side = max(b.w(), b.h())
        short_side = max(1, min(b.w(), b.h()))
        ratio = (long_side * 100) // short_side
        shape_penalty = abs(ratio - 100)

        dist_cost = abs(dx) + abs(dy)
        size_gain = b.pixels() // 6
        center_bonus = 60 if inside_ref else 0
        score = (dist_cost * 3) + shape_penalty - size_gain - center_bonus
        if (best_score is None) or (score < best_score):
            best_score = score
            best = b

    if best is None:
        return None, ref_cx, ref_cy

    eq_d = int(math.sqrt((4.0 * best.pixels()) / math.pi))
    blob_d = max(best.w(), best.h())
    color_d = max(eq_d, blob_d)
    return color_d, best.cx(), best.cy()


def estimate_edge_strength(img, roi):
    edge_img = img.copy(roi=roi)
    edge_img.to_grayscale()
    edge_img.find_edges(image.EDGE_CANNY, threshold=(EDGE_LOW_TH, EDGE_HIGH_TH))
    return edge_img.get_statistics().l_mean()


def estimate_glare_ratio_pct(img, roi):
    bright_blobs = img.find_blobs(
        [(GLARE_L_TH, 100, -128, 127, -128, 127)],
        roi=roi,
        x_stride=1,
        y_stride=1,
        area_threshold=1,
        pixels_threshold=1,
        merge=True,
        margin=1,
    )
    if not bright_blobs:
        return 0

    bright_pixels = 0
    for b in bright_blobs:
        bright_pixels += b.pixels()

    roi_pixels = max(1, roi[2] * roi[3])
    return (bright_pixels * 100) // roi_pixels


def estimate_hough_circle(img, roi, ref_cx, ref_cy, r_guess, edge_strength, glare_ratio_pct):
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        hough_threshold = 3300 if edge_strength >= 24 else 2900
    else:
        hough_threshold = 3000 if edge_strength >= 24 else 2650

    r_guess = max(3, r_guess)
    r_min = max(3, (r_guess * 7) // 10)
    r_max = min(105, (r_guess * 13) // 10)
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        r_max = max(r_min, (r_max * 9) // 10)

    circles = img.find_circles(
        roi=roi,
        threshold=hough_threshold,
        x_margin=6,
        y_margin=6,
        r_margin=6,
        r_min=r_min,
        r_max=r_max,
        r_step=1,
    )

    if not circles:
        return None, ref_cx, ref_cy

    best = None
    best_score = None
    for c in circles:
        dx = c.x() - ref_cx
        dy = c.y() - ref_cy
        center_cost = abs(dx) + abs(dy)
        radius_cost = abs(c.r() - r_guess)
        score = (center_cost * 2) + radius_cost - (c.r() // 4)
        if (best_score is None) or (score < best_score):
            best_score = score
            best = c

    if best is None:
        return None, ref_cx, ref_cy

    return best.r() * 2, best.x(), best.y()


def estimate_tennis_diameter(img, x, y, w, h, allow_hough=True):
    ball_size = max(w, h)
    if ball_size < ROI_NEAR_SWITCH:
        pad = max(FAR_ROI_PAD_MIN, (ball_size * 2) // 3)
    else:
        pad = max(12, ball_size)
    rx = clamp(x - pad, 0, img.width() - 1)
    ry = clamp(y - pad, 0, img.height() - 1)
    rw = clamp(w + (pad * 2), 1, img.width() - rx)
    rh = clamp(h + (pad * 2), 1, img.height() - ry)
    roi = (rx, ry, rw, rh)

    cx = x + (w // 2)
    cy = y + (h // 2)
    bbox_d = max(w, h)

    color_d, color_cx, color_cy = estimate_color_blob(img, roi, cx, cy, bbox_d)

    hough_d = None
    if allow_hough:
        edge_strength = estimate_edge_strength(img, roi)
        glare_ratio_pct = estimate_glare_ratio_pct(img, roi)
        if color_d is not None:
            hough_guess = max(color_d // 2, bbox_d // 2)
        else:
            hough_guess = max(5, (bbox_d * 8) // 10)
        hough_d, _, _ = estimate_hough_circle(
            img, roi, color_cx, color_cy, hough_guess, edge_strength, glare_ratio_pct
        )
    else:
        glare_ratio_pct = 0

    cue_conf = 0
    if (hough_d is not None) and (color_d is not None):
        if abs(hough_d - color_d) <= 10:
            d = ((hough_d * 5) + (color_d * 5)) // 10
            cue_conf = 2
        else:
            d = max(hough_d, color_d)
            cue_conf = 1
    elif hough_d is not None:
        d = hough_d
        cue_conf = 1
    elif color_d is not None:
        d = color_d
        cue_conf = 1
    else:
        d = int((bbox_d * 13) // 10)
        cue_conf = 0

    d = clamp(d, 8, 210)

    if (ball_size <= SMALL_BALL_SIZE_TH) and (glare_ratio_pct >= GLARE_RATIO_TH_PCT):
        glare_cap = max(8, (bbox_d * GLARE_DIAMETER_CAP_PCT) // 100)
        if d > glare_cap:
            d = glare_cap
        cue_conf = min(cue_conf, 1)

    return d, cue_conf


def estimate_ball_radius(w, h):
    mx = max(w, h)
    r = ((mx * 13) + 10) // 20
    area = w * h
    if area >= 900:
        r += 3
    elif area >= 400:
        r += 2
    return clamp(r, 4, 55)


def fuse_tennis_radius(w, h, detected_diameter):
    circle_r = detected_diameter // 2
    bbox_r = estimate_ball_radius(w, h)
    if max(w, h) >= 20:
        r = max(circle_r, bbox_r)
    else:
        r = max(circle_r, (bbox_r * 9) // 10)
    r = int((r * DRAW_RADIUS_SCALE) + 0.5)
    return clamp(r, 4, 105)


def smooth_ball_radius(curr_r, prev_r):
    if prev_r is None:
        return curr_r

    diff = curr_r - prev_r
    if -2 <= diff <= 2:
        return prev_r

    up_step = 8 if prev_r < 24 else 12
    down_step = 8 if prev_r >= 20 else 6
    if diff > up_step:
        curr_r = prev_r + up_step
    elif diff < -down_step:
        curr_r = prev_r - down_step

    return ((prev_r * 13) + (curr_r * 7)) // 20


def choose_tennis_target(candidates):
    global tracked_tennis

    if not candidates:
        if tracked_tennis is not None:
            tracked_tennis["miss"] += 1
            if tracked_tennis["miss"] > TRACK_MAX_MISS:
                tracked_tennis = None
        return tracked_tennis

    if tracked_tennis is None:
        best = choose_nearest_tennis(candidates)
        if best is None:
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
    tracked_tennis["x"] = best["x"]
    tracked_tennis["y"] = best["y"]
    tracked_tennis["dist_cm"] = best["dist_cm"]
    tracked_tennis["row"] = best["row"]
    tracked_tennis["col"] = best["col"]
    tracked_tennis["kind"] = best["kind"]
    if "measure_src" in best:
        tracked_tennis["measure_src"] = best["measure_src"]
    if "cue_conf" in best:
        tracked_tennis["cue_conf"] = best["cue_conf"]
    if "refined_diameter" in best:
        tracked_tennis["refined_diameter"] = best["refined_diameter"]
    if "refined_radius" in best:
        tracked_tennis["refined_radius"] = best["refined_radius"]
    if "refined_dist_cm" in best:
        tracked_tennis["refined_dist_cm"] = best["refined_dist_cm"]
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
    if "measure_src" in target:
        img.draw_string(2, 128, "measure:%s" % target["measure_src"], color=WHITE, mono_space=False)


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


def move_tilt_toward(target_angle, step=SCAN_TILT_STEP):
    global tilt_angle

    if tilt_angle < (target_angle - step):
        tilt_angle += step
    elif tilt_angle > (target_angle + step):
        tilt_angle -= step
    else:
        tilt_angle = target_angle

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


def should_switch_to_nearest(current_target, nearest_tennis):
    if nearest_tennis is None:
        return False
    if current_target is None:
        return True
    if current_target.get("miss", 0) > 0:
        return True

    current_dist = current_target.get("dist_cm")
    nearest_dist = nearest_tennis.get("dist_cm")
    if nearest_dist is None:
        return False
    if current_dist is None:
        return True

    if nearest_dist + NEAREST_SWITCH_MARGIN_CM < current_dist:
        return True

    current_area = current_target["w"] * current_target["h"]
    nearest_area = nearest_tennis["w"] * nearest_tennis["h"]
    if (nearest_dist <= current_dist) and (nearest_area > current_area * 2):
        return True

    return False


def update_scan_motion(img, nearest_tennis):
    global pan_angle, scan_direction, tilt_angle

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

    if nearest_tennis is None:
        move_tilt_toward(SCAN_TILT_TARGET)
    else:
        update_tilt_tracking(nearest_tennis, img)
        if tilt_angle < SCAN_TILT_FLOOR:
            tilt_angle = SCAN_TILT_FLOOR


def enter_pick_mode():
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, scan_lock_count, nearest_switch_count, play_ready_frames
    global racket_seen_prev, best_scan_tennis, tracked_tennis, tracked_player

    mode = MODE_PICK
    pick_substate = PICK_SCAN
    play_substate = PLAY_TRACK_PLAYER
    capture_cmd = 0
    capture_flash_frames = 0
    player_locked = False
    balls_served = 0
    scan_lock_count = 0
    nearest_switch_count = 0
    play_ready_frames = 0
    racket_seen_prev = False
    best_scan_tennis = None
    tracked_tennis = None
    tracked_player = None


def enter_play_mode():
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, scan_lock_count, nearest_switch_count, play_ready_frames
    global racket_seen_prev, best_scan_tennis, tracked_tennis, tracked_player

    mode = MODE_PLAY
    pick_substate = PICK_SCAN
    play_substate = PLAY_TRACK_PLAYER
    capture_cmd = 0
    capture_flash_frames = 0
    player_locked = False
    balls_served = 0
    scan_lock_count = 0
    nearest_switch_count = 0
    play_ready_frames = 0
    racket_seen_prev = False
    best_scan_tennis = None
    tracked_tennis = None
    tracked_player = None


def run_state_machine(img, tennis_target, tennis_candidates, player_target, racket_candidates):
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, play_ready_frames, racket_seen_prev
    global scan_lock_count, nearest_switch_count, best_scan_tennis, tracked_tennis

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
            active_target = nearest_tennis if nearest_tennis is not None else tennis_target
            if active_target is not None and active_target.get("miss", 0) == 0:
                scan_lock_count += 1
                best_scan_tennis = active_target.copy()
                if scan_lock_count >= PICK_LOCK_CONFIRM_FRAMES:
                    tracked_tennis = active_target.copy()
                    tracked_tennis["miss"] = 0
                    pick_substate = PICK_TRACK
            else:
                scan_lock_count = 0
                best_scan_tennis = None
        else:
            if tennis_target is None or tennis_target.get("miss", 0) > 0:
                pick_substate = PICK_SCAN
                scan_lock_count = 0
                nearest_switch_count = 0
                best_scan_tennis = None
                active_target = None
            else:
                active_target = tennis_target
                if should_switch_to_nearest(tennis_target, nearest_tennis):
                    nearest_switch_count += 1
                    if nearest_switch_count >= NEAREST_SWITCH_CONFIRM_FRAMES:
                        tracked_tennis = nearest_tennis.copy()
                        tracked_tennis["miss"] = 0
                        active_target = tracked_tennis
                        nearest_switch_count = 0
                else:
                    nearest_switch_count = 0
                update_servo_tracking(active_target, img)
                if (
                    active_target["dist_cm"] is not None
                    and active_target["dist_cm"] <= CAPTURE_DISTANCE_CM
                ):
                    capture_flash_frames = CAPTURE_HOLD_FRAMES
                    capture_cmd = 1
                    tracked_tennis = None
                    pick_substate = PICK_SCAN
                    scan_lock_count = 0
                    nearest_switch_count = 0
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
            164,
            "pan:%d tilt:%d" % (int(pan_angle), int(tilt_angle)),
            color=WHITE,
            mono_space=False,
        )
    if mode_name == "PLAY":
        img.draw_string(
            2,
            182,
            "player:%d racket:%d" % (1 if player_locked else 0, 1 if racket_present else 0),
            color=WHITE,
            mono_space=False,
        )
    elif best_scan_tennis is not None and best_scan_tennis.get("dist_cm") is not None:
        img.draw_string(
            2,
            182,
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


def estimate_corrected_distance_cm(pixel_diameter, radius):
    base_distance_cm = estimate_distance_cm(pixel_diameter)
    if base_distance_cm is None:
        return None
    corrected = base_distance_cm * DISTANCE_SCALE
    return correct_tennis_distance(corrected, radius)


def refine_tennis_target(img, target, allow_hough=None):
    global frame_index

    if not ENABLE_TENNIS_REFINEMENT:
        return target
    if target is None:
        return None
    if target.get("miss", 0) > 0:
        return target

    if allow_hough is None:
        allow_hough = (frame_index % HOUGH_INTERVAL) == 0
    diameter, cue_conf = estimate_tennis_diameter(
        img, target["x"], target["y"], target["w"], target["h"], allow_hough=allow_hough
    )
    refined_radius = fuse_tennis_radius(target["w"], target["h"], diameter)
    prev_radius = target.get("refined_radius")
    refined_radius = smooth_ball_radius(refined_radius, prev_radius)

    refined_dist_cm = estimate_corrected_distance_cm(diameter, refined_radius)
    prev_dist_cm = target.get("refined_dist_cm")
    if (prev_dist_cm is not None) and (refined_dist_cm is not None):
        refined_dist_cm = (prev_dist_cm * 0.7) + (refined_dist_cm * 0.3)

    target["radius"] = refined_radius
    target["dist_cm"] = refined_dist_cm
    target["refined_radius"] = refined_radius
    target["refined_dist_cm"] = refined_dist_cm
    target["refined_diameter"] = diameter
    target["cue_conf"] = cue_conf
    if allow_hough and cue_conf >= 2:
        target["measure_src"] = "mix"
    elif allow_hough and cue_conf >= 1:
        target["measure_src"] = "hough"
    elif cue_conf >= 1:
        target["measure_src"] = "lab"
    else:
        target["measure_src"] = "bbox"
    return target


def refine_tennis_candidates(img, candidates, allow_hough=True):
    if not ENABLE_TENNIS_REFINEMENT:
        return candidates

    refined = []
    for cand in candidates:
        refined.append(refine_tennis_target(img, cand, allow_hough=allow_hough))
    return refined


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
    global frame_index

    clock = time.clock()

    while True:
        clock.tick()
        try:
            frame_index += 1
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
                        dist_cm = estimate_distance_cm(max(w, h))
                        tennis_candidates.append(
                            {
                                "x": x,
                                "y": y,
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

            refine_all_tennis = (
                ENABLE_TENNIS_REFINEMENT
                and REFINE_ALL_TENNIS_IN_PICK_SCAN
                and (mode == MODE_PICK)
                and (pick_substate == PICK_SCAN)
            )
            if refine_all_tennis:
                tennis_candidates = refine_tennis_candidates(img, tennis_candidates, allow_hough=True)

            tennis_target = choose_tennis_target(tennis_candidates)
            if not refine_all_tennis:
                tennis_target = refine_tennis_target(img, tennis_target)
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
