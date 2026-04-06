import gc
import math
import sys
import time
import uos

import display
import image
import ml
import sensor
from pyb import Pin, Timer, UART
from pid import PID


MODEL_PATH = "trained.tflite"
LABELS_PATH = "labels.txt"
LCD_HINT = image.ROTATE_270

# 需要舵机追踪时打开；如果 4.8.1 下再次出现白屏，可先改回 False 做隔离。
ENABLE_SERVOS = True
PAN_SERVO_PIN = "P1"
TILT_SERVO_PIN = "P9"
ENABLE_UART = True
UART_PORT = 3
UART_BAUDRATE = 115200
UART_TIMEOUT_CHAR = 120
UART_RX_BUFFER_MAX = 96

# 通信状态：0=no link, 1=linked(收到HI), 2=ok(收到OK)
COMM_NO_LINK = 0
COMM_LINKED = 1
COMM_OK = 2
comm_state = COMM_NO_LINK
last_hello_ms = 0
HELLO_INTERVAL_MS = 5000

THRESH_TENNIS = 0.35
# 人和球拍更容易误触发，阈值调高后会更保守，降低敏感度。
THRESH_PLAYER = 0.75
THRESH_RACKET = 0.70

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
uart = None
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
pan_pwm_started = False
tilt_pwm_started = False
frame_index = 0

TRACK_MAX_MISS = 8
TRACK_GATE_MIN = 28
TRACK_SMOOTH_OLD_NUM = 7
TRACK_SMOOTH_NEW_NUM = 3

PAN_INIT_ANGLE = 90.0
TILT_INIT_ANGLE = 132.0
SERVO_INIT_HOLD_FRAMES = 28
SERVO_PWM_STAGGER_FRAMES = 8

pan_angle = PAN_INIT_ANGLE
tilt_angle = TILT_INIT_ANGLE
# 第一阶段全局找球需要更大的水平覆盖范围，但仍保留一点机械安全余量。
pan_angle_limit = [20.0, 160.0]
tilt_angle_limit = [80.0, 150.0]
SERVO_DEADBAND = 6
SCAN_TILT_TARGET = 132.0
SCAN_TILT_FLOOR = 128.0
SCAN_TILT_STEP = 1.2
SERVO_INIT_STEP = 0.35
SCAN_PAN_STEP = 0.75
TRACK_PAN_MAX_STEP = 1.1
TRACK_TILT_MAX_STEP = 1.0

pan_pid = PID(p=0.07, i=0, imax=90)
tilt_pid = PID(p=0.05, i=0, imax=90)

GRID_ROWS = 9
GRID_COLS = 12
GRID_COLOR = (96, 96, 96)

MODE_PICK = 0
MODE_PLAY = 1
PICK_SCAN = 0
PICK_RETURN = 1
PICK_CONFIRM = 2
PICK_TRACK = 3
PLAY_TRACK_PLAYER = 0
PLAY_WAIT_SERVE = 1

AUTO_SWITCH_TO_PLAY = False
PLAY_ENTER_CONFIRM_FRAMES = 5
PICK_LOCK_CONFIRM_FRAMES = 3
CAPTURE_DISTANCE_CM = 10.0
CAPTURE_HOLD_FRAMES = 8
NEAREST_SWITCH_MARGIN_CM = 6.0
NEAREST_SWITCH_CONFIRM_FRAMES = 2
SCAN_EMPTY_ROUNDS_TO_PLAY = 2
PICK_CONFIRM_DURATION_MS = 2000
PICK_TRACK_DURATION_MS = 20000
PLAYER_DETECT_COUNTDOWN_MS = 5000
COUNTDOWN_FONT_SCALE = 5
PLAY_RACKET_CONFIRM_FRAMES = 2
RACKET_LINK_MARGIN_X_RATIO = 0.60
RACKET_LINK_MARGIN_Y_RATIO = 0.35
SCAN_ALIGN_MARGIN = 1.0
RETURN_LOCK_MARGIN = 2.0
RETURN_PAN_STEP = 1.0
RETURN_TILT_STEP = 1.0

PICKER_FEEDBACK_PIN = None
PICKER_FEEDBACK_ACTIVE_LEVEL = 1

mode = MODE_PICK
pick_substate = PICK_SCAN
play_substate = PLAY_TRACK_PLAYER

capture_cmd = 0
capture_flash_frames = 0
player_locked = False
balls_served = 0
balls_picked = 0
target_balls = 5
scan_direction = 1
scan_speed = SCAN_PAN_STEP
scan_lock_count = 0
nearest_switch_count = 0
play_ready_frames = 0
racket_seen_prev = False
best_scan_tennis = None
scan_ranked_tennis = []
scan_candidate_index = 0
scan_empty_rounds = 0
player_lock_start_ms = 0
servo_init_frames_remaining = 0
scan_seek_left = True
pick_confirm_start_ms = 0
pick_track_start_ms = 0
picker_feedback_pin = None
picker_feedback_state = 0
picker_uart_done_pending = 0
next_target_id = 1
current_racket_id = 0
uart_rx_buffer = ""

ENABLE_TENNIS_REFINEMENT = True
HOUGH_INTERVAL = 3
DISTANCE_SCALE = 2.15
REFINE_ALL_TENNIS_IN_PICK_SCAN = True
MEASURE_WINDOW_LEN = 10
MEASURE_TRIM_COUNT = 2

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
    sensor.set_auto_whitebal(False)
    sensor.skip_frames(10)


def init_lcd():
    global lcd
    time.sleep_ms(300)
    lcd = display.SPIDisplay(width=240, height=320)


def init_uart():
    global uart

    if not ENABLE_UART:
        uart = None
        return

    try:
        uart = UART(UART_PORT, UART_BAUDRATE, timeout_char=UART_TIMEOUT_CHAR)
    except Exception as err:
        uart = None
        sys.print_exception(err)


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
    global pan_angle, tilt_angle, servo_init_frames_remaining
    global pan_pwm_started, tilt_pwm_started

    if not ENABLE_SERVOS:
        return

    pan_angle = PAN_INIT_ANGLE
    tilt_angle = TILT_INIT_ANGLE
    servo_init_frames_remaining = SERVO_INIT_HOLD_FRAMES
    pan_pwm_started = False
    tilt_pwm_started = False

    p1 = Pin(PAN_SERVO_PIN, Pin.OUT_PP)
    p9 = Pin(TILT_SERVO_PIN, Pin.OUT_PP)

    p1_tim_pluse = Timer(12)
    p1_tim_main = Timer(13, freq=50)

    p9_tim_pluse = Timer(14)
    p9_tim_main = Timer(15, freq=50)


def start_pan_pwm():
    global pan_pwm_started

    if pan_pwm_started or (p1_tim_main is None):
        return
    p1_tim_main.callback(P1_ISR)
    pan_pwm_started = True


def start_tilt_pwm():
    global tilt_pwm_started

    if tilt_pwm_started or (p9_tim_main is None):
        return
    p9_tim_main.callback(P9_ISR)
    tilt_pwm_started = True


def init_picker_feedback():
    global picker_feedback_pin

    if not PICKER_FEEDBACK_PIN:
        picker_feedback_pin = None
        return

    picker_feedback_pin = Pin(PICKER_FEEDBACK_PIN, Pin.IN)


def read_picker_feedback(consume=True):
    global picker_feedback_state, picker_uart_done_pending

    if picker_uart_done_pending > 0:
        picker_feedback_state = 1
        if consume:
            picker_uart_done_pending -= 1
        return 1

    if picker_feedback_pin is None:
        picker_feedback_state = 0
        return 0

    try:
        picker_feedback_state = 1 if picker_feedback_pin.value() == PICKER_FEEDBACK_ACTIVE_LEVEL else 0
    except Exception:
        picker_feedback_state = 0
    return picker_feedback_state


def uart_bytes_to_text(data):
    if data is None:
        return ""
    if isinstance(data, str):
        return data

    try:
        return data.decode()
    except Exception:
        pass

    text = ""
    for b in data:
        if b in (10, 13) or (32 <= b <= 126):
            text += chr(b)
    return text


def parse_uart_event_count(text):
    separators = (":", ",", "=", " ")
    for sep in separators:
        idx = text.find(sep)
        if idx < 0:
            continue
        tail = text[idx + 1 :].strip()
        if not tail:
            continue
        try:
            value = int(tail)
            if value > 0:
                return value
        except Exception:
            pass
    return 1


def handle_uart_line(line):
    global balls_served, balls_picked, picker_uart_done_pending
    global comm_state

    if not line:
        return

    text = line.strip()
    if not text:
        return

    upper = text.upper()
    count = parse_uart_event_count(upper)

    # 处理握手与链路状态：HI 表示链路建立，OK 表示主机确认
    if upper == "HI" or upper.startswith("HI "):
        try:
            comm_state = COMM_LINKED
        except Exception:
            pass
        return
    if upper == "OK" or upper.startswith("OK "):
        try:
            comm_state = COMM_OK
        except Exception:
            pass
        return

    if upper.startswith("SERVED") or upper.startswith("SERVE"):
        balls_served += count
        return

    if (
        upper.startswith("PICKED")
        or upper.startswith("PICK")
        or upper.startswith("PICKUP")
        or upper.startswith("COLLECT")
    ):
        balls_picked += count
        picker_uart_done_pending += count


def process_uart_rx():
    global uart_rx_buffer

    if uart is None:
        return

    try:
        waiting = uart.any()
    except Exception:
        return

    if not waiting:
        return

    try:
        data = uart.read(waiting)
    except Exception:
        return

    text = uart_bytes_to_text(data)
    if not text:
        return

    uart_rx_buffer += text
    if len(uart_rx_buffer) > UART_RX_BUFFER_MAX:
        uart_rx_buffer = uart_rx_buffer[-UART_RX_BUFFER_MAX:]

    while True:
        line_end = uart_rx_buffer.find("\n")
        if line_end < 0:
            break
        line = uart_rx_buffer[:line_end].strip()
        uart_rx_buffer = uart_rx_buffer[line_end + 1 :]
        handle_uart_line(line)


def display_frame(img):
    if lcd is not None:
        lcd.write(img, hint=LCD_HINT)


def try_send_hello():
    global last_hello_ms, comm_state
    if uart is None:
        return
    if comm_state != COMM_NO_LINK:
        return
    now = time.ticks_ms()
    if last_hello_ms <= 0:
        # 发送初始 hello
        uart_write_line("hello")
        last_hello_ms = now
        return
    if time.ticks_diff(now, last_hello_ms) >= HELLO_INTERVAL_MS:
        uart_write_line("hello")
        last_hello_ms = now


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


def append_measure_window(history, value):
    if value is None:
        return list(history) if history else []

    values = list(history) if history else []
    values.append(int(value))
    if len(values) > MEASURE_WINDOW_LEN:
        values.pop(0)
    return values


def trimmed_window_mean(values):
    if not values:
        return None

    data = list(values)
    if len(data) < MEASURE_WINDOW_LEN:
        return int((sum(data) / len(data)) + 0.5)

    data.sort()
    core = data[MEASURE_TRIM_COUNT : len(data) - MEASURE_TRIM_COUNT]
    if not core:
        core = data
    return int((sum(core) / len(core)) + 0.5)


def update_trimmed_tennis_measure(target, sample_diameter, sample_radius):
    diameter_history = append_measure_window(target.get("diameter_history"), sample_diameter)
    radius_history = append_measure_window(target.get("radius_history"), sample_radius)

    target["diameter_history"] = diameter_history
    target["radius_history"] = radius_history

    filtered_diameter = trimmed_window_mean(diameter_history)
    filtered_radius = trimmed_window_mean(radius_history)

    if filtered_diameter is None:
        filtered_diameter = int(sample_diameter) if sample_diameter is not None else None
    if filtered_radius is None:
        filtered_radius = int(sample_radius) if sample_radius is not None else None

    target["refined_diameter"] = filtered_diameter
    target["refined_radius"] = filtered_radius
    return filtered_diameter, filtered_radius


def allocate_target_id():
    global next_target_id

    target_id = next_target_id
    next_target_id += 1
    return target_id


def ensure_target_id(target, forced_id=None):
    if target is None:
        return None

    if forced_id is not None:
        target["id"] = forced_id
        return forced_id

    target_id = target.get("id")
    if target_id is None:
        target_id = allocate_target_id()
        target["id"] = target_id
    return target_id


def format_grid_id(row, col):
    if row is None or col is None:
        return "(?,?)"
    return "(%d,%d)" % (row, col)


def target_distance_cm(target):
    if target is None:
        return 0.0
    dist_cm = target.get("dist_cm")
    if dist_cm is None:
        return 0.0
    return float(dist_cm)


def is_live_target(target):
    return target is not None and target.get("miss", 0) == 0


def is_racket_linked_to_player(player_target, racket_target):
    if (not is_live_target(player_target)) or racket_target is None:
        return False

    required_keys = ("x", "y", "w", "h", "cx", "cy")
    for key in required_keys:
        if key not in player_target or key not in racket_target:
            return False

    px = player_target["x"]
    py = player_target["y"]
    pw = max(1, player_target["w"])
    ph = max(1, player_target["h"])
    rcx = racket_target["cx"]
    rcy = racket_target["cy"]
    rw = max(1, racket_target["w"])
    rh = max(1, racket_target["h"])

    margin_x = max(18, int(pw * RACKET_LINK_MARGIN_X_RATIO))
    margin_y = max(16, int(ph * RACKET_LINK_MARGIN_Y_RATIO))
    left = px - margin_x
    right = px + pw + margin_x
    top = py - margin_y
    bottom = py + ph + margin_y

    if rcx < left or rcx > right or rcy < top or rcy > bottom:
        return False

    if rh > int(ph * 12 / 10):
        return False
    if rw > int(pw * 9 / 10):
        return False

    return True


def choose_racket_target(candidates, player_target=None):
    if not candidates:
        return None

    if is_live_target(player_target):
        linked_candidates = []
        for cand in candidates:
            if is_racket_linked_to_player(player_target, cand):
                linked_candidates.append(cand)
        candidates = linked_candidates
        if not candidates:
            return None

    best = None
    best_key = None
    for cand in candidates:
        center_dx = cand["cx"] - 160
        center_dy = cand["cy"] - 120
        center_d2 = (center_dx * center_dx) + (center_dy * center_dy)
        area = cand["w"] * cand["h"]
        if is_live_target(player_target):
            player_dx = abs(cand["cx"] - player_target["cx"])
            player_dy = abs(cand["cy"] - player_target["cy"])
            player_d = player_dx + player_dy
        else:
            player_d = 0
        key = (-int(cand["score"] * 100), player_d, center_d2, -area)
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand

    return best.copy()


def uart_write_line(line):
    if uart is None:
        return

    try:
        uart.write(line)
        uart.write("\n")
    except Exception as err:
        sys.print_exception(err)


def choose_uart_ball_target(mode_name, tennis_candidates, active_target):
    if mode_name == "PLAY":
        return None

    nearest_target = choose_nearest_tennis(tennis_candidates)
    if nearest_target is not None:
        return nearest_target

    if is_live_target(active_target) and active_target.get("kind") == "tennis":
        return active_target

    return None


def send_runtime_packets(mode_name, target):
    row = -1
    col = -1
    distance_cm = 0.0

    if is_live_target(target) and target.get("kind") == "tennis":
        row = int(target.get("row", -1))
        col = int(target.get("col", -1))
        distance_cm = target_distance_cm(target)

    uart_write_line("%s,%d,%d,%.1f" % (mode_name, row, col, distance_cm))


def send_command_packet(cmd, arg, mode_name, state_name):
    """简单的命令封装发送，避免未定义时崩溃。"""
    try:
        if uart is None:
            return
        # 格式：CMD,ARG,MODE,STATE
        line = "%s,%s,%s,%s" % (str(cmd), str(arg), str(mode_name), str(state_name))
        uart_write_line(line)
    except Exception:
        pass


def age_and_prune_tracks():
    global tracked_tennis

    if tracked_tennis is not None:
        tracked_tennis["miss"] += 1
        if tracked_tennis["miss"] > TRACK_MAX_MISS:
            tracked_tennis = None
    return tracked_tennis


def match_tennis_track(candidates):
    global tracked_tennis

    if not candidates:
        return age_and_prune_tracks()

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
        if ("refined_diameter" in best) or ("refined_radius" in best):
            update_trimmed_tennis_measure(
                tracked_tennis,
                best.get("refined_diameter"),
                best.get("refined_radius", best.get("radius")),
            )
            if tracked_tennis.get("refined_radius") is not None:
                tracked_tennis["radius"] = tracked_tennis["refined_radius"]
        ensure_target_id(tracked_tennis)
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
        return age_and_prune_tracks()

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
    if ("refined_diameter" in best) or ("refined_radius" in best):
        filtered_diameter, filtered_radius = update_trimmed_tennis_measure(
            tracked_tennis,
            best.get("refined_diameter"),
            best.get("refined_radius", best.get("radius")),
        )
        if filtered_radius is not None:
            tracked_tennis["radius"] = filtered_radius
        if (filtered_diameter is not None) and (filtered_radius is not None):
            tracked_tennis["dist_cm"] = estimate_corrected_distance_cm(filtered_diameter, filtered_radius)
            tracked_tennis["refined_dist_cm"] = tracked_tennis["dist_cm"]
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

    best = None
    best_key = None
    for cand in candidates:
        center_dx = cand["cx"] - 160
        center_dy = cand["cy"] - 120
        center_d2 = (center_dx * center_dx) + (center_dy * center_dy)
        area = cand["w"] * cand["h"]
        key = (-int(cand["score"] * 1000), -area, center_d2)
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand

    previous_id = 0
    if tracked_player is not None:
        previous_id = int(tracked_player.get("id", 0))

    tracked_player = best.copy()
    if previous_id > 0:
        ensure_target_id(tracked_player, previous_id)
    else:
        ensure_target_id(tracked_player)
    tracked_player["miss"] = 0
    return tracked_player


def draw_active_target(img, target):
    if target is None or target.get("miss", 0) > 0:
        img.draw_string(2, 74, "target:search", color=YELLOW, mono_space=False)
        return

    cx = clamp(target["cx"], 0, img.width() - 1)
    cy = clamp(target["cy"], 0, img.height() - 1)
    radius = clamp(target["radius"], 8, 50)
    kind = target.get("kind", "target").lower()

    if kind == "tennis":
        target_color = GREEN
    elif kind == "player":
        target_color = BLUE
    elif kind == "racket":
        target_color = RED
    else:
        target_color = YELLOW

    if (kind == "racket") and all(k in target for k in ("x", "y", "w", "h")):
        img.draw_rectangle((target["x"], target["y"], target["w"], target["h"]), color=target_color, thickness=2)
    else:
        img.draw_circle((cx, cy, radius + 3), color=target_color)
    img.draw_cross(cx, cy, color=WHITE, size=10, thickness=2)
    img.draw_line((img.width() // 2, img.height() // 2, cx, cy), color=target_color)

    kind_label = target.get("kind", "target").upper()
    img.draw_string(2, 74, "%s LOCK" % kind_label[:6], color=target_color, mono_space=False)
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


def draw_centered_text(img, text, y, color, scale=1):
    char_w = 8 * scale
    text_w = len(text) * char_w
    x = max(0, (img.width() - text_w) // 2)

    img.draw_string(x + 2, y + 2, text, color=(0, 0, 0), scale=scale, mono_space=True)
    img.draw_string(x, y, text, color=color, scale=scale, mono_space=True)


def draw_seek_tennis_candidates(img, mode_name, state_name, tennis_candidates):
    if mode_name != "SEEK":
        return
    if state_name != "SCAN":
        return

    for cand in tennis_candidates:
        cx = clamp(cand["cx"], 0, img.width() - 1)
        cy = clamp(cand["cy"], 0, img.height() - 1)
        radius = clamp(cand["radius"], 8, 50)
        img.draw_circle((cx, cy, radius), color=GREEN, thickness=2)


def draw_player_countdown(img, mode_name, racket_present):
    if mode_name != "PLAY":
        return
    if not player_locked:
        return
    if not racket_present:
        return
    if player_lock_start_ms <= 0:
        return

    elapsed_ms = time.ticks_diff(time.ticks_ms(), player_lock_start_ms)
    remain_ms = PLAYER_DETECT_COUNTDOWN_MS - elapsed_ms
    remain_s = max(0, (remain_ms + 999) // 1000)
    text = str(remain_s)
    text_h = 10 * COUNTDOWN_FONT_SCALE
    y = max(0, (img.height() - text_h) // 2)
    draw_centered_text(img, text, y, YELLOW, scale=COUNTDOWN_FONT_SCALE)


def update_servo_tracking(target, img):
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

    apply_pan_delta(-pan_output, TRACK_PAN_MAX_STEP)
    apply_tilt_delta(tilt_output, TRACK_TILT_MAX_STEP)


def update_tilt_tracking(target, img):
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
    apply_tilt_delta(tilt_output, TRACK_TILT_MAX_STEP)


def move_tilt_toward(target_angle, step=SCAN_TILT_STEP):
    global tilt_angle

    if tilt_angle < (target_angle - step):
        tilt_angle += step
    elif tilt_angle > (target_angle + step):
        tilt_angle -= step
    else:
        tilt_angle = target_angle

    tilt_angle = clamp(tilt_angle, tilt_angle_limit[0], tilt_angle_limit[1])


def move_pan_toward(target_angle, step=SCAN_PAN_STEP):
    global pan_angle

    if pan_angle < (target_angle - step):
        pan_angle += step
    elif pan_angle > (target_angle + step):
        pan_angle -= step
    else:
        pan_angle = target_angle

    pan_angle = clamp(pan_angle, pan_angle_limit[0], pan_angle_limit[1])


def apply_pan_delta(delta, max_step=TRACK_PAN_MAX_STEP):
    global pan_angle

    delta = clamp(delta, -max_step, max_step)
    pan_angle = clamp(pan_angle + delta, pan_angle_limit[0], pan_angle_limit[1])


def apply_tilt_delta(delta, max_step=TRACK_TILT_MAX_STEP):
    global tilt_angle

    delta = clamp(delta, -max_step, max_step)
    tilt_angle = clamp(tilt_angle + delta, tilt_angle_limit[0], tilt_angle_limit[1])


def update_servo_init():
    global servo_init_frames_remaining

    if not ENABLE_SERVOS:
        return False
    if servo_init_frames_remaining <= 0:
        return False

    start_pan_pwm()
    if servo_init_frames_remaining <= (SERVO_INIT_HOLD_FRAMES - SERVO_PWM_STAGGER_FRAMES):
        start_tilt_pwm()

    move_pan_toward(PAN_INIT_ANGLE, SERVO_INIT_STEP)
    move_tilt_toward(TILT_INIT_ANGLE, SERVO_INIT_STEP)
    servo_init_frames_remaining -= 1
    return True


def scan_rank_key(target):
    distance = target.get("dist_cm")
    if distance is None:
        distance = 9999.0
    area = target["w"] * target["h"]
    return (distance, -area, -int(target.get("score", 0) * 1000))


def scan_observation_key(target):
    center_dx = abs(target["cx"] - 160)
    center_dy = abs(target["cy"] - 120)
    return ((center_dx + center_dy), -int(target.get("score", 0) * 1000), scan_rank_key(target)[0])


def is_same_scan_target(saved_target, new_target):
    if saved_target is None or new_target is None:
        return False

    row_diff = abs(saved_target["row"] - new_target["row"])
    col_diff = abs(saved_target["col"] - new_target["col"])
    pan_diff = abs(saved_target.get("servo_pan", PAN_INIT_ANGLE) - new_target.get("servo_pan", PAN_INIT_ANGLE))
    tilt_diff = abs(saved_target.get("servo_tilt", TILT_INIT_ANGLE) - new_target.get("servo_tilt", TILT_INIT_ANGLE))

    dist_a = saved_target.get("dist_cm")
    dist_b = new_target.get("dist_cm")
    if dist_a is None or dist_b is None:
        dist_diff = 9999.0
    else:
        dist_diff = abs(dist_a - dist_b)

    if (row_diff <= 1) and (col_diff <= 1) and (pan_diff <= 8.0) and (tilt_diff <= 8.0):
        return True
    if (pan_diff <= 5.0) and (tilt_diff <= 5.0) and (dist_diff <= 15.0):
        return True
    return False


def select_scan_candidate(index):
    global best_scan_tennis, scan_candidate_index

    if index < 0 or index >= len(scan_ranked_tennis):
        best_scan_tennis = None
        return False

    scan_candidate_index = index
    best_scan_tennis = scan_ranked_tennis[index].copy()
    ensure_target_id(best_scan_tennis)
    return True


def remember_scan_target(target):
    global best_scan_tennis

    if target is None:
        return
    if target.get("miss", 0) > 0:
        return
    if target.get("dist_cm") is None:
        return

    scan_target = target.copy()
    scan_target["servo_pan"] = pan_angle
    scan_target["servo_tilt"] = tilt_angle

    matched_index = -1
    for idx in range(len(scan_ranked_tennis)):
        if is_same_scan_target(scan_ranked_tennis[idx], scan_target):
            matched_index = idx
            break

    if matched_index >= 0:
        if scan_observation_key(scan_target) < scan_observation_key(scan_ranked_tennis[matched_index]):
            prev_id = int(scan_ranked_tennis[matched_index].get("id", 0))
            scan_ranked_tennis[matched_index] = scan_target
            if prev_id > 0:
                ensure_target_id(scan_ranked_tennis[matched_index], prev_id)
    else:
        scan_ranked_tennis.append(scan_target)

    scan_ranked_tennis.sort(key=scan_rank_key)
    select_scan_candidate(0)


def begin_scan_round():
    global pick_substate, scan_seek_left, scan_direction, scan_lock_count
    global nearest_switch_count, best_scan_tennis, tracked_tennis
    global scan_ranked_tennis, scan_candidate_index, pick_confirm_start_ms, pick_track_start_ms

    pick_substate = PICK_SCAN
    scan_seek_left = True
    scan_direction = 1
    scan_lock_count = 0
    nearest_switch_count = 0
    best_scan_tennis = None
    scan_ranked_tennis = []
    scan_candidate_index = 0
    tracked_tennis = None
    pick_confirm_start_ms = 0
    pick_track_start_ms = 0


def update_global_scan(img, nearest_tennis):
    global scan_seek_left, pan_angle, tilt_angle

    if not ENABLE_SERVOS:
        if nearest_tennis is not None:
            remember_scan_target(nearest_tennis)
        return len(scan_ranked_tennis) > 0

    if scan_seek_left:
        move_pan_toward(pan_angle_limit[0], SCAN_PAN_STEP)
        move_tilt_toward(SCAN_TILT_TARGET)
        if abs(pan_angle - pan_angle_limit[0]) <= SCAN_ALIGN_MARGIN:
            scan_seek_left = False
            return False
        return False

    apply_pan_delta(SCAN_PAN_STEP, SCAN_PAN_STEP)
    if nearest_tennis is None:
        move_tilt_toward(SCAN_TILT_TARGET)
    else:
        update_tilt_tracking(nearest_tennis, img)
        if tilt_angle < SCAN_TILT_FLOOR:
            tilt_angle = SCAN_TILT_FLOOR
        remember_scan_target(nearest_tennis)

    if pan_angle >= (pan_angle_limit[1] - SCAN_ALIGN_MARGIN):
        pan_angle = pan_angle_limit[1]
        return True
    return False


def update_return_to_saved_target():
    if best_scan_tennis is None:
        return True

    move_pan_toward(best_scan_tennis.get("servo_pan", PAN_INIT_ANGLE), RETURN_PAN_STEP)
    move_tilt_toward(best_scan_tennis.get("servo_tilt", TILT_INIT_ANGLE), RETURN_TILT_STEP)

    pan_ok = abs(pan_angle - best_scan_tennis.get("servo_pan", PAN_INIT_ANGLE)) <= RETURN_LOCK_MARGIN
    tilt_ok = abs(tilt_angle - best_scan_tennis.get("servo_tilt", TILT_INIT_ANGLE)) <= RETURN_LOCK_MARGIN
    return pan_ok and tilt_ok


def choose_confirmed_scan_target(saved_target, candidates):
    if saved_target is None or not candidates:
        return None

    best = None
    best_key = None
    gate = max(TRACK_GATE_MIN + 8, saved_target["radius"] * 4, 36)
    gate2 = gate * gate
    saved_dist = saved_target.get("dist_cm")

    for cand in candidates:
        dx = cand["cx"] - saved_target["cx"]
        dy = cand["cy"] - saved_target["cy"]
        d2 = (dx * dx) + (dy * dy)
        if d2 > gate2:
            continue

        dist_penalty = 0
        cand_dist = cand.get("dist_cm")
        if (saved_dist is not None) and (cand_dist is not None):
            dist_penalty = int(abs(saved_dist - cand_dist) * 8)

        key = (d2 + dist_penalty, -(cand["w"] * cand["h"]), -int(cand["score"] * 1000))
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand

    if best is None:
        return None

    matched = best.copy()
    keep_id = int(saved_target.get("id", 0))
    if keep_id > 0:
        ensure_target_id(matched, keep_id)
    else:
        ensure_target_id(matched)
    if "servo_pan" in saved_target:
        matched["servo_pan"] = saved_target["servo_pan"]
    if "servo_tilt" in saved_target:
        matched["servo_tilt"] = saved_target["servo_tilt"]
    return matched


def select_next_scan_candidate():
    global tracked_tennis, pick_confirm_start_ms

    tracked_tennis = None
    pick_confirm_start_ms = 0
    return select_scan_candidate(scan_candidate_index + 1)


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

    apply_pan_delta(scan_direction * scan_speed, SCAN_PAN_STEP)
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
    global racket_seen_prev, best_scan_tennis, tracked_tennis, tracked_player, scan_empty_rounds
    global scan_ranked_tennis, scan_candidate_index
    global scan_seek_left, pick_confirm_start_ms, pick_track_start_ms, current_racket_id, player_lock_start_ms

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
    scan_ranked_tennis = []
    scan_candidate_index = 0
    scan_empty_rounds = 0
    player_lock_start_ms = 0
    tracked_tennis = None
    tracked_player = None
    scan_seek_left = True
    pick_confirm_start_ms = 0
    pick_track_start_ms = 0
    current_racket_id = 0


def enter_play_mode():
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, scan_lock_count, nearest_switch_count, play_ready_frames
    global racket_seen_prev, best_scan_tennis, tracked_tennis, tracked_player, current_racket_id
    global scan_ranked_tennis, scan_candidate_index
    global scan_empty_rounds, player_lock_start_ms, pick_confirm_start_ms, pick_track_start_ms

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
    scan_ranked_tennis = []
    scan_candidate_index = 0
    scan_empty_rounds = 0
    player_lock_start_ms = 0
    pick_confirm_start_ms = 0
    pick_track_start_ms = 0
    tracked_tennis = None
    tracked_player = None
    current_racket_id = 0


def run_state_machine(img, tennis_target, tennis_candidates, player_target, racket_candidates):
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, play_ready_frames, racket_seen_prev
    global scan_lock_count, nearest_switch_count, best_scan_tennis, tracked_tennis
    global scan_ranked_tennis, scan_candidate_index
    global pick_confirm_start_ms, pick_track_start_ms, current_racket_id
    global scan_empty_rounds, player_lock_start_ms

    command_event = None
    racket_target = choose_racket_target(racket_candidates, player_target)
    racket_visible = racket_target is not None
    if is_live_target(player_target) and racket_visible:
        play_ready_frames += 1
    else:
        play_ready_frames = 0
    racket_present = play_ready_frames >= PLAY_RACKET_CONFIRM_FRAMES
    if racket_present:
        if current_racket_id <= 0:
            current_racket_id = allocate_target_id()
        ensure_target_id(racket_target, current_racket_id)
    else:
        current_racket_id = 0
        racket_target = None
    nearest_tennis = choose_nearest_tennis(tennis_candidates)
    active_target = tennis_target
    now_ms = time.ticks_ms()

    if capture_flash_frames > 0:
        capture_cmd = 1
        capture_flash_frames -= 1
    else:
        capture_cmd = 0

    read_picker_feedback()

    if mode == MODE_PICK:
        if pick_substate == PICK_SCAN:
            if update_servo_init():
                return None, "INIT", "INIT", racket_present, racket_target, command_event

            scan_done = update_global_scan(img, nearest_tennis)
            if scan_done:
                if len(scan_ranked_tennis) > 0 and select_scan_candidate(0):
                    scan_empty_rounds = 0
                    tracked_tennis = None
                    pick_confirm_start_ms = 0
                    pick_track_start_ms = 0
                    pick_substate = PICK_RETURN
                else:
                    scan_empty_rounds += 1
                    if scan_empty_rounds >= SCAN_EMPTY_ROUNDS_TO_PLAY:
                        enter_play_mode()
                        return None, "PLAY", "TRACK_P", False, None, command_event
                    begin_scan_round()
            # 扫描阶段只记录候选，不提前锁定或显示跟踪目标。
            return None, "SEEK", "SCAN", racket_present, racket_target, command_event

        if pick_substate == PICK_RETURN:
            if best_scan_tennis is None:
                if not select_scan_candidate(scan_candidate_index):
                    begin_scan_round()
                    return None, "SEEK", "SCAN", racket_present, racket_target, command_event

            active_target = best_scan_tennis
            if update_return_to_saved_target():
                pick_confirm_start_ms = now_ms
                pick_substate = PICK_CONFIRM
            return active_target, "SEEK", "RETURN", racket_present, racket_target, command_event

        if pick_substate == PICK_CONFIRM:
            if best_scan_tennis is None:
                if select_next_scan_candidate():
                    pick_substate = PICK_RETURN
                    return best_scan_tennis, "SEEK", "RETURN", racket_present, racket_target, command_event
                begin_scan_round()
                return None, "SEEK", "SCAN", racket_present, racket_target, command_event

            confirmed_target = choose_confirmed_scan_target(best_scan_tennis, tennis_candidates)
            if confirmed_target is None:
                if select_next_scan_candidate():
                    pick_substate = PICK_RETURN
                    return best_scan_tennis, "SEEK", "RETURN", racket_present, racket_target, command_event
                begin_scan_round()
                return None, "SEEK", "SCAN", racket_present, racket_target, command_event

            best_scan_tennis = confirmed_target.copy()
            active_target = confirmed_target
            update_servo_tracking(active_target, img)

            if pick_confirm_start_ms <= 0:
                pick_confirm_start_ms = now_ms

            if time.ticks_diff(now_ms, pick_confirm_start_ms) >= PICK_CONFIRM_DURATION_MS:
                tracked_tennis = confirmed_target.copy()
                ensure_target_id(tracked_tennis, int(confirmed_target.get("id", 0)))
                tracked_tennis["miss"] = 0
                pick_confirm_start_ms = 0
                pick_track_start_ms = now_ms
                pick_substate = PICK_TRACK
                return tracked_tennis, "SEEK", "TRACK", racket_present, racket_target, command_event

            return active_target, "SEEK", "CONFIRM", racket_present, racket_target, command_event

        active_target = tennis_target if tennis_target is not None else tracked_tennis
        if active_target is not None and active_target.get("miss", 0) == 0:
            update_servo_tracking(active_target, img)

        picker_done = read_picker_feedback()
        track_timeout = False
        if pick_track_start_ms > 0:
            track_timeout = time.ticks_diff(now_ms, pick_track_start_ms) >= PICK_TRACK_DURATION_MS

        if picker_done or track_timeout:
            capture_flash_frames = CAPTURE_HOLD_FRAMES
            capture_cmd = 1
            scan_empty_rounds = 0
            begin_scan_round()
            pick_track_start_ms = 0

        return active_target, "SEEK", "TRACK", racket_present, racket_target, command_event

    active_target = player_target
    if is_live_target(player_target):
        player_locked = True
        if racket_present and player_lock_start_ms <= 0:
            player_lock_start_ms = now_ms
        elif not racket_present:
            player_lock_start_ms = 0
        update_servo_tracking(player_target, img)
    else:
        player_locked = False
        player_lock_start_ms = 0

    if not player_locked:
        play_substate = PLAY_TRACK_PLAYER
        racket_seen_prev = False
        current_racket_id = 0
        return active_target, "PLAY", "TRACK_P", racket_present, racket_target, command_event

    play_substate = PLAY_TRACK_PLAYER
    racket_seen_prev = racket_present
    return active_target, "PLAY", "TRACK_P", racket_present, racket_target, command_event


def draw_status_panel(img, fps, mode_name, state_name, racket_present):
    global servo_init_frames_remaining, pick_confirm_start_ms, pick_track_start_ms, comm_state

    # 显示通信状态：no link / linked / ok
    try:
        if comm_state == COMM_NO_LINK:
            status_text = "no link"
            status_color = RED
        elif comm_state == COMM_LINKED:
            status_text = "linked"
            status_color = YELLOW
        elif comm_state == COMM_OK:
            status_text = "ok"
            status_color = GREEN
        else:
            status_text = "?"
            status_color = YELLOW
    except Exception:
        status_text = "?"
        status_color = YELLOW
    img.draw_string(2, 2, status_text, color=status_color, mono_space=False)
    img.draw_string(2, 20, "%.2f fps" % fps, color=WHITE, mono_space=False)
    img.draw_string(2, 38, "mode:%s" % mode_name, color=YELLOW, mono_space=False)
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
    if servo_init_frames_remaining > 0:
        img.draw_string(2, 182, "servo:init", color=YELLOW, mono_space=False)
    if mode_name == "PLAY":
        img.draw_string(
            2,
            200,
            "player:%d racket:%d" % (1 if player_locked else 0, 1 if racket_present else 0),
            color=WHITE,
            mono_space=False,
        )
    elif state_name == "CONFIRM" and pick_confirm_start_ms > 0:
        remain_ms = max(0, PICK_CONFIRM_DURATION_MS - time.ticks_diff(time.ticks_ms(), pick_confirm_start_ms))
        remain_ds = remain_ms // 100
        img.draw_string(
            2,
            200,
            "confirm:%d.%ds cand:%d/%d"
            % (
                remain_ds // 10,
                remain_ds % 10,
                scan_candidate_index + 1,
                len(scan_ranked_tennis),
            ),
            color=WHITE,
            mono_space=False,
        )
    elif state_name == "TRACK" and pick_track_start_ms > 0:
        remain_s = max(0, (PICK_TRACK_DURATION_MS - time.ticks_diff(time.ticks_ms(), pick_track_start_ms)) // 1000)
        img.draw_string(
            2,
            200,
            "track:%ds fb:%d" % (remain_s, picker_feedback_state),
            color=WHITE,
            mono_space=False,
        )
    elif best_scan_tennis is not None and best_scan_tennis.get("dist_cm") is not None:
        img.draw_string(
            2,
            200,
            "scan:%.1fcm p:%d t:%d fb:%d"
            % (
                best_scan_tennis["dist_cm"],
                int(best_scan_tennis.get("servo_pan", pan_angle)),
                int(best_scan_tennis.get("servo_tilt", tilt_angle)),
                picker_feedback_state,
            ),
            color=WHITE,
            mono_space=False,
        )


def estimate_distance(pixel_diameter, image_width=320):
    # 简化版距离估计，只用检测框尺度，避免额外的重图像处理。
    focal_length_mm = 2.8
    tennis_diameter_mm = 67
    sensor_width_mm = 4.896

    if pixel_diameter <= 0:
        return None

    mm_per_pixel = sensor_width_mm / image_width
    projected_mm = pixel_diameter * mm_per_pixel
    if projected_mm <= 0:
        return None

    distance_mm = (focal_length_mm * tennis_diameter_mm) / projected_mm
    return distance_mm / 10.0


def estimate_corrected_distance_cm(pixel_diameter, radius):
    base_distance_cm = estimate_distance(pixel_diameter)
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
    prev_radius = target.get("refined_radius")
    diameter, cue_conf = estimate_tennis_diameter(
        img, target["x"], target["y"], target["w"], target["h"], allow_hough=allow_hough
    )
    raw_radius = fuse_tennis_radius(target["w"], target["h"], diameter)
    filtered_diameter, refined_radius = update_trimmed_tennis_measure(target, diameter, raw_radius)
    refined_radius = smooth_ball_radius(refined_radius, prev_radius)

    refined_dist_cm = estimate_corrected_distance_cm(filtered_diameter, refined_radius)
    prev_dist_cm = target.get("refined_dist_cm")
    if (prev_dist_cm is not None) and (refined_dist_cm is not None):
        refined_dist_cm = (prev_dist_cm * 0.7) + (refined_dist_cm * 0.3)

    target["radius"] = refined_radius
    target["dist_cm"] = refined_dist_cm
    target["refined_radius"] = refined_radius
    target["refined_dist_cm"] = refined_dist_cm
    target["refined_diameter"] = filtered_diameter
    target["raw_refined_diameter"] = diameter
    target["raw_refined_radius"] = raw_radius
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
    global labels, net, last_hello_ms

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
    init_uart()
    # 启动阶段先发送一次 hello，之后由 try_send_hello 在主循环按周期发送
    try:
        if uart is not None:
            uart_write_line("hello")
            try:
                last_hello_ms = time.ticks_ms()
            except Exception:
                pass
    except Exception:
        pass
    init_picker_feedback()
    if ENABLE_SERVOS:
        try:
            show_message("Model OK", "Init servos...")
            init_servos()
        except Exception as err:
            halt_with_error("SERVO FAIL", err)
    begin_scan_round()
    show_message("Model OK", "Running...")


def main_loop():
    global frame_index

    clock = time.clock()

    while True:
        clock.tick()
        try:
            frame_index += 1
            # 处理 UART 接收并尝试非阻塞发送 hello（不影响主循环）
            process_uart_rx()
            try_send_hello()
            img = sensor.snapshot()
            draw_grid(img, GRID_ROWS, GRID_COLS, GRID_COLOR)
            predictions = net.predict([img], callback=fomo_post_process)
            tennis_candidates = []
            player_candidates = []
            racket_candidates = []
            detect_tennis = mode == MODE_PICK
            detect_play_targets = mode == MODE_PLAY

            for i, detection_list in enumerate(predictions):
                if i == 0:
                    continue

                conf_th = threshold_for_class(i)
                detection_list = [d for d in detection_list if d[4] >= conf_th]
                if not detection_list:
                    continue

                label_name = labels[i] if i < len(labels) else ("class_%d" % i)
                label_l = label_name.lower()
                is_tennis = ("tennis" in label_l) and ("player" not in label_l) and ("racket" not in label_l)
                is_player = "player" in label_l
                is_racket = "racket" in label_l

                if detect_tennis:
                    if not is_tennis:
                        continue
                elif detect_play_targets:
                    if (not is_player) and (not is_racket):
                        continue
                else:
                    continue

                for x, y, w, h, score in detection_list:
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    row, col = get_grid_position(
                        center_x, center_y, img.width(), img.height(), GRID_ROWS, GRID_COLS
                    )

                    if detect_tennis and is_tennis:
                        radius = max(8, min(40, int(max(w, h) * 0.7)))
                        dist_cm = estimate_distance(max(w, h), image_width=img.width())
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

                    elif detect_play_targets and is_player:
                        player_candidates.append(
                            {
                                "cx": center_x,
                                "cy": center_y,
                                "radius": 20,
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "score": score,
                                "dist_cm": None,
                                "row": row,
                                "col": col,
                                "kind": "player",
                            }
                        )

                    elif detect_play_targets and is_racket:
                        racket_candidates.append(
                            {
                                "cx": center_x,
                                "cy": center_y,
                                "radius": max(8, min(24, max(w, h) // 2)),
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "score": score,
                                "dist_cm": None,
                                "row": row,
                                "col": col,
                                "kind": "racket",
                            }
                        )

            refine_all_tennis = (
                ENABLE_TENNIS_REFINEMENT
                and REFINE_ALL_TENNIS_IN_PICK_SCAN
                and (mode == MODE_PICK)
                and (pick_substate == PICK_SCAN)
            )
            if refine_all_tennis:
                tennis_candidates = refine_tennis_candidates(img, tennis_candidates, allow_hough=True)

            seek_track_enabled = detect_tennis and (pick_substate == PICK_TRACK)
            tennis_target = match_tennis_track(tennis_candidates) if seek_track_enabled else None
            if detect_tennis and (not refine_all_tennis):
                tennis_target = refine_tennis_target(img, tennis_target)
            player_target = choose_player_target(player_candidates) if detect_play_targets else None
            active_target, mode_name, state_name, racket_present, racket_target, command_event = run_state_machine(
                img, tennis_target, tennis_candidates, player_target, racket_candidates
            )
            draw_seek_tennis_candidates(img, mode_name, state_name, tennis_candidates)
            draw_active_target(img, active_target)
            draw_status_panel(img, clock.fps(), mode_name, state_name, racket_present)
            draw_player_countdown(img, mode_name, racket_present)
            send_runtime_packets(mode_name, active_target)
            if command_event is not None:
                send_command_packet(command_event[0], command_event[1], mode_name, state_name)
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
