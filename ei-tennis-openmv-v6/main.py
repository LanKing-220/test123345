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
UART_SEND_INTERVAL_FRAMES = 5
ENABLE_GRID_OVERLAY = True

# 通信状态：0=no link, 1=linked(收到HI), 2=ok(收到OK)
COMM_NO_LINK = 0
COMM_LINKED = 1
COMM_OK = 2
comm_state = COMM_NO_LINK
last_hello_ms = 0
HELLO_INTERVAL_MS = 5000

# 主机控制命令：采用 REQ/CONFIRM 二次确认，避免串口噪声误触发。
# 仅在链路 COMM_OK 且当前状态允许时才会真正执行。
HOST_CTRL_CONFIRM_TIMEOUT_MS = 3000
HOST_CTRL_ACTION_UNLOCK_TRACK = "UNLOCK_TRACK"
HOST_CTRL_ACTION_MODE_PICK = "MODE_PICK"
HOST_CTRL_ACTION_MODE_PLAY = "MODE_PLAY"
HOST_CTRL_ACTIONS = (
    HOST_CTRL_ACTION_MODE_PICK,
    HOST_CTRL_ACTION_MODE_PLAY,
)

THRESH_TENNIS = 0.4
# 人和球拍更容易误触发，阈值调高后会更保守，降低敏感度。
THRESH_PLAYER = 0.2
THRESH_RACKET = 0.2

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
grid_overlay = None
grid_mask = None
grid_overlay_ready = False

TENNIS_TRACK_MAX_MISS = 30
PLAYER_TRACK_MAX_MISS = 20
TRACK_GATE_MIN = 28
TRACK_SMOOTH_OLD_NUM = 7
TRACK_SMOOTH_NEW_NUM = 3

PAN_INIT_ANGLE = 90.0
TILT_INIT_ANGLE = 120.0
SERVO_INIT_HOLD_FRAMES = 28
SERVO_PWM_STAGGER_FRAMES = 8
SERVO_SPEED_SCALE = 1.5

pan_angle = PAN_INIT_ANGLE
tilt_angle = TILT_INIT_ANGLE
# 扩大水平扫描范围到 180° 舵机可用行程。
pan_angle_limit = [0.0, 180.0]
tilt_angle_limit = [80.0, 150.0]
SERVO_DEADBAND = 6
SCAN_TILT_TARGET = 120.0
SCAN_TILT_FLOOR = 119.0
SCAN_TILT_STEP = 1.2 * SERVO_SPEED_SCALE
SERVO_INIT_STEP = 0.35 * SERVO_SPEED_SCALE
SCAN_PAN_STEP = 0.75 * SERVO_SPEED_SCALE
TRACK_PAN_MAX_STEP = 1.1 * SERVO_SPEED_SCALE
TRACK_TILT_MAX_STEP = 1.0 * SERVO_SPEED_SCALE
PLAY_SEARCH_TILT_STEP = 0.45 * SERVO_SPEED_SCALE

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
PLAY_SEARCH_PLAYER = 1
PLAY_WAIT_SERVE = PLAY_SEARCH_PLAYER

PICK_CONFIRM_MAX_FRAMES = 20
PICK_EARLY_LOCK_WINDOW_FRAMES = 10
PICK_EARLY_LOCK_SEEN_FRAMES = 4
PICK_FINAL_LOCK_MIN_SEEN_FRAMES = 7
PICK_SCAN_FAIL_ROUNDS_TO_PLAY = 3
CAPTURE_DISTANCE_CM = 10.0
CAPTURE_HOLD_FRAMES = 8
NEAREST_SWITCH_MARGIN_CM = 6.0
NEAREST_SWITCH_CONFIRM_FRAMES = 2
PICK_TRACK_LOST_CONSECUTIVE_TH = 30
PLAY_RACKET_CONFIRM_FRAMES = 1
PLAY_HIT_WINDOW_FRAMES = 5
PLAY_HIT_MIN_PLAYER_FRAMES = 2
PLAY_HIT_MIN_RACKET_FRAMES = 2
PLAY_HIT_SIGNAL_CMD = "HIT"
PLAY_HIT_SIGNAL_ARG = 1
PLAYER_LOCK_WINDOW_FRAMES = 20
PLAYER_LOCK_MIN_HIT_FRAMES = 6
RACKET_LINK_MARGIN_X_RATIO = 0.60
RACKET_LINK_MARGIN_Y_RATIO = 0.35
SCAN_ALIGN_MARGIN = 1.0
RETURN_LOCK_MARGIN = 2.0
RETURN_PAN_STEP = 1.0 * SERVO_SPEED_SCALE
RETURN_TILT_STEP = 1.0 * SERVO_SPEED_SCALE
SEEK_TILT_RESET_MARGIN = 1.0
LOCK_ALIGN_MARGIN_X = 18
LOCK_ALIGN_MARGIN_Y = 16
LOCK_PAN_RECORD_MARGIN_X = 26
LOCK_PAN_RECORD_MARGIN_Y = 22

# 收紧中心颜色过滤，降低浅黄色非网球目标的误判概率。
TENNIS_CENTER_GREEN_L_MIN = 20
TENNIS_CENTER_GREEN_L_MAX = 90
TENNIS_CENTER_GREEN_A_MIN = -38
TENNIS_CENTER_GREEN_A_MAX = 18
TENNIS_CENTER_GREEN_B_MIN = 14
TENNIS_CENTER_GREEN_B_MAX = 84
TENNIS_CENTER_GREEN_RATIO_MIN_PCT = 16
TENNIS_CENTER_GREEN_STRONG_RATIO_PCT = 24
TENNIS_CENTER_GREEN_BIAS_MIN = 16
TENNIS_CENTER_GREEN_CHROMA_MIN = 20
TENNIS_CENTER_GREEN_DOMINANCE_MARGIN_PCT = 10
TENNIS_CENTER_GREEN_SMALL_DOMINANCE_MARGIN_PCT = 4
TENNIS_CENTER_GREEN_ROI_MIN = 6
TENNIS_CENTER_GREEN_ROI_MAX = 18
TENNIS_CENTER_WHITE_L_MIN = 54
TENNIS_CENTER_WHITE_A_MIN = -10
TENNIS_CENTER_WHITE_A_MAX = 10
TENNIS_CENTER_WHITE_B_MIN = -8
TENNIS_CENTER_WHITE_B_MAX = 18
TENNIS_CENTER_WHITE_CHROMA_MAX = 18
TENNIS_CENTER_WHITE_RATIO_MIN_PCT = 42
TENNIS_CENTER_WHITE_STRONG_RATIO_PCT = 62
TENNIS_CENTER_WHITE_DOMINANCE_MARGIN_PCT = 18
TENNIS_CENTER_WHITE_GREEN_BIAS_MAX = 16
TENNIS_CENTER_PALE_YELLOW_L_MIN = 58
TENNIS_CENTER_PALE_YELLOW_A_MIN = -6
TENNIS_CENTER_PALE_YELLOW_A_MAX = 18
TENNIS_CENTER_PALE_YELLOW_B_MIN = 18
TENNIS_CENTER_PALE_YELLOW_B_MAX = 44
TENNIS_CENTER_PALE_YELLOW_CHROMA_MAX = 42
TENNIS_CENTER_PALE_YELLOW_GREEN_RATIO_MAX_PCT = 26
TENNIS_CENTER_PALE_YELLOW_BIAS_MAX = 20
PLAY_ENTER_TILT_LIFT_DEG = 30.0

PICKER_FEEDBACK_PIN = None
PICKER_FEEDBACK_ACTIVE_LEVEL = 1

mode = MODE_PICK
pick_substate = PICK_SCAN

capture_cmd = 0
capture_flash_frames = 0
player_locked = False
balls_served = 0
balls_picked = 0
target_balls = 5
scan_direction = 1
scan_speed = SCAN_PAN_STEP
nearest_switch_count = 0
play_ready_frames = 0
play_presence_window = []
play_hit_latched = False
player_lock_window = []
best_scan_tennis = None
scan_ranked_tennis = []
scan_candidate_index = 0
servo_init_frames_remaining = 0
scan_seek_left = True
scan_tilt_reset_pending = False
pick_confirm_total_frames = 0
pick_confirm_seen_frames = 0
pick_scan_fail_rounds = 0
pick_track_lost_count = 0
picker_feedback_pin = None
picker_feedback_state = 0
picker_uart_done_pending = 0
next_target_id = 1
current_racket_id = 0
uart_rx_buffer = ""
host_ctrl_pending_action = ""
host_ctrl_pending_token = ""
host_ctrl_pending_ms = 0
host_track_unlock_pending = 0
host_mode_switch_pending = -1

ENABLE_TENNIS_REFINEMENT = True
HOUGH_INTERVAL = 3
DISTANCE_SCALE = 2.15
DISTANCE_OUTPUT_SCALE = 0.65
REFINE_ALL_TENNIS_IN_PICK_SCAN = True
MEASURE_WINDOW_LEN = 10
MEASURE_TRIM_COUNT = 2
SCAN_LOCK_RADIUS_RANK_WINDOW = 3

COLOR_TOL_L_BASE = 12
COLOR_TOL_A_BASE = 10
COLOR_TOL_B_BASE = 10
COLOR_TOL_L_EXTRA = 3
COLOR_TOL_A_EXTRA = 3
COLOR_TOL_B_EXTRA = 3.5

DRAW_RADIUS_SCALE = 1.15
CLOSE_DRAW_RADIUS_SCALE = 1.00
ROI_NEAR_SWITCH = 26
FAR_ROI_PAD_MIN = 6
MAX_COLOR_BLOB_AREA_MULT = 6
CLOSE_COLOR_BLOB_AREA_RATIO_PCT = 96
CLOSE_COLOR_PERCENTILE_LO = 0.10
CLOSE_COLOR_PERCENTILE_HI = 0.90
CLOSE_COLOR_L_MARGIN = 6
CLOSE_COLOR_A_MARGIN = 5
CLOSE_COLOR_B_MARGIN = 6
CLOSE_COLOR_ROUNDNESS_MIN = 0.20
CLOSE_COLOR_DENSITY_MIN = 0.20
CLOSE_BALL_BBOX_SIZE_TH = 32
CLOSE_BALL_MERGE_COUNT_TH = 2
CLOSE_BALL_ROI_PAD_MIN = 18
CLOSE_HOUGH_X_MARGIN = 12
CLOSE_HOUGH_Y_MARGIN = 12
CLOSE_HOUGH_R_MARGIN = 14
CLOSE_HOUGH_R_MAX = 140
FOMO_DUPLICATE_GENERAL_IOU = 0.35
FOMO_DUPLICATE_CLOSE_IOU = 0.08
FOMO_DUPLICATE_CLOSE_SIZE_TH = 18
FOMO_DUPLICATE_CENTER_PAD = 14
CLOSE_REFINE_DUPLICATE_DISTANCE_CM = 10.0
CLOSE_REFINE_DUPLICATE_CENTER_MIN = 10
CLOSE_REFINE_DUPLICATE_CENTER_RATIO_PCT = 40
CLOSE_REFINE_DUPLICATE_DIAMETER_RATIO_MAX_PCT = 180
CLOSE_REFINE_DUPLICATE_DIST_DIFF_CM = 4.0
CLOSE_REFINE_DUPLICATE_CENTER_PAD = 8
CIRCLE_DENSITY_REF = 0.78539816339

EDGE_LOW_TH = 55
EDGE_HIGH_TH = 110

GLARE_L_TH = 88
GLARE_RATIO_TH_PCT = 18
SMALL_BALL_SIZE_TH = 24
GLARE_DIAMETER_CAP_PCT = 115
FAR_BALL_SIZE_TH = 16
FAR_BALL_HOUGH_R_MAX = 20
FAR_BALL_DIAMETER_RATIO_MIN_PCT = 70
FAR_BALL_DIAMETER_RATIO_MAX_PCT = 150
GENERAL_DIAMETER_RATIO_MIN_PCT = 65
GENERAL_DIAMETER_RATIO_MAX_PCT = 190
MID_BALL_HOUGH_BBOX_MIN = 10
MID_BALL_HOUGH_BBOX_MAX = 22
MID_BALL_HOUGH_DIAMETER_BOOST_PCT = 108

FOMO_HEATMAP_AREA_TH = 2
FOMO_HEATMAP_PIXELS_TH = 2
FOMO_HEATMAP_MERGE_MARGIN = 1
FOMO_HEATMAP_SCORE_TH = 0.22

LOCAL_NEAREST_GATE_MIN = 24
LOCAL_NEAREST_GATE_RADIUS_SCALE = 4

# 相机色彩参数：为避免“局部色彩异常”，默认启用自动白平衡做整屏统一校正。
# 若需固定色彩可把 CAMERA_AUTO_WHITEBAL 设为 False，并启用手动 RGB 增益。
CAMERA_AUTO_WHITEBAL = True
CAMERA_MANUAL_WHITEBAL = False
CAMERA_MANUAL_RGB_GAIN_DB = (0.3, 0.0, 0.8)
CAMERA_MANUAL_GAIN_DB = 1.0
CAMERA_MANUAL_EXPOSURE_US = 160000



def init_camera():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)

    # 完全手动：不走自动收敛，启动后直接固定参数。
    sensor.skip_frames(time=200)

    try:
        if CAMERA_AUTO_WHITEBAL:
            sensor.set_auto_whitebal(True)
        elif CAMERA_MANUAL_WHITEBAL:
            sensor.set_auto_whitebal(False, rgb_gain_db=CAMERA_MANUAL_RGB_GAIN_DB)
        else:
            sensor.set_auto_whitebal(False)
    except Exception:
        sensor.set_auto_whitebal(True if CAMERA_AUTO_WHITEBAL else False)

    try:
        sensor.set_auto_gain(False, gain_db=CAMERA_MANUAL_GAIN_DB)
    except Exception:
        sensor.set_auto_gain(False)

    try:
        sensor.set_auto_exposure(False, exposure_us=CAMERA_MANUAL_EXPOSURE_US)
    except Exception:
        sensor.set_auto_exposure(False)

    sensor.skip_frames(time=300)


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


def init_grid_overlay():
    global grid_overlay, grid_mask, grid_overlay_ready

    grid_overlay = None
    grid_mask = None
    grid_overlay_ready = False

    if not ENABLE_GRID_OVERLAY:
        return

    try:
        gc.collect()
        w = sensor.width()
        h = sensor.height()
        grid_overlay = sensor.alloc_extra_fb(w, h, sensor.RGB565)
        grid_mask = sensor.alloc_extra_fb(w, h, sensor.BINARY)
        grid_overlay.draw_rectangle((0, 0, w, h), color=(0, 0, 0), fill=True)
        grid_mask.draw_rectangle((0, 0, w, h), color=0, fill=True)
        draw_grid(grid_overlay, GRID_ROWS, GRID_COLS, GRID_COLOR)
        draw_grid(grid_mask, GRID_ROWS, GRID_COLS, 1)
        grid_overlay_ready = True
    except Exception as err:
        grid_overlay = None
        grid_mask = None
        grid_overlay_ready = False
        sys.print_exception(err)


def draw_grid_overlay(img):
    global grid_overlay_ready

    if not grid_overlay_ready:
        return

    try:
        img.draw_image(grid_overlay, 0, 0, mask=grid_mask)
    except Exception as err:
        grid_overlay_ready = False
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


def host_control_enabled():
    return comm_state == COMM_OK


def host_action_supported(action):
    return action in HOST_CTRL_ACTIONS


def host_action_allowed(action):
    if action == HOST_CTRL_ACTION_MODE_PICK:
        return mode == MODE_PLAY
    if action == HOST_CTRL_ACTION_MODE_PLAY:
        return mode == MODE_PICK
    return False


def clear_host_ctrl_pending():
    global host_ctrl_pending_action, host_ctrl_pending_token, host_ctrl_pending_ms

    host_ctrl_pending_action = ""
    host_ctrl_pending_token = ""
    host_ctrl_pending_ms = 0


def consume_host_track_unlock_event():
    global host_track_unlock_pending

    if host_track_unlock_pending <= 0:
        return False
    host_track_unlock_pending -= 1
    return True


def consume_host_mode_switch_event():
    global host_mode_switch_pending

    mode_to_switch = host_mode_switch_pending
    host_mode_switch_pending = -1
    return mode_to_switch


def parse_host_ctrl_message(text):
    global host_ctrl_pending_action, host_ctrl_pending_token, host_ctrl_pending_ms
    global host_track_unlock_pending, host_mode_switch_pending

    # 协议：CTRL,ACTION,REQ|CONFIRM,TOKEN
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 4:
        return False
    if parts[0].upper() != "CTRL":
        return False

    action = parts[1].upper()
    phase = parts[2].upper()
    token = parts[3]
    if (not action) or (not phase) or (not token):
        return True

    if (not host_control_enabled()) or (not host_action_supported(action)):
        uart_write_line("NACK,%s,%s,%s" % (action, phase, token))
        return True

    if phase == "REQ":
        if not host_action_allowed(action):
            uart_write_line("NACK,%s,REQ,%s" % (action, token))
            return True
        host_ctrl_pending_action = action
        host_ctrl_pending_token = token
        host_ctrl_pending_ms = time.ticks_ms()
        uart_write_line("ACK,%s,REQ,%s" % (action, token))
        return True

    if phase != "CONFIRM":
        uart_write_line("NACK,%s,%s,%s" % (action, phase, token))
        return True

    now_ms = time.ticks_ms()
    age_ms = time.ticks_diff(now_ms, host_ctrl_pending_ms) if host_ctrl_pending_ms > 0 else -1
    pending_alive = (host_ctrl_pending_ms > 0) and (age_ms >= 0) and (age_ms <= HOST_CTRL_CONFIRM_TIMEOUT_MS)
    matched = pending_alive and (action == host_ctrl_pending_action) and (token == host_ctrl_pending_token)

    if matched and host_action_allowed(action):
        if action == HOST_CTRL_ACTION_UNLOCK_TRACK:
            host_track_unlock_pending += 1
        elif action == HOST_CTRL_ACTION_MODE_PICK:
            host_mode_switch_pending = MODE_PICK
        elif action == HOST_CTRL_ACTION_MODE_PLAY:
            host_mode_switch_pending = MODE_PLAY
        uart_write_line("ACK,%s,CONFIRM,%s" % (action, token))
        clear_host_ctrl_pending()
        return True

    uart_write_line("NACK,%s,CONFIRM,%s" % (action, token))
    if matched or ((host_ctrl_pending_ms > 0) and (not pending_alive)):
        clear_host_ctrl_pending()
    return True


def handle_uart_line(line):
    global balls_served, balls_picked, picker_uart_done_pending
    global comm_state

    if not line:
        return

    text = line.strip()
    if not text:
        return

    if parse_host_ctrl_message(text):
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
    if lcd is None:
        return

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


def center_sample_roi(img, cx, cy, size):
    roi_size = int(clamp(size, 1, min(img.width(), img.height())))
    rx = clamp(cx - (roi_size // 2), 0, img.width() - 1)
    ry = clamp(cy - (roi_size // 2), 0, img.height() - 1)
    rw = clamp(roi_size, 1, img.width() - rx)
    rh = clamp(roi_size, 1, img.height() - ry)
    return (rx, ry, rw, rh)


def tennis_center_is_green(img, x, y, w, h):
    cx = x + (w // 2)
    cy = y + (h // 2)
    roi_size = max(TENNIS_CENTER_GREEN_ROI_MIN, max(w, h) // 3)
    roi_size = min(TENNIS_CENTER_GREEN_ROI_MAX, roi_size)
    roi = center_sample_roi(img, cx, cy, roi_size)

    stats = img.get_statistics(roi=roi)
    l_mean = stats.l_mean()
    a_mean = stats.a_mean()
    b_mean = stats.b_mean()
    green_bias = b_mean - a_mean
    chroma = abs(a_mean) + abs(b_mean)

    green_blobs = img.find_blobs(
        [
            (
                TENNIS_CENTER_GREEN_L_MIN,
                TENNIS_CENTER_GREEN_L_MAX,
                TENNIS_CENTER_GREEN_A_MIN,
                TENNIS_CENTER_GREEN_A_MAX,
                TENNIS_CENTER_GREEN_B_MIN,
                TENNIS_CENTER_GREEN_B_MAX,
            )
        ],
        roi=roi,
        x_stride=1,
        y_stride=1,
        area_threshold=1,
        pixels_threshold=1,
        merge=True,
        margin=1,
    )

    green_pixels = 0
    for blob in green_blobs:
        green_pixels += blob.pixels()

    white_blobs = img.find_blobs(
        [
            (
                TENNIS_CENTER_WHITE_L_MIN,
                100,
                TENNIS_CENTER_WHITE_A_MIN,
                TENNIS_CENTER_WHITE_A_MAX,
                TENNIS_CENTER_WHITE_B_MIN,
                TENNIS_CENTER_WHITE_B_MAX,
            )
        ],
        roi=roi,
        x_stride=1,
        y_stride=1,
        area_threshold=1,
        pixels_threshold=1,
        merge=True,
        margin=1,
    )

    white_pixels = 0
    for blob in white_blobs:
        white_pixels += blob.pixels()

    roi_pixels = max(1, roi[2] * roi[3])
    small_target = max(w, h) <= FAR_BALL_SIZE_TH
    green_ratio_pct = (green_pixels * 100) // roi_pixels
    white_ratio_pct = (white_pixels * 100) // roi_pixels

    if small_target:
        green_ratio_min_pct = max(8, TENNIS_CENTER_GREEN_RATIO_MIN_PCT - 4)
        strong_green_min_pct = max(14, TENNIS_CENTER_GREEN_STRONG_RATIO_PCT - 4)
        white_ratio_min_pct = TENNIS_CENTER_WHITE_RATIO_MIN_PCT + 8
        white_strong_min_pct = TENNIS_CENTER_WHITE_STRONG_RATIO_PCT + 6
        white_dom_margin_pct = TENNIS_CENTER_WHITE_DOMINANCE_MARGIN_PCT + 8
        green_dom_margin_pct = TENNIS_CENTER_GREEN_SMALL_DOMINANCE_MARGIN_PCT
        strong_green_chroma_min = TENNIS_CENTER_GREEN_CHROMA_MIN + 2
    else:
        green_ratio_min_pct = TENNIS_CENTER_GREEN_RATIO_MIN_PCT
        strong_green_min_pct = TENNIS_CENTER_GREEN_STRONG_RATIO_PCT
        white_ratio_min_pct = TENNIS_CENTER_WHITE_RATIO_MIN_PCT
        white_strong_min_pct = TENNIS_CENTER_WHITE_STRONG_RATIO_PCT
        white_dom_margin_pct = TENNIS_CENTER_WHITE_DOMINANCE_MARGIN_PCT
        green_dom_margin_pct = TENNIS_CENTER_GREEN_DOMINANCE_MARGIN_PCT
        strong_green_chroma_min = TENNIS_CENTER_GREEN_CHROMA_MIN + 4

    mean_green = (
        (TENNIS_CENTER_GREEN_L_MIN <= l_mean <= TENNIS_CENTER_GREEN_L_MAX)
        and (TENNIS_CENTER_GREEN_A_MIN <= a_mean <= TENNIS_CENTER_GREEN_A_MAX)
        and (TENNIS_CENTER_GREEN_B_MIN <= b_mean <= TENNIS_CENTER_GREEN_B_MAX)
        and (green_bias >= TENNIS_CENTER_GREEN_BIAS_MIN)
        and (chroma >= TENNIS_CENTER_GREEN_CHROMA_MIN)
    )

    strong_green = (green_ratio_pct >= strong_green_min_pct) and (
        green_bias >= TENNIS_CENTER_GREEN_BIAS_MIN
    ) and (chroma >= strong_green_chroma_min)
    mean_white = (
        (l_mean >= TENNIS_CENTER_WHITE_L_MIN)
        and (TENNIS_CENTER_WHITE_A_MIN <= a_mean <= TENNIS_CENTER_WHITE_A_MAX)
        and (TENNIS_CENTER_WHITE_B_MIN <= b_mean <= TENNIS_CENTER_WHITE_B_MAX)
        and (chroma <= TENNIS_CENTER_WHITE_CHROMA_MAX)
        and (green_bias <= TENNIS_CENTER_WHITE_GREEN_BIAS_MAX)
    )
    green_dominant = green_ratio_pct >= (white_ratio_pct + green_dom_margin_pct)
    pale_bright = (l_mean >= (TENNIS_CENTER_WHITE_L_MIN + 8)) and (
        chroma <= (TENNIS_CENTER_WHITE_CHROMA_MAX + 8)
    )
    pale_yellow = (
        (l_mean >= TENNIS_CENTER_PALE_YELLOW_L_MIN)
        and (TENNIS_CENTER_PALE_YELLOW_A_MIN <= a_mean <= TENNIS_CENTER_PALE_YELLOW_A_MAX)
        and (TENNIS_CENTER_PALE_YELLOW_B_MIN <= b_mean <= TENNIS_CENTER_PALE_YELLOW_B_MAX)
        and (chroma <= TENNIS_CENTER_PALE_YELLOW_CHROMA_MAX)
        and (green_ratio_pct <= TENNIS_CENTER_PALE_YELLOW_GREEN_RATIO_MAX_PCT)
        and (green_bias <= TENNIS_CENTER_PALE_YELLOW_BIAS_MAX)
    )
    white_dominant = (white_ratio_pct >= white_ratio_min_pct) and (
        white_ratio_pct >= (green_ratio_pct + white_dom_margin_pct)
    )
    if pale_bright and (green_ratio_pct < strong_green_min_pct):
        return False
    if pale_yellow:
        return False
    if (mean_white and (white_ratio_pct >= white_ratio_min_pct)) or white_dominant:
        return False
    if white_ratio_pct >= white_strong_min_pct:
        return False
    if not green_dominant:
        return False
    return (mean_green and (green_ratio_pct >= green_ratio_min_pct)) or strong_green


def lift_tilt_for_play_mode():
    if not ENABLE_SERVOS:
        return
    if p9 is None:
        return

    start_tilt_pwm()


def play_search_tilt_target():
    return clamp(
        TILT_INIT_ANGLE - PLAY_ENTER_TILT_LIFT_DEG,
        tilt_angle_limit[0],
        tilt_angle_limit[1],
    )


def update_play_search_motion(img):
    global pan_angle, scan_direction

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

    move_tilt_toward(play_search_tilt_target(), PLAY_SEARCH_TILT_STEP)


def clamp_diameter_by_bbox(diameter, bbox_d, close_mode=False):
    bbox_d = max(3, int(bbox_d))
    if close_mode:
        min_ratio = max(60, GENERAL_DIAMETER_RATIO_MIN_PCT - 5)
        max_ratio = min(210, GENERAL_DIAMETER_RATIO_MAX_PCT + 15)
    else:
        min_ratio = GENERAL_DIAMETER_RATIO_MIN_PCT
        max_ratio = GENERAL_DIAMETER_RATIO_MAX_PCT

    if bbox_d <= FAR_BALL_SIZE_TH:
        min_ratio = max(min_ratio, FAR_BALL_DIAMETER_RATIO_MIN_PCT)
        max_ratio = min(max_ratio, FAR_BALL_DIAMETER_RATIO_MAX_PCT)

    d_min = max(4, (bbox_d * min_ratio) // 100)
    d_max = max(d_min, (bbox_d * max_ratio) // 100)
    return int(clamp(diameter, d_min, d_max))


def target_center_error(target, img):
    if target is None:
        return 9999, 9999
    return target["cx"] - (img.width() // 2), target["cy"] - (img.height() // 2)


def target_is_aligned(target, img, margin_x=LOCK_ALIGN_MARGIN_X, margin_y=LOCK_ALIGN_MARGIN_Y):
    if target is None:
        return False
    dx, dy = target_center_error(target, img)
    return (abs(dx) <= margin_x) and (abs(dy) <= margin_y)


def update_target_lock_pan(target, img, pan_value=None):
    if target is None:
        return False
    if target.get("miss", 0) > 0:
        return False
    if not target_is_aligned(target, img, LOCK_PAN_RECORD_MARGIN_X, LOCK_PAN_RECORD_MARGIN_Y):
        return False

    if pan_value is None:
        pan_value = pan_angle
    target["lock_pan"] = pan_value
    target["lock_pan_ms"] = time.ticks_ms()
    return True


def get_grid_position(x, y, img_w, img_h, rows, cols):
    col = min(cols - 1, max(0, (x * cols) // img_w))
    row = min(rows - 1, max(0, (y * rows) // img_h))
    return row, col


def rect_iou(rect_a, rect_b):
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_w = min(ax2, bx2) - max(ax, bx)
    inter_h = min(ay2, by2) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter_area = inter_w * inter_h
    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def rect_contains_point(rect, px, py, pad=0):
    x, y, w, h = rect
    return (px >= (x - pad)) and (px <= (x + w + pad)) and (py >= (y - pad)) and (py <= (y + h + pad))


def is_close_range_candidate(target):
    if target is None:
        return False

    merge_count = int(target.get("merge_count", 1))
    if merge_count >= CLOSE_BALL_MERGE_COUNT_TH:
        return True

    bbox_size = max(target.get("w", 0), target.get("h", 0))
    if bbox_size >= CLOSE_BALL_BBOX_SIZE_TH:
        return True

    return False


def target_raw_distance_cm(target):
    if target is None:
        return None

    dist_cm = target.get("raw_dist_cm")
    if dist_cm is None:
        dist_cm = target.get("dist_cm")
    if dist_cm is None:
        return None
    return float(dist_cm)


def target_measure_diameter(target):
    if target is None:
        return None

    diameter = target.get("raw_refined_diameter")
    if diameter is None:
        diameter = target.get("refined_diameter")
    if diameter is None and ("w" in target) and ("h" in target):
        diameter = max(target["w"], target["h"])
    if diameter is None:
        return None
    return int(diameter)


def target_measure_center(target):
    if target is None:
        return None, None

    cx = target.get("measure_cx", target.get("cx"))
    cy = target.get("measure_cy", target.get("cy"))
    if cx is None or cy is None:
        return None, None
    return int(cx), int(cy)


def is_ultra_close_tennis_candidate(target):
    if target is None:
        return False

    raw_dist_cm = target_raw_distance_cm(target)
    if (raw_dist_cm is not None) and (raw_dist_cm <= CLOSE_REFINE_DUPLICATE_DISTANCE_CM):
        return True
    if int(target.get("close_fit", 0)):
        return True
    return is_close_range_candidate(target)


def measure_source_rank(target):
    src = str(target.get("measure_src", ""))
    if src == "hough_close":
        return 5
    if src == "mix_close":
        return 4
    if src == "hough":
        return 4
    if src == "mix":
        return 3
    if src == "lab_close":
        return 2
    if src == "lab":
        return 1
    if src == "bbox_close":
        return 1
    return 0


def close_refined_tennis_candidates_should_merge(a, b):
    if a is None or b is None:
        return False
    if not (is_ultra_close_tennis_candidate(a) or is_ultra_close_tennis_candidate(b)):
        return False

    ax, ay = target_measure_center(a)
    bx, by = target_measure_center(b)
    if ax is None or ay is None or bx is None or by is None:
        return False

    dia_a = target_measure_diameter(a)
    dia_b = target_measure_diameter(b)
    if dia_a is None or dia_b is None:
        return False

    center_dx = abs(ax - bx)
    center_dy = abs(ay - by)
    min_d = max(1, min(dia_a, dia_b))
    max_d = max(dia_a, dia_b)
    center_gate = max(
        CLOSE_REFINE_DUPLICATE_CENTER_MIN,
        (min_d * CLOSE_REFINE_DUPLICATE_CENTER_RATIO_PCT) // 100,
    )
    loose_gate = max(center_gate + 4, (max_d * CLOSE_REFINE_DUPLICATE_CENTER_RATIO_PCT) // 100)
    ratio_pct = (max_d * 100) // max(1, min_d)

    rect_a = (a["x"], a["y"], a["w"], a["h"])
    rect_b = (b["x"], b["y"], b["w"], b["h"])
    contains = rect_contains_point(rect_a, bx, by, CLOSE_REFINE_DUPLICATE_CENTER_PAD) or rect_contains_point(
        rect_b, ax, ay, CLOSE_REFINE_DUPLICATE_CENTER_PAD
    )
    overlap = rect_iou(rect_a, rect_b)

    dist_close = True
    dist_a = target_raw_distance_cm(a)
    dist_b = target_raw_distance_cm(b)
    if dist_a is not None and dist_b is not None:
        dist_close = abs(dist_a - dist_b) <= CLOSE_REFINE_DUPLICATE_DIST_DIFF_CM

    similar_size = ratio_pct <= CLOSE_REFINE_DUPLICATE_DIAMETER_RATIO_MAX_PCT
    if (center_dx <= center_gate) and (center_dy <= center_gate) and similar_size:
        return True
    if contains and (center_dx <= loose_gate) and (center_dy <= loose_gate):
        return dist_close or similar_size or (overlap > 0.0)
    if (overlap >= FOMO_DUPLICATE_CLOSE_IOU) and (center_dx <= loose_gate) and (center_dy <= loose_gate):
        return dist_close and similar_size
    return False


def select_close_refined_tennis_representative(group):
    if len(group) == 1:
        merged = group[0].copy()
        merged["merge_count"] = int(group[0].get("merge_count", 1))
        return merged

    best = None
    best_key = None
    merge_count = 0

    for cand in group:
        merge_count += int(cand.get("merge_count", 1))
        raw_dist_cm = target_raw_distance_cm(cand)
        if raw_dist_cm is None:
            raw_dist_cm = 9999.0
        diameter = target_measure_diameter(cand)
        if diameter is None:
            diameter = 0
        key = (
            -int(cand.get("cue_conf", 0)),
            -measure_source_rank(cand),
            raw_dist_cm,
            -diameter,
            -(cand["w"] * cand["h"]),
            -int(cand.get("score", 0) * 1000),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = cand

    merged = best.copy()
    merged["merge_count"] = max(1, merge_count)
    return merged


def dedupe_close_refined_tennis_candidates(candidates):
    if len(candidates) <= 1:
        merged = []
        for cand in candidates:
            item = cand.copy()
            item["merge_count"] = int(cand.get("merge_count", 1))
            merged.append(item)
        return merged

    used = [False] * len(candidates)
    merged = []

    for i in range(len(candidates)):
        if used[i]:
            continue

        group = [candidates[i]]
        used[i] = True
        expanded = True
        while expanded:
            expanded = False
            for j in range(len(candidates)):
                if used[j]:
                    continue
                for saved in group:
                    if close_refined_tennis_candidates_should_merge(saved, candidates[j]):
                        group.append(candidates[j])
                        used[j] = True
                        expanded = True
                        break

        merged.append(select_close_refined_tennis_representative(group))

    return merged


def tennis_candidates_should_merge(a, b):
    rect_a = (a["x"], a["y"], a["w"], a["h"])
    rect_b = (b["x"], b["y"], b["w"], b["h"])
    iou = rect_iou(rect_a, rect_b)
    center_dx = abs(a["cx"] - b["cx"])
    center_dy = abs(a["cy"] - b["cy"])
    max_size = max(max(a["w"], a["h"]), max(b["w"], b["h"]))
    close_pair = (
        is_close_range_candidate(a)
        or is_close_range_candidate(b)
        or (max(a["w"], a["h"]) >= FOMO_DUPLICATE_CLOSE_SIZE_TH)
        or (max(b["w"], b["h"]) >= FOMO_DUPLICATE_CLOSE_SIZE_TH)
    )

    contains = rect_contains_point(rect_a, b["cx"], b["cy"], FOMO_DUPLICATE_CENTER_PAD) or rect_contains_point(
        rect_b, a["cx"], a["cy"], FOMO_DUPLICATE_CENTER_PAD
    )
    center_close = center_dx <= (max_size + FOMO_DUPLICATE_CENTER_PAD) and center_dy <= (
        max_size + FOMO_DUPLICATE_CENTER_PAD
    )

    if close_pair:
        if contains and center_close:
            return True
        if (iou >= FOMO_DUPLICATE_CLOSE_IOU) and center_close:
            return True
        if ((center_dx + center_dy) <= max(18, max_size)) and ((iou > 0.02) or contains):
            return True
        return False

    if (iou >= FOMO_DUPLICATE_GENERAL_IOU) and center_close:
        return True
    return False


def merge_tennis_candidate_group(group, img_w, img_h):
    if len(group) == 1:
        merged = group[0].copy()
        merged["merge_count"] = int(group[0].get("merge_count", 1))
        return merged

    left = group[0]["x"]
    top = group[0]["y"]
    right = group[0]["x"] + group[0]["w"]
    bottom = group[0]["y"] + group[0]["h"]
    weighted_cx = 0
    weighted_cy = 0
    weight_sum = 0
    best_score = 0.0
    nearest_dist = None
    merge_count = 0

    for cand in group:
        left = min(left, cand["x"])
        top = min(top, cand["y"])
        right = max(right, cand["x"] + cand["w"])
        bottom = max(bottom, cand["y"] + cand["h"])
        weight = max(1, int(cand["score"] * 1000)) + max(1, cand["w"] * cand["h"])
        weighted_cx += cand["cx"] * weight
        weighted_cy += cand["cy"] * weight
        weight_sum += weight
        if cand["score"] > best_score:
            best_score = cand["score"]
        cand_dist = cand.get("dist_cm")
        if cand_dist is not None:
            if nearest_dist is None or cand_dist < nearest_dist:
                nearest_dist = cand_dist
        merge_count += int(cand.get("merge_count", 1))

    merged_cx = weighted_cx // max(1, weight_sum)
    merged_cy = weighted_cy // max(1, weight_sum)
    merged_w = max(1, right - left)
    merged_h = max(1, bottom - top)
    merged_cx = clamp(merged_cx, left, right - 1)
    merged_cy = clamp(merged_cy, top, bottom - 1)
    row, col = get_grid_position(merged_cx, merged_cy, img_w, img_h, GRID_ROWS, GRID_COLS)

    return {
        "x": left,
        "y": top,
        "cx": merged_cx,
        "cy": merged_cy,
        "radius": max(4, min(55, estimate_ball_radius(merged_w, merged_h))),
        "w": merged_w,
        "h": merged_h,
        "score": best_score,
        "dist_cm": nearest_dist,
        "row": row,
        "col": col,
        "kind": "tennis",
        "merge_count": max(1, merge_count),
    }


def merge_duplicate_tennis_candidates(candidates, img_w, img_h):
    if len(candidates) <= 1:
        merged = []
        for cand in candidates:
            item = cand.copy()
            item["merge_count"] = int(cand.get("merge_count", 1))
            merged.append(item)
        return merged

    used = [False] * len(candidates)
    merged = []

    for i in range(len(candidates)):
        if used[i]:
            continue

        group = [candidates[i]]
        used[i] = True
        expanded = True
        while expanded:
            expanded = False
            for j in range(len(candidates)):
                if used[j]:
                    continue
                for saved in group:
                    if tennis_candidates_should_merge(saved, candidates[j]):
                        group.append(candidates[j])
                        used[j] = True
                        expanded = True
                        break

        merged.append(merge_tennis_candidate_group(group, img_w, img_h))

    return merged


def correct_tennis_distance(distance_cm, pixel_diameter):
    # 距离修正仅依赖像素直径，不使用拟合半径作为参考。
    if pixel_diameter >= 46:
        factor = 1.10
    elif pixel_diameter >= 34:
        factor = 1.15
    elif pixel_diameter >= 24:
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


def build_close_tennis_color_threshold(img, x, y, w, h):
    seed_w = max(8, (w * 3) // 10)
    seed_h = max(8, (h * 3) // 10)
    seed_x = clamp(x + (w - seed_w) // 2, 0, img.width() - 1)
    seed_y = clamp(y + (h - seed_h) // 2, 0, img.height() - 1)
    seed_w = clamp(seed_w, 1, img.width() - seed_x)
    seed_h = clamp(seed_h, 1, img.height() - seed_y)
    seed_roi = (seed_x, seed_y, seed_w, seed_h)

    hist = img.get_histogram(roi=seed_roi)
    lo = hist.get_percentile(CLOSE_COLOR_PERCENTILE_LO)
    hi = hist.get_percentile(CLOSE_COLOR_PERCENTILE_HI)
    stats = img.get_statistics(roi=seed_roi)

    l_lo = int(clamp(min(lo.l_value(), stats.l_mean()) - CLOSE_COLOR_L_MARGIN, 0, 100))
    l_hi = int(clamp(max(hi.l_value(), stats.l_mean()) + CLOSE_COLOR_L_MARGIN, 0, 100))
    a_lo = int(clamp(min(lo.a_value(), stats.a_mean()) - CLOSE_COLOR_A_MARGIN, -128, 127))
    a_hi = int(clamp(max(hi.a_value(), stats.a_mean()) + CLOSE_COLOR_A_MARGIN, -128, 127))
    b_lo = int(clamp(min(lo.b_value(), stats.b_mean()) - CLOSE_COLOR_B_MARGIN, -128, 127))
    b_hi = int(clamp(max(hi.b_value(), stats.b_mean()) + CLOSE_COLOR_B_MARGIN, -128, 127))

    a_hi = min(a_hi, TENNIS_CENTER_GREEN_A_MAX + 18)
    b_lo = max(b_lo, TENNIS_CENTER_GREEN_B_MIN - 8)
    return (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi)


def estimate_color_blob(img, roi, ref_cx, ref_cy, bbox_d, close_mode=False):
    rx, ry, rw, rh = roi
    if close_mode:
        thr = build_close_tennis_color_threshold(
            img, ref_cx - (rw // 6), ref_cy - (rh // 6), max(8, rw // 3), max(8, rh // 3)
        )
    else:
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
        return None, ref_cx, ref_cy, None, None

    best = None
    best_score = None
    if close_mode:
        max_blob_area = max(36, (rw * rh * CLOSE_COLOR_BLOB_AREA_RATIO_PCT) // 100)
    else:
        max_blob_area = max(36, bbox_d * bbox_d * MAX_COLOR_BLOB_AREA_MULT)
    for b in blobs:
        if b.pixels() > max_blob_area:
            continue

        roundness = b.roundness()
        density = b.density()
        if close_mode and (roundness < CLOSE_COLOR_ROUNDNESS_MIN) and (density < CLOSE_COLOR_DENSITY_MIN):
            continue

        dx = b.cx() - ref_cx
        dy = b.cy() - ref_cy
        inside_ref = (b.x() <= ref_cx <= (b.x() + b.w())) and (b.y() <= ref_cy <= (b.y() + b.h()))

        long_side = max(b.w(), b.h())
        short_side = max(1, min(b.w(), b.h()))
        ratio = (long_side * 100) // short_side
        roundness_penalty = int((1.0 - roundness) * (180 if close_mode else 80))
        density_penalty = int(abs(density - CIRCLE_DENSITY_REF) * (220 if close_mode else 90))
        shape_penalty = abs(ratio - 100) + roundness_penalty + density_penalty

        dist_cost = abs(dx) + abs(dy)
        if close_mode:
            dist_cost *= 2
            size_gain = b.pixels() // 5
            center_bonus = 80 if inside_ref else 0
        else:
            dist_cost *= 3
            size_gain = b.pixels() // 6
            center_bonus = 60 if inside_ref else 0

        score = dist_cost + shape_penalty - size_gain - center_bonus
        if (best_score is None) or (score < best_score):
            best_score = score
            best = b

    if best is None:
        return None, ref_cx, ref_cy, None, None

    eq_d = int(math.sqrt((4.0 * best.pixels()) / math.pi) + 0.5)
    blob_d = max(best.w(), best.h())
    if close_mode:
        short_side = min(best.w(), best.h())
        avg_side = (best.w() + best.h()) // 2
        roundness = best.roundness()
        density = best.density()
        if (roundness >= 0.55) and (density >= 0.60):
            color_d = int(((eq_d * 7) + (avg_side * 3) + 5) // 10)
        else:
            color_d = int(((eq_d * 4) + (short_side * 6) + 5) // 10)
    else:
        color_d = max(eq_d, blob_d)
    return color_d, best.cx(), best.cy(), best.roundness(), best.density()


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


def maybe_boost_mid_ball_hough_diameter(diameter, bbox_d, close_mode=False):
    if diameter is None or close_mode:
        return diameter
    if bbox_d < MID_BALL_HOUGH_BBOX_MIN or bbox_d > MID_BALL_HOUGH_BBOX_MAX:
        return diameter
    return int(((diameter * MID_BALL_HOUGH_DIAMETER_BOOST_PCT) + 50) // 100)


def estimate_hough_circle(
    img,
    roi,
    ref_cx,
    ref_cy,
    r_guess,
    edge_strength,
    glare_ratio_pct,
    close_mode=False,
    bbox_d=0,
):
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        hough_threshold = 3300 if edge_strength >= 24 else 2900
    else:
        hough_threshold = 3000 if edge_strength >= 24 else 2650

    if close_mode:
        hough_threshold = max(2200, hough_threshold - 150)

    r_guess = max(3, r_guess)
    if close_mode:
        r_min = max(4, (r_guess * 7) // 10)
        r_max = min(CLOSE_HOUGH_R_MAX, (r_guess * 13) // 10)
        x_margin = CLOSE_HOUGH_X_MARGIN
        y_margin = CLOSE_HOUGH_Y_MARGIN
        r_margin = CLOSE_HOUGH_R_MARGIN
        x_stride = 2
        y_stride = 2
    else:
        r_min = max(3, (r_guess * 7) // 10)
        r_max = min(105, (r_guess * 13) // 10)
        x_margin = 6
        y_margin = 6
        r_margin = 6
        x_stride = 2
        y_stride = 1

    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        r_max = max(r_min, (r_max * 9) // 10)

    if (not close_mode) and (bbox_d > 0) and (bbox_d <= FAR_BALL_SIZE_TH):
        # 远距离小球时，限制霍夫半径，避免高亮区域被拟合成超大光圈。
        far_r_min = max(2, (bbox_d * FAR_BALL_DIAMETER_RATIO_MIN_PCT) // 200)
        far_r_max = max(
            far_r_min,
            min(FAR_BALL_HOUGH_R_MAX, (bbox_d * FAR_BALL_DIAMETER_RATIO_MAX_PCT) // 200),
        )
        r_min = max(r_min, far_r_min)
        r_max = min(r_max, far_r_max)

    if r_max < r_min:
        return None, ref_cx, ref_cy

    circles = img.find_circles(
        roi=roi,
        threshold=hough_threshold,
        x_margin=x_margin,
        y_margin=y_margin,
        r_margin=r_margin,
        r_min=r_min,
        r_max=r_max,
        x_stride=x_stride,
        y_stride=y_stride,
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
        if close_mode:
            score = (center_cost * 3) + (radius_cost * 2)
        else:
            score = (center_cost * 2) + radius_cost - (c.r() // 4)
        if (best_score is None) or (score < best_score):
            best_score = score
            best = c

    if best is None:
        return None, ref_cx, ref_cy

    return best.r() * 2, best.x(), best.y()


def estimate_tennis_diameter(img, x, y, w, h, allow_hough=True, close_mode=False):
    ball_size = max(w, h)
    if close_mode:
        pad = max(CLOSE_BALL_ROI_PAD_MIN, (ball_size * 5) // 4)
    elif ball_size < ROI_NEAR_SWITCH:
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

    color_d, color_cx, color_cy, color_roundness, color_density = estimate_color_blob(
        img, roi, cx, cy, bbox_d, close_mode=close_mode
    )

    hough_d = None
    hough_cx = cx
    hough_cy = cy
    use_hough = allow_hough or close_mode
    if use_hough:
        edge_strength = estimate_edge_strength(img, roi)
        glare_ratio_pct = estimate_glare_ratio_pct(img, roi)
        if color_d is not None:
            hough_guess = max(color_d // 2, bbox_d // 2)
        else:
            hough_guess = max(5, (bbox_d * 8) // 10)
        hough_d, hough_cx, hough_cy = estimate_hough_circle(
            img,
            roi,
            color_cx,
            color_cy,
            hough_guess,
            edge_strength,
            glare_ratio_pct,
            close_mode=close_mode,
            bbox_d=bbox_d,
        )
        hough_d = maybe_boost_mid_ball_hough_diameter(hough_d, bbox_d, close_mode=close_mode)
    else:
        glare_ratio_pct = 0

    cue_conf = 0
    measure_src = "bbox_close" if close_mode else "bbox"
    if (hough_d is not None) and (color_d is not None):
        delta = abs(hough_d - color_d)
        if close_mode:
            if delta <= 10:
                d = ((hough_d * 8) + (color_d * 2)) // 10
                cue_conf = 2
            else:
                d = hough_d
                cue_conf = 2
            measure_src = "mix_close" if delta <= 10 else "hough_close"
        elif delta <= 10:
            d = ((hough_d * 5) + (color_d * 5)) // 10
            cue_conf = 2
            measure_src = "mix"
        else:
            d = max(hough_d, color_d)
            cue_conf = 1
            measure_src = "hough"
    elif hough_d is not None:
        d = hough_d
        cue_conf = 2 if close_mode else 1
        measure_src = "hough_close" if close_mode else "hough"
    elif color_d is not None:
        d = color_d
        cue_conf = 1
        measure_src = "lab_close" if close_mode else "lab"
    else:
        d = int((bbox_d * 13) // 10)
        cue_conf = 0

    d = clamp(d, 4, 210)
    d = clamp_diameter_by_bbox(d, bbox_d, close_mode=close_mode)

    if (ball_size <= SMALL_BALL_SIZE_TH) and (glare_ratio_pct >= GLARE_RATIO_TH_PCT):
        glare_cap = max(8, (bbox_d * GLARE_DIAMETER_CAP_PCT) // 100)
        if d > glare_cap:
            d = glare_cap
        cue_conf = min(cue_conf, 1)
        if hough_d is None:
            measure_src = "lab_close" if color_d is not None and close_mode else measure_src
            measure_src = "lab" if color_d is not None and not close_mode else measure_src

    measure_cx = cx
    measure_cy = cy
    if hough_d is not None:
        measure_cx = hough_cx
        measure_cy = hough_cy
    elif color_d is not None:
        measure_cx = color_cx
        measure_cy = color_cy

    return d, cue_conf, measure_src, measure_cx, measure_cy


def estimate_ball_radius(w, h):
    mx = max(w, h)
    r = ((mx * 13) + 10) // 20
    area = w * h
    if area >= 900:
        r += 3
    elif area >= 400:
        r += 2
    return clamp(r, 4, 55)


def fuse_tennis_radius(w, h, detected_diameter, close_mode=False):
    circle_r = max(4, detected_diameter // 2)
    long_side = max(w, h)
    if close_mode:
        r = circle_r
        scale = CLOSE_DRAW_RADIUS_SCALE
    elif long_side >= 20:
        bbox_r = estimate_ball_radius(w, h)
        r = max(circle_r, bbox_r)
        scale = DRAW_RADIUS_SCALE
    else:
        bbox_r = estimate_ball_radius(w, h)
        r = max(circle_r, (bbox_r * 9) // 10)
        scale = DRAW_RADIUS_SCALE
    r = int((r * scale) + 0.5)
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


def high_rank_window_mean(values, rank_window=SCAN_LOCK_RADIUS_RANK_WINDOW):
    if not values:
        return None

    data = [int(v) for v in values if v is not None]
    if not data:
        return None

    data.sort(reverse=True)
    window = min(rank_window, len(data))
    if len(data) >= 5:
        start = 2
    elif len(data) == 4:
        start = 1
    else:
        start = 0
    end = min(len(data), start + window)
    core = data[start:end]
    if not core:
        core = data[:window]
    return int((sum(core) / len(core)) + 0.5)


def scan_sample_radius(target):
    if target is None:
        return None

    radius = target.get("refined_radius")
    if radius is None:
        radius = target.get("radius")
    if (radius is None) and ("w" in target) and ("h" in target):
        radius = estimate_ball_radius(target["w"], target["h"])
    if radius is None:
        return None
    return int(radius)


def scan_sample_diameter(target):
    if target is None:
        return None

    diameter = target.get("refined_diameter")
    if diameter is None:
        radius = scan_sample_radius(target)
        if radius is not None:
            diameter = radius * 2
        elif ("w" in target) and ("h" in target):
            diameter = max(target["w"], target["h"])
    if diameter is None:
        return None
    return int(diameter)


def scan_lock_radius(target):
    radius = target.get("scan_lock_radius")
    if radius is None:
        radius = scan_sample_radius(target)
    return int(radius) if radius is not None else 0


def scan_lock_distance(target):
    distance = target.get("scan_lock_dist_cm")
    if distance is None:
        distance = target.get("dist_cm")
    if distance is None:
        return 9999.0
    return float(distance)


def refresh_scan_target_measure(target, sample_target=None):
    sample = sample_target if sample_target is not None else target
    sample_radius = scan_sample_radius(sample)
    sample_diameter = scan_sample_diameter(sample)

    radius_history = append_measure_window(target.get("scan_radius_history"), sample_radius)
    diameter_history = append_measure_window(target.get("scan_diameter_history"), sample_diameter)
    target["scan_radius_history"] = radius_history
    target["scan_diameter_history"] = diameter_history

    lock_radius = high_rank_window_mean(radius_history)
    lock_diameter = high_rank_window_mean(diameter_history)
    if lock_radius is None:
        lock_radius = sample_radius
    if lock_diameter is None:
        lock_diameter = sample_diameter
    if (lock_diameter is None) and (lock_radius is not None):
        lock_diameter = lock_radius * 2

    target["scan_lock_radius"] = lock_radius
    target["scan_lock_diameter"] = lock_diameter

    if lock_radius is not None:
        target["radius"] = lock_radius
        target["refined_radius"] = lock_radius
    if lock_diameter is not None:
        target["refined_diameter"] = lock_diameter

    lock_dist_cm = None
    if lock_diameter is not None:
        lock_dist_cm = estimate_corrected_distance_cm(lock_diameter)
    if lock_dist_cm is None:
        lock_dist_cm = target.get("dist_cm")

    target["scan_lock_dist_cm"] = lock_dist_cm
    target["dist_cm"] = lock_dist_cm
    target["refined_dist_cm"] = lock_dist_cm
    return target


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


def reset_pick_confirm_window():
    global pick_confirm_total_frames, pick_confirm_seen_frames

    pick_confirm_total_frames = 0
    pick_confirm_seen_frames = 0


def note_pick_confirm_frame(seen):
    global pick_confirm_total_frames, pick_confirm_seen_frames

    pick_confirm_total_frames += 1
    if seen:
        pick_confirm_seen_frames += 1


def pick_confirm_ready_to_lock():
    if pick_confirm_total_frames <= 0:
        return False
    if pick_confirm_total_frames > PICK_EARLY_LOCK_WINDOW_FRAMES:
        return False
    return pick_confirm_seen_frames >= PICK_EARLY_LOCK_SEEN_FRAMES


def reset_pick_scan_fail_rounds():
    global pick_scan_fail_rounds

    pick_scan_fail_rounds = 0


def reset_play_hit_state():
    global play_presence_window, play_hit_latched, play_ready_frames

    play_presence_window = []
    play_hit_latched = False
    play_ready_frames = 0


def reset_play_lock_state():
    global player_lock_window, player_locked

    player_lock_window = []
    player_locked = False


def update_player_lock_state(player_present):
    global player_lock_window, player_locked

    player_lock_window.append(1 if player_present else 0)
    if len(player_lock_window) > PLAYER_LOCK_WINDOW_FRAMES:
        player_lock_window.pop(0)

    if player_present:
        player_locked = True
        return True

    if not player_locked:
        return False

    if len(player_lock_window) < PLAYER_LOCK_WINDOW_FRAMES:
        return True

    player_hits = 0
    for hit in player_lock_window:
        player_hits += hit

    if player_hits < PLAYER_LOCK_MIN_HIT_FRAMES:
        player_locked = False
        return False
    return True


def note_play_presence(player_present, racket_present):
    global play_presence_window, play_hit_latched, play_ready_frames

    flags = 0
    if player_present:
        flags |= 1
    if racket_present:
        flags |= 2

    play_presence_window.append(flags)
    if len(play_presence_window) > PLAY_HIT_WINDOW_FRAMES:
        play_presence_window.pop(0)

    player_hits = 0
    racket_hits = 0
    for state in play_presence_window:
        if state & 1:
            player_hits += 1
        if state & 2:
            racket_hits += 1

    play_ready_frames = min(player_hits, racket_hits)
    ready = (
        player_hits >= PLAY_HIT_MIN_PLAYER_FRAMES
        and racket_hits >= PLAY_HIT_MIN_RACKET_FRAMES
    )
    if ready and (not play_hit_latched):
        play_hit_latched = True
        return True
    if not ready:
        play_hit_latched = False
    return False


def handle_pick_scan_round_failed(racket_present, racket_target, command_event):
    global pick_scan_fail_rounds

    pick_scan_fail_rounds += 1
    if pick_scan_fail_rounds >= PICK_SCAN_FAIL_ROUNDS_TO_PLAY:
        enter_play_mode()
        return None, "PLAY", "TRACK_P", False, None, command_event

    begin_scan_round()
    return None, "SEEK", "SCAN", racket_present, racket_target, command_event


def clear_pick_unlock_events():
    global picker_uart_done_pending, host_track_unlock_pending

    picker_uart_done_pending = 0
    host_track_unlock_pending = 0


def is_racket_linked_to_player(player_target, racket_target):
    if player_target is None or racket_target is None:
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

    if player_target is not None:
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
        if player_target is not None:
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


def send_runtime_packets(mode_name, state_name, target):
    if uart is None:
        return
    if mode_name != "SEEK" or state_name != "TRACK":
        return
    if (frame_index % UART_SEND_INTERVAL_FRAMES) != 0:
        return
    if (not is_live_target(target)) or target.get("kind") != "tennis":
        return

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
        if tracked_tennis["miss"] > TENNIS_TRACK_MAX_MISS:
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
    tracked_tennis["merge_count"] = int(best.get("merge_count", 1))
    if "measure_src" in best:
        tracked_tennis["measure_src"] = best["measure_src"]
    if "cue_conf" in best:
        tracked_tennis["cue_conf"] = best["cue_conf"]
    if "close_fit" in best:
        tracked_tennis["close_fit"] = best["close_fit"]
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
        if filtered_diameter is not None:
            tracked_tennis["dist_cm"] = estimate_corrected_distance_cm(filtered_diameter)
            tracked_tennis["refined_dist_cm"] = tracked_tennis["dist_cm"]
    tracked_tennis["miss"] = 0
    return tracked_tennis


def choose_player_target(candidates):
    global tracked_player

    if not candidates:
        if tracked_player is not None:
            tracked_player["miss"] += 1
            if tracked_player["miss"] > PLAYER_TRACK_MAX_MISS:
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


def draw_active_target(img, target, mode_name, state_name):
    if target is None:
        img.draw_string(2, 74, "target:search", color=YELLOW, mono_space=False)
        return

    target_missing = target.get("miss", 0) > 0
    cx = clamp(target["cx"], 0, img.width() - 1)
    cy = clamp(target["cy"], 0, img.height() - 1)
    radius = clamp(target["radius"], 4, 55)
    kind = target.get("kind", "target").lower()
    draw_cx = cx
    draw_cy = cy

    if kind == "tennis":
        target_color = GREEN
    elif kind == "player":
        target_color = BLUE
    elif kind == "racket":
        target_color = RED
    else:
        target_color = YELLOW

    show_lock_circle = True
    if (mode_name == "SEEK") and (kind == "tennis") and (not target_missing):
        if state_name == "RETURN":
            show_lock_circle = False
        elif state_name == "CONFIRM":
            show_lock_circle = target_is_aligned(target, img)

    if not show_lock_circle:
        img.draw_string(2, 74, "target:align", color=YELLOW, mono_space=False)
        dx, dy = target_center_error(target, img)
        img.draw_string(2, 92, "dx:%d dy:%d" % (dx, dy), color=WHITE, mono_space=False)
        if target["dist_cm"] is not None:
            img.draw_string(2, 110, "dist:%.1fcm" % target["dist_cm"], color=WHITE, mono_space=False)
        return

    if (kind == "player") and target_missing and (mode_name == "PLAY"):
        draw_cx = img.width() // 2
        draw_cy = img.height() // 2

    if (kind == "racket") and all(k in target for k in ("x", "y", "w", "h")):
        img.draw_rectangle((target["x"], target["y"], target["w"], target["h"]), color=target_color, thickness=2)
    else:
        img.draw_circle((draw_cx, draw_cy, radius + 3), color=target_color)
    img.draw_cross(draw_cx, draw_cy, color=WHITE, size=10, thickness=2)
    if (draw_cx != (img.width() // 2)) or (draw_cy != (img.height() // 2)):
        img.draw_line((img.width() // 2, img.height() // 2, draw_cx, draw_cy), color=target_color)

    kind_label = target.get("kind", "target").upper()
    status_text = "%s HOLD" % kind_label[:6] if target_missing else "%s LOCK" % kind_label[:6]
    status_color = target_color if ((kind == "player") and target_missing) else (YELLOW if target_missing else target_color)
    img.draw_string(2, 74, status_text, color=status_color, mono_space=False)
    img.draw_string(
        2,
        92,
        "dx:%d dy:%d" % (draw_cx - (img.width() // 2), draw_cy - (img.height() // 2)),
        color=WHITE,
        mono_space=False,
    )
    if target["dist_cm"] is not None:
        img.draw_string(2, 110, "dist:%.1fcm" % target["dist_cm"], color=WHITE, mono_space=False)
    else:
        img.draw_string(
            2,
            110,
            "grid:(%d,%d)" % (int(target.get("row", -1)), int(target.get("col", -1))),
            color=WHITE,
            mono_space=False,
        )
    if target_missing:
        img.draw_string(2, 128, "miss:%d" % int(target.get("miss", 0)), color=YELLOW, mono_space=False)
    elif "measure_src" in target:
        img.draw_string(2, 128, "measure:%s" % target["measure_src"], color=WHITE, mono_space=False)


def draw_racket_target(img, target):
    if target is None:
        return
    if not all(k in target for k in ("x", "y", "w", "h", "cx", "cy")):
        return

    x = int(clamp(target["x"], 0, img.width() - 1))
    y = int(clamp(target["y"], 0, img.height() - 1))
    w = int(clamp(target["w"], 1, img.width() - x))
    h = int(clamp(target["h"], 1, img.height() - y))
    cx = int(clamp(target["cx"], 0, img.width() - 1))
    cy = int(clamp(target["cy"], 0, img.height() - 1))
    label_y = y - 14 if y >= 14 else y + h + 2

    img.draw_rectangle((x, y, w, h), color=RED, thickness=2)
    img.draw_cross(cx, cy, color=RED, size=8, thickness=2)
    img.draw_string(x, label_y, "RACKET", color=RED, mono_space=False)


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

    pan_output = (pan_pid.get_pid(pan_error, 1) / 2) * SERVO_SPEED_SCALE
    tilt_output = tilt_pid.get_pid(tilt_error, 1) * SERVO_SPEED_SCALE

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

    tilt_output = tilt_pid.get_pid(tilt_error, 1) * SERVO_SPEED_SCALE
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
    distance = scan_lock_distance(target)
    radius = scan_lock_radius(target)
    area = target["w"] * target["h"]
    return (distance, -radius, -area, -int(target.get("score", 0) * 1000))


def scan_observation_key(target):
    center_dx = abs(target["cx"] - 160)
    center_dy = abs(target["cy"] - 120)
    radius = scan_sample_radius(target)
    if radius is None:
        radius = 0
    return (-radius, (center_dx + center_dy), -int(target.get("score", 0) * 1000))


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


def remember_scan_target(target, img=None, observed_pan=None, observed_tilt=None):
    global best_scan_tennis

    if target is None:
        return
    if target.get("miss", 0) > 0:
        return
    if target.get("dist_cm") is None:
        return

    scan_target = target.copy()
    if observed_pan is None:
        observed_pan = pan_angle
    if observed_tilt is None:
        observed_tilt = tilt_angle
    scan_target["servo_pan"] = observed_pan
    scan_target["servo_tilt"] = observed_tilt
    if img is not None:
        update_target_lock_pan(scan_target, img, pan_value=observed_pan)

    matched_index = -1
    for idx in range(len(scan_ranked_tennis)):
        if is_same_scan_target(scan_ranked_tennis[idx], scan_target):
            matched_index = idx
            break

    if matched_index >= 0:
        saved_target = scan_ranked_tennis[matched_index]
        prev_id = int(saved_target.get("id", 0))
        merged_target = saved_target.copy()
        if scan_observation_key(scan_target) < scan_observation_key(saved_target):
            merged_target.update(scan_target)
        refresh_scan_target_measure(merged_target, scan_target)
        if "lock_pan" in saved_target:
            merged_target["lock_pan"] = saved_target["lock_pan"]
            if "lock_pan_ms" in saved_target:
                merged_target["lock_pan_ms"] = saved_target["lock_pan_ms"]
        if "lock_pan" in scan_target:
            if scan_target.get("lock_pan_ms", 0) >= merged_target.get("lock_pan_ms", 0):
                merged_target["lock_pan"] = scan_target["lock_pan"]
                merged_target["lock_pan_ms"] = scan_target.get("lock_pan_ms", 0)
        if prev_id > 0:
            ensure_target_id(merged_target, prev_id)
        scan_ranked_tennis[matched_index] = merged_target
    else:
        refresh_scan_target_measure(scan_target, scan_target)
        scan_ranked_tennis.append(scan_target)

    scan_ranked_tennis.sort(key=scan_rank_key)
    select_scan_candidate(0)


def begin_scan_round():
    global pick_substate, scan_seek_left, scan_direction
    global nearest_switch_count, best_scan_tennis, tracked_tennis
    global scan_ranked_tennis, scan_candidate_index
    global scan_tilt_reset_pending, pick_track_lost_count

    pick_substate = PICK_SCAN
    scan_seek_left = True
    scan_tilt_reset_pending = True
    scan_direction = 1
    nearest_switch_count = 0
    best_scan_tennis = None
    scan_ranked_tennis = []
    scan_candidate_index = 0
    tracked_tennis = None
    reset_pick_confirm_window()
    pick_track_lost_count = 0


def update_global_scan(img, tennis_candidates):
    global scan_seek_left, pan_angle, tilt_angle, scan_tilt_reset_pending

    observed_pan = pan_angle
    observed_tilt = TILT_INIT_ANGLE

    if not ENABLE_SERVOS:
        for cand in tennis_candidates:
            remember_scan_target(cand, img, observed_pan, observed_tilt)
        return len(scan_ranked_tennis) > 0

    if scan_tilt_reset_pending:
        move_tilt_toward(TILT_INIT_ANGLE, RETURN_TILT_STEP)
        if abs(tilt_angle - TILT_INIT_ANGLE) <= SEEK_TILT_RESET_MARGIN:
            tilt_angle = TILT_INIT_ANGLE
            scan_tilt_reset_pending = False
        else:
            return False

    if scan_seek_left:
        move_pan_toward(pan_angle_limit[0], SCAN_PAN_STEP)
        tilt_angle = TILT_INIT_ANGLE
        if abs(pan_angle - pan_angle_limit[0]) <= SCAN_ALIGN_MARGIN:
            scan_seek_left = False
            return False
        return False

    apply_pan_delta(SCAN_PAN_STEP, SCAN_PAN_STEP)
    tilt_angle = TILT_INIT_ANGLE

    for cand in tennis_candidates:
        remember_scan_target(cand, img, observed_pan, observed_tilt)

    if pan_angle >= (pan_angle_limit[1] - SCAN_ALIGN_MARGIN):
        pan_angle = pan_angle_limit[1]
        return True
    return False


def update_return_to_saved_target():
    if best_scan_tennis is None:
        return True

    target_pan = best_scan_tennis.get("lock_pan", best_scan_tennis.get("servo_pan", PAN_INIT_ANGLE))
    move_pan_toward(target_pan, RETURN_PAN_STEP)
    move_tilt_toward(best_scan_tennis.get("servo_tilt", TILT_INIT_ANGLE), RETURN_TILT_STEP)

    pan_ok = abs(pan_angle - target_pan) <= RETURN_LOCK_MARGIN
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

        cand_radius = scan_sample_radius(cand)
        if cand_radius is None:
            cand_radius = 0
        key = (d2 + dist_penalty, -cand_radius, -(cand["w"] * cand["h"]), -int(cand["score"] * 1000))
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
    if "lock_pan" in saved_target:
        matched["lock_pan"] = saved_target["lock_pan"]
    if "lock_pan_ms" in saved_target:
        matched["lock_pan_ms"] = saved_target["lock_pan_ms"]
    return matched


def choose_local_nearest_tennis(saved_target, candidates):
    if saved_target is None or not candidates:
        return None

    local = []
    gate = max(
        LOCAL_NEAREST_GATE_MIN,
        saved_target.get("radius", 12) * LOCAL_NEAREST_GATE_RADIUS_SCALE,
    )
    gate2 = gate * gate

    for cand in candidates:
        dx = cand["cx"] - saved_target["cx"]
        dy = cand["cy"] - saved_target["cy"]
        d2 = (dx * dx) + (dy * dy)
        if d2 <= gate2:
            local.append(cand)

    if not local:
        return None

    return choose_nearest_tennis(local)


def select_next_scan_candidate():
    global tracked_tennis

    tracked_tennis = None
    reset_pick_confirm_window()
    return select_scan_candidate(scan_candidate_index + 1)


def choose_nearest_tennis(candidates):
    best = None
    best_key = None
    for cand in candidates:
        distance = cand["dist_cm"] if cand["dist_cm"] is not None else 9999.0
        radius = scan_sample_radius(cand)
        if radius is None:
            radius = 0
        key = (distance, -radius, -(cand["w"] * cand["h"]), -cand["score"])
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


def switch_tracked_target_to_nearest(current_target, nearest_target):
    if nearest_target is None:
        return current_target

    switched = nearest_target.copy()

    if current_target is not None:
        if "diameter_history" in current_target and "diameter_history" not in switched:
            switched["diameter_history"] = list(current_target["diameter_history"])
        if "radius_history" in current_target and "radius_history" not in switched:
            switched["radius_history"] = list(current_target["radius_history"])
        if "scan_radius_history" in current_target and "scan_radius_history" not in switched:
            switched["scan_radius_history"] = list(current_target["scan_radius_history"])
        if "scan_diameter_history" in current_target and "scan_diameter_history" not in switched:
            switched["scan_diameter_history"] = list(current_target["scan_diameter_history"])

    ensure_target_id(switched)
    switched["miss"] = 0
    return switched


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
    global mode, pick_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, nearest_switch_count, play_ready_frames
    global play_presence_window, play_hit_latched
    global best_scan_tennis, tracked_tennis, tracked_player
    global scan_ranked_tennis, scan_candidate_index
    global scan_seek_left, scan_direction, current_racket_id
    global scan_tilt_reset_pending
    global pick_track_lost_count, pick_scan_fail_rounds

    mode = MODE_PICK
    pick_substate = PICK_SCAN
    capture_cmd = 0
    capture_flash_frames = 0
    player_locked = False
    balls_served = 0
    nearest_switch_count = 0
    play_ready_frames = 0
    best_scan_tennis = None
    scan_ranked_tennis = []
    scan_candidate_index = 0
    tracked_tennis = None
    tracked_player = None
    scan_seek_left = True
    scan_tilt_reset_pending = True
    scan_direction = 1
    reset_pick_confirm_window()
    reset_play_lock_state()
    reset_play_hit_state()
    current_racket_id = 0
    pick_track_lost_count = 0
    pick_scan_fail_rounds = 0


def enter_play_mode():
    global mode, pick_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, nearest_switch_count, play_ready_frames
    global play_presence_window, play_hit_latched
    global best_scan_tennis, tracked_tennis, tracked_player, current_racket_id
    global scan_ranked_tennis, scan_candidate_index
    global scan_direction, pick_track_lost_count, pick_scan_fail_rounds

    mode = MODE_PLAY
    pick_substate = PICK_SCAN
    capture_cmd = 0
    capture_flash_frames = 0
    player_locked = False
    balls_served = 0
    nearest_switch_count = 0
    play_ready_frames = 0
    best_scan_tennis = None
    scan_ranked_tennis = []
    scan_candidate_index = 0
    scan_direction = 1
    reset_pick_confirm_window()
    reset_play_lock_state()
    reset_play_hit_state()
    tracked_tennis = None
    tracked_player = None
    current_racket_id = 0
    pick_track_lost_count = 0
    pick_scan_fail_rounds = 0
    lift_tilt_for_play_mode()


def run_state_machine(img, tennis_target, tennis_candidates, player_target, racket_candidates):
    global mode, pick_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, play_ready_frames
    global nearest_switch_count, best_scan_tennis, tracked_tennis
    global scan_ranked_tennis, scan_candidate_index
    global current_racket_id
    global pick_track_lost_count

    command_event = None
    player_live = is_live_target(player_target)
    racket_target = choose_racket_target(racket_candidates, player_target)
    racket_visible = racket_target is not None
    racket_present = racket_visible
    if racket_present:
        if current_racket_id <= 0:
            current_racket_id = allocate_target_id()
        ensure_target_id(racket_target, current_racket_id)
    else:
        current_racket_id = 0
        racket_target = None
    nearest_tennis = choose_nearest_tennis(tennis_candidates)
    active_target = tennis_target
    host_ctrl_ok = host_control_enabled()
    requested_mode = -1

    if host_ctrl_ok:
        requested_mode = consume_host_mode_switch_event()

    if requested_mode == MODE_PICK and mode != MODE_PICK:
        enter_pick_mode()
        return None, "SEEK", "SCAN", False, None, command_event
    if requested_mode == MODE_PLAY and mode != MODE_PLAY:
        enter_play_mode()
        return None, "PLAY", "TRACK_P", False, None, command_event

    if capture_flash_frames > 0:
        capture_cmd = 1
        capture_flash_frames -= 1
    else:
        capture_cmd = 0

    read_picker_feedback(consume=False)

    if mode == MODE_PICK:
        if pick_substate == PICK_SCAN:
            if update_servo_init():
                return None, "INIT", "INIT", racket_present, racket_target, command_event

            scan_done = update_global_scan(img, tennis_candidates)
            if scan_done:
                if len(scan_ranked_tennis) > 0 and select_scan_candidate(0):
                    tracked_tennis = None
                    reset_pick_confirm_window()
                    pick_substate = PICK_RETURN
                else:
                    return handle_pick_scan_round_failed(racket_present, racket_target, command_event)
            # 扫描阶段只记录候选，不提前锁定或显示跟踪目标。
            return None, "SEEK", "SCAN", racket_present, racket_target, command_event

        if pick_substate == PICK_RETURN:
            if best_scan_tennis is None:
                if not select_scan_candidate(scan_candidate_index):
                    return handle_pick_scan_round_failed(racket_present, racket_target, command_event)

            active_target = best_scan_tennis
            if update_return_to_saved_target():
                reset_pick_confirm_window()
                pick_substate = PICK_CONFIRM
            return active_target, "SEEK", "RETURN", racket_present, racket_target, command_event

        if pick_substate == PICK_CONFIRM:
            if best_scan_tennis is None:
                if select_next_scan_candidate():
                    pick_substate = PICK_RETURN
                    return best_scan_tennis, "SEEK", "RETURN", racket_present, racket_target, command_event
                return handle_pick_scan_round_failed(racket_present, racket_target, command_event)

            confirmed_target = choose_local_nearest_tennis(best_scan_tennis, tennis_candidates)
            if confirmed_target is None:
                confirmed_target = choose_confirmed_scan_target(best_scan_tennis, tennis_candidates)

            if confirmed_target is not None:
                note_pick_confirm_frame(True)
                best_scan_tennis = confirmed_target.copy()
                active_target = best_scan_tennis
                update_target_lock_pan(active_target, img)
            else:
                note_pick_confirm_frame(False)
                active_target = best_scan_tennis

            if pick_confirm_ready_to_lock() and best_scan_tennis is not None:
                tracked_tennis = best_scan_tennis.copy()
                ensure_target_id(tracked_tennis, int(best_scan_tennis.get("id", 0)))
                tracked_tennis["miss"] = 0
                reset_pick_confirm_window()
                reset_pick_scan_fail_rounds()
                clear_pick_unlock_events()
                pick_track_lost_count = 0
                pick_substate = PICK_TRACK
                return tracked_tennis, "SEEK", "TRACK", racket_present, racket_target, command_event

            if pick_confirm_total_frames >= PICK_CONFIRM_MAX_FRAMES:
                if (pick_confirm_seen_frames >= PICK_FINAL_LOCK_MIN_SEEN_FRAMES) and best_scan_tennis is not None:
                    tracked_tennis = best_scan_tennis.copy()
                    ensure_target_id(tracked_tennis, int(best_scan_tennis.get("id", 0)))
                    tracked_tennis["miss"] = 0
                    reset_pick_confirm_window()
                    reset_pick_scan_fail_rounds()
                    clear_pick_unlock_events()
                    pick_track_lost_count = 0
                    pick_substate = PICK_TRACK
                    return tracked_tennis, "SEEK", "TRACK", racket_present, racket_target, command_event

                if select_next_scan_candidate():
                    pick_substate = PICK_RETURN
                    return best_scan_tennis, "SEEK", "RETURN", racket_present, racket_target, command_event
                return handle_pick_scan_round_failed(racket_present, racket_target, command_event)

            return active_target, "SEEK", "CONFIRM", racket_present, racket_target, command_event

        active_target = tennis_target if tennis_target is not None else tracked_tennis

        if should_switch_to_nearest(active_target, nearest_tennis):
            nearest_switch_count += 1
        else:
            nearest_switch_count = 0

        if nearest_switch_count >= NEAREST_SWITCH_CONFIRM_FRAMES:
            active_target = switch_tracked_target_to_nearest(active_target, nearest_tennis)
            tracked_tennis = active_target.copy()
            nearest_switch_count = 0

        picker_done = read_picker_feedback()
        host_unlock_track = host_ctrl_ok and consume_host_track_unlock_event()
        unlock_pick_track = picker_done or host_unlock_track

        if unlock_pick_track:
            capture_flash_frames = CAPTURE_HOLD_FRAMES
            capture_cmd = 1
            reset_pick_scan_fail_rounds()
            begin_scan_round()
            return None, "SEEK", "SCAN", racket_present, racket_target, command_event

        if is_live_target(active_target):
            pick_track_lost_count = 0
            update_target_lock_pan(active_target, img)
            update_servo_tracking(active_target, img)
        else:
            pick_track_lost_count += 1
            if pick_track_lost_count >= PICK_TRACK_LOST_CONSECUTIVE_TH:
                reset_pick_scan_fail_rounds()
                begin_scan_round()
                return None, "SEEK", "SCAN", racket_present, racket_target, command_event

        return active_target, "SEEK", "TRACK", racket_present, racket_target, command_event

    player_locked = update_player_lock_state(player_live)
    active_target = player_target
    if player_live:
        update_servo_tracking(player_target, img)
    elif player_locked:
        active_target = tracked_player

    if player_locked:
        hit_triggered = note_play_presence(player_live, racket_present)
        if hit_triggered:
            capture_flash_frames = CAPTURE_HOLD_FRAMES
            capture_cmd = 1
            command_event = (PLAY_HIT_SIGNAL_CMD, PLAY_HIT_SIGNAL_ARG)
        return active_target, "PLAY", "TRACK_P", racket_present, racket_target, command_event

    reset_play_hit_state()
    current_racket_id = 0
    update_play_search_motion(img)
    return None, "PLAY", "SEARCH_P", racket_present, racket_target, command_event


def draw_status_panel(img, fps, mode_name, state_name, racket_present):
    global servo_init_frames_remaining, comm_state
    global pick_confirm_seen_frames, pick_confirm_total_frames

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
        play_state_text = "search" if state_name == "SEARCH_P" else "track"
        img.draw_string(
            2,
            200,
            "player:%d racket:%d %s"
            % (1 if player_locked else 0, 1 if racket_present else 0, play_state_text),
            color=WHITE,
            mono_space=False,
        )
    elif state_name == "CONFIRM":
        img.draw_string(
            2,
            200,
            "confirm:%d/%d hit:%d"
            % (
                pick_confirm_total_frames,
                PICK_CONFIRM_MAX_FRAMES,
                pick_confirm_seen_frames,
            ),
            color=WHITE,
            mono_space=False,
        )
    elif state_name == "TRACK":
        if host_control_enabled():
            track_text = "track:host fb:%d lost:%d/%d" % (
                picker_feedback_state,
                pick_track_lost_count,
                PICK_TRACK_LOST_CONSECUTIVE_TH,
            )
        else:
            track_text = "track:fb:%d lost:%d/%d" % (
                picker_feedback_state,
                pick_track_lost_count,
                PICK_TRACK_LOST_CONSECUTIVE_TH,
            )
        img.draw_string(
            2,
            200,
            track_text,
            color=WHITE,
            mono_space=False,
        )
    elif best_scan_tennis is not None and best_scan_tennis.get("dist_cm") is not None:
        scan_radius = scan_lock_radius(best_scan_tennis)
        img.draw_string(
            2,
            200,
            "scan:%.1fcm r:%d p:%d fb:%d"
            % (
                best_scan_tennis["dist_cm"],
                scan_radius,
                int(best_scan_tennis.get("servo_pan", pan_angle)),
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
    return (distance_mm / 10.0) * DISTANCE_OUTPUT_SCALE


def estimate_scaled_distance_cm(pixel_diameter):
    base_distance_cm = estimate_distance(pixel_diameter)
    if base_distance_cm is None:
        return None
    return base_distance_cm * DISTANCE_SCALE


def estimate_corrected_distance_cm(pixel_diameter):
    scaled_distance_cm = estimate_scaled_distance_cm(pixel_diameter)
    if scaled_distance_cm is None:
        return None
    return correct_tennis_distance(scaled_distance_cm, pixel_diameter)


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
    close_mode = is_close_range_candidate(target)
    # 近球始终优先跑霍夫圆，避免用依赖半径的距离估计反过来决定半径策略。
    diameter, cue_conf, measure_src, measure_cx, measure_cy = estimate_tennis_diameter(
        img,
        target["x"],
        target["y"],
        target["w"],
        target["h"],
        allow_hough=allow_hough,
        close_mode=close_mode,
    )
    raw_radius = fuse_tennis_radius(target["w"], target["h"], diameter, close_mode=close_mode)
    filtered_diameter, refined_radius = update_trimmed_tennis_measure(target, diameter, raw_radius)
    refined_radius = smooth_ball_radius(refined_radius, prev_radius)

    raw_dist_cm = estimate_scaled_distance_cm(diameter)
    refined_dist_cm = estimate_corrected_distance_cm(filtered_diameter)
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
    target["raw_dist_cm"] = raw_dist_cm
    target["cue_conf"] = cue_conf
    target["close_fit"] = 1 if close_mode else 0
    target["measure_src"] = measure_src
    target["measure_cx"] = measure_cx
    target["measure_cy"] = measure_cy
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

    threshold_list = [(math.ceil(FOMO_HEATMAP_SCORE_TH * 255), 255)]
    results = [[] for _ in range(oc)]

    # FOMO 第 0 通道是 background，主循环不会使用，直接跳过可少做一轮热力图后处理。
    for i in range(1, oc):
        channel = outputs[0][0, :, :, i]
        heatmap = make_grayscale_image(channel, oh, ow)
        blobs = heatmap.find_blobs(
            threshold_list,
            x_stride=1,
            y_stride=1,
            area_threshold=FOMO_HEATMAP_AREA_TH,
            pixels_threshold=FOMO_HEATMAP_PIXELS_TH,
            merge=True,
            margin=FOMO_HEATMAP_MERGE_MARGIN,
        )
        for blob in blobs:
            rect = blob.rect()
            x, y, w, h = rect
            score = heatmap.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            if score < FOMO_HEATMAP_SCORE_TH:
                continue
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
    # enter_play_mode()
    init_grid_overlay()
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
                        radius = max(4, min(40, int(max(w, h) * 0.7)))
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
                                "merge_count": 1,
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

            if tennis_candidates:
                tennis_candidates = merge_duplicate_tennis_candidates(
                    tennis_candidates, img.width(), img.height()
                )
                tennis_candidates = [
                    cand
                    for cand in tennis_candidates
                    if tennis_center_is_green(img, cand["x"], cand["y"], cand["w"], cand["h"])
                ]

            refine_all_tennis = (
                ENABLE_TENNIS_REFINEMENT
                and REFINE_ALL_TENNIS_IN_PICK_SCAN
                and (mode == MODE_PICK)
                and (pick_substate == PICK_SCAN)
            )
            if refine_all_tennis:
                tennis_candidates = refine_tennis_candidates(img, tennis_candidates, allow_hough=True)
                tennis_candidates = dedupe_close_refined_tennis_candidates(tennis_candidates)

            seek_track_enabled = detect_tennis and (pick_substate == PICK_TRACK)
            tennis_target = match_tennis_track(tennis_candidates) if seek_track_enabled else None
            if detect_tennis and (not refine_all_tennis):
                tennis_target = refine_tennis_target(img, tennis_target)
            player_target = choose_player_target(player_candidates) if detect_play_targets else None
            active_target, mode_name, state_name, racket_present, racket_target, command_event = run_state_machine(
                img, tennis_target, tennis_candidates, player_target, racket_candidates
            )
            draw_grid_overlay(img)
            draw_seek_tennis_candidates(img, mode_name, state_name, tennis_candidates)
            draw_active_target(img, active_target, mode_name, state_name)
            draw_racket_target(img, racket_target)
            draw_status_panel(img, clock.fps(), mode_name, state_name, racket_present)
            send_runtime_packets(mode_name, state_name, active_target)
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
