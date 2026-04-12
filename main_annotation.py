"""
main_annotation.py
------------------
此文件为 OpenMV/微控制器上运行的物体检测与舵机跟踪主程序。
实现功能包括：
- 加载轻量级神经网络（tflite）并运行前处理/后处理
- 对“网球/球员/球拍”等目标进行检测、候选筛选与多帧融合
- 细化球的直径/半径估计（颜色分割 + Hough 圆检测），并估算距离
- 基于 PID 的舵机控制（panning/tilting）以跟踪目标
- 与外部主机通过 UART 简单通讯（握手、上报事件、接收计数）
- 提供拾取（PICK）与比赛（PLAY）两类运行模式与状态机

注：注释以中文给出，目的是帮助理解算法和实现细节，尽量说明每个函数的输入/输出/副作用。
"""

import gc  # 垃圾回收模块，用于手动触发或检查内存
import math  # 数学函数库，提供 sqrt、abs 等
import sys  # 访问解释器和系统相关功能（如异常打印）
import time  # 时间相关函数（ticks_ms、sleep 等）
import uos  # 文件系统接口（用于检查模型文件大小等）

import display  # OpenMV 的显示驱动，用于 SPI/LCD 输出
import image  # OpenMV 的图像处理库（draw, find_blobs, find_circles 等）
import ml  # OpenMV 的轻量级机器学习模型接口（加载 tflite）
import sensor  # 摄像头传感器控制接口
from pyb import Pin, Timer, UART  # 硬件引脚、定时器与串口接口
from pid import PID  # 简单 PID 控制器实现（用于舵机跟踪）


MODEL_PATH = "trained.tflite"
LABELS_PATH = "labels.txt"
LCD_HINT = image.ROTATE_270

# 是否启用舵机控制（物理伺服）。测试或无硬件时可设为 False
ENABLE_SERVOS = True
# Pan 舵机所在的引脚名称（板上连接）
PAN_SERVO_PIN = "P1"
# Tilt 舵机所在的引脚名称（板上连接）
TILT_SERVO_PIN = "P9"
# 是否启用 UART 与主机/拾取器通信
ENABLE_UART = True
# 串口端口号（pyb.UART 的端口索引）
UART_PORT = 3
# 串口波特率
UART_BAUDRATE = 115200
# UART 读取超时依据的字符数（microPython 特性）
UART_TIMEOUT_CHAR = 120
# 串口接收缓存最大长度，超过则保留尾部以避免内存无限增长
UART_RX_BUFFER_MAX = 96

# UART 通信链路状态机：
# COMM_NO_LINK: 未建立链路
# COMM_LINKED: 收到对端 HI 握手
# COMM_OK: 收到对端 OK 确认
COMM_NO_LINK = 0  # 无链路状态
COMM_LINKED = 1  # 收到 HI，链路已建立但未完全确认
COMM_OK = 2  # 收到 OK，链路完全正常
comm_state = COMM_NO_LINK  # 当前链路状态，初始为无链路
# 上次发送 hello 的时间戳（毫秒），用于周期性重发握手
last_hello_ms = 0
HELLO_INTERVAL_MS = 5000  # ms，hello 重试间隔

# 各类别置信度阈值（由模型预测的 score 与之比较）
THRESH_TENNIS = 0.35  # 网球检测置信度阈值（score >= 此值认为有效）
# player/racket 门限设置更高以减少误检
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

# 全局硬件/运行时状态变量（初始化为 None/False）
lcd = None  # SPI LCD 显示对象（display.SPIDisplay 实例）
uart = None  # UART 对象（若启用）
labels = None  # 模型标签列表
net = None  # 已加载的 ml.Model 对象
# 当前正在跟踪的目标（tennis / player），为字典或 None
tracked_tennis = None
tracked_player = None
# 舵机引脚对象和定时器对象（在 init_servos 中创建）
p1 = None
p9 = None
p1_tim_pluse = None
p1_tim_main = None
p9_tim_pluse = None
p9_tim_main = None
pan_pwm_started = False  # pan PWM 是否已启动标志
tilt_pwm_started = False  # tilt PWM 是否已启动标志
# 帧计数索引用于控制某些周期性计算（如 Hough 间隔）
frame_index = 0

TRACK_MAX_MISS = 8
TRACK_GATE_MIN = 28
TRACK_SMOOTH_OLD_NUM = 7
TRACK_SMOOTH_NEW_NUM = 3

PAN_INIT_ANGLE = 90.0
TILT_INIT_ANGLE = 132.0
SERVO_INIT_HOLD_FRAMES = 28
SERVO_PWM_STAGGER_FRAMES = 8

pan_angle = PAN_INIT_ANGLE  # 当前 pan 角度（度）
tilt_angle = TILT_INIT_ANGLE  # 当前 tilt 角度（度）
# 舵机角度安全/扫描边界
pan_angle_limit = [20.0, 160.0]
tilt_angle_limit = [80.0, 150.0]
SERVO_DEADBAND = 6  # 误差死区（像素），小于该值不触发舵机动作
SCAN_TILT_TARGET = 132.0  # 扫描时默认 tilt 的目标角度
SCAN_TILT_FLOOR = 128.0  # 扫描时 tilt 的下限
SCAN_TILT_STEP = 1.2  # 扫描 tilt 逐步移动的步长
SERVO_INIT_STEP = 0.35  # 舵机回中初始步长
SCAN_PAN_STEP = 0.75  # 扫描 pan 的步长
TRACK_PAN_MAX_STEP = 1.1  # 跟踪时 pan 的最大步长限制
TRACK_TILT_MAX_STEP = 1.0  # 跟踪时 tilt 的最大步长限制

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

mode = MODE_PICK  # 当前运行模式（捡球 / 比赛）
pick_substate = PICK_SCAN  # pick 模式下的子状态初始为扫描
play_substate = PLAY_TRACK_PLAYER  # play 模式下的子状态

capture_cmd = 0  # 是否发起拍照/捕获命令（由 picker 使用）
capture_flash_frames = 0  # 捕获保留帧计数
player_locked = False  # player 是否被锁定
balls_served = 0  # 上报的发球计数
balls_picked = 0  # 上报的拾取计数
target_balls = 5  # 目标拾取/发球数（可配置）
scan_direction = 1  # 扫描方向（左右切换）
scan_speed = SCAN_PAN_STEP  # 扫描速度（pan 步长）
scan_lock_count = 0  # 扫描时的锁定计数
nearest_switch_count = 0  # 切换到最近目标时的确认计数
play_ready_frames = 0  # play 模式准备就绪帧计数（用于确认 racket）
racket_seen_prev = False  # 上一帧是否看见 racket
best_scan_tennis = None  # 当前最佳扫描目标
scan_ranked_tennis = []  # 扫描结果的候选列表（按优先级排序）
scan_candidate_index = 0  # 当前选择的扫描候选索引
scan_empty_rounds = 0  # 连续空扫描轮数计数
player_lock_start_ms = 0  # player 锁定开始时间戳（ms）
servo_init_frames_remaining = 0  # 舵机回中时剩余帧数
scan_seek_left = True  # 初始扫描方向标志（左寻）
pick_confirm_start_ms = 0  # pick 确认开始时间戳
pick_track_start_ms = 0  # pick 跟踪开始时间戳
picker_feedback_pin = None  # picker 的反馈引脚对象
picker_feedback_state = 0  # picker 反馈当前状态
picker_uart_done_pending = 0  # 通过 UART 通知的 picker 完成待处理计数
next_target_id = 1  # 分配目标 id 的自增计数器
current_racket_id = 0  # 当前跟踪的 racket id
uart_rx_buffer = ""  # 串口接收缓存（累积未处理的文本）

ENABLE_TENNIS_REFINEMENT = True  # 是否启用网球精化（颜色/Hough 融合）
HOUGH_INTERVAL = 3  # Hough 检测的帧间隔（每隔 n 帧运行一次）
DISTANCE_SCALE = 2.15  # 经验尺度因子，用于将像素距离换算为实际距离
REFINE_ALL_TENNIS_IN_PICK_SCAN = True  # 在 pick scan 阶段是否对所有候选做精化
MEASURE_WINDOW_LEN = 10  # 测量滑动窗口长度（用于截尾均值）
MEASURE_TRIM_COUNT = 2  # 截尾均值两端要去掉的元素数量

COLOR_TOL_L_BASE = 12  # LAB 色彩阈值基准：L 分量基础公差
COLOR_TOL_A_BASE = 10  # LAB a 分量基础公差
COLOR_TOL_B_BASE = 10  # LAB b 分量基础公差
COLOR_TOL_L_EXTRA = 3  # L 分量额外容差
COLOR_TOL_A_EXTRA = 3  # a 分量额外容差
COLOR_TOL_B_EXTRA = 3.5  # b 分量额外容差

DRAW_RADIUS_SCALE = 1.15  # 绘制圆半径的缩放因子
ROI_NEAR_SWITCH = 26  # 判断近距离 ROI 的阈值（像素）
FAR_ROI_PAD_MIN = 6  # 远 ROI 的最小 padding
MAX_COLOR_BLOB_AREA_MULT = 6  # 颜色斑块最大面积相对 bbox 面积的倍数阈值


EDGE_LOW_TH = 55  # Canny 边缘检测低阈值
EDGE_HIGH_TH = 110  # Canny 边缘检测高阈值

GLARE_L_TH = 88  # 亮度阈值，用于检测眩光/高光区域
GLARE_RATIO_TH_PCT = 18  # 视为眩光的像素占比阈值百分比
SMALL_BALL_SIZE_TH = 24  # 被视为小球的像素尺寸阈值
GLARE_DIAMETER_CAP_PCT = 115  # 眩光情况下直径上界的百分比上限

# 以上为配置/阈值：可以根据摄像头、光照、见样本调优。
# 注：大多数常量带有注释说明其目的与调整效果，便于工程调参。


def init_camera():
    """
    初始化摄像头传感器配置。

    配置项：
    - 像素格式：RGB565（彩色）
    - 分辨率：QVGA（320x240）
    - 关闭自动白平衡以保证颜色稳定性（对于颜色分割与 Hough 更可靠）
    - 跳过若干帧以让传感器稳定

    无参数，直接通过 sensor 模块进行硬件设置。
    """
    sensor.reset()  # 重置摄像头传感器到默认状态
    sensor.set_pixformat(sensor.RGB565)  # 设置像素格式为 RGB565（彩色）
    sensor.set_framesize(sensor.QVGA)  # 设置分辨率为 QVGA（320x240）
    sensor.set_auto_whitebal(False)  # 关闭自动白平衡以保持颜色一致性
    sensor.skip_frames(10)  # 跳过若干帧等待摄像头参数稳定


def init_lcd():
    """
    初始化 LCD 显示器（SPI 接口）。

    等待 300ms 再创建 SPI 显示对象以避免上电时序问题。
    将全局变量 `lcd` 赋值为 Display 对象，后续通过 `lcd.write(img)` 刷屏。
    """
    global lcd
    time.sleep_ms(300)  # 等待电源/屏幕上电稳定（避免 SPI 时序问题）
    lcd = display.SPIDisplay(width=240, height=320)  # 创建 SPI LCD 显示对象并赋值给全局


def init_uart():
    """
    初始化 UART 串口连接（如果启用）。

    成功时将全局 `uart` 置为 UART 对象，失败则设为 None 并打印异常。
    timeout_char 用于指定读取超时的字符阈值（microPython 特性）。
    """
    global uart

    if not ENABLE_UART:
        uart = None  # 如果未启用 UART，则直接置 None 并返回
        return

    try:
        uart = UART(UART_PORT, UART_BAUDRATE, timeout_char=UART_TIMEOUT_CHAR)  # 初始化 UART
    except Exception as err:
        uart = None  # 出错则设置为 None 并打印异常（不抛出）
        sys.print_exception(err)


def P1_ISR0(t):
    """
    Pan（P1）舵机脉冲结束中断回调。

    由主 PWM 定时器启动短脉冲后调用此回调以拉低引脚并停止脉冲定时器。
    """
    p1.low()  # 将 pan 引脚拉低，结束脉冲
    p1_tim_pluse.deinit()  # 停用短脉冲定时器


def P1_ISR(t):
    """
    Pan（P1）主 PWM 定时器回调：根据当前 pan_angle 计算短脉冲时长，拉高引脚并安排 P1_ISR0 在短脉冲后拉低。

    这里使用两个定时器配合：主定时器周期驱动脉冲触发，短定时器负责拉低以形成伺服脉宽。
    """
    psr = int(pan_angle * 1000 / 9 + 5000) - 1  # 根据角度计算短脉冲的 prescaler（经验公式）
    p1.high()  # 将引脚拉高开始脉冲
    p1_tim_pluse.init(prescaler=psr, period=23)  # 启动短脉冲定时器，period 固定形成脉宽
    p1_tim_pluse.callback(P1_ISR0)  # 在短脉冲结束时由 P1_ISR0 拉低引脚


def P9_ISR0(t):
    """
    Tilt（P9）舵机脉冲结束中断回调，作用同 P1_ISR0。
    """
    p9.low()  # 将 tilt 引脚拉低，结束短脉冲
    p9_tim_pluse.deinit()  # 停用 tilt 的短脉冲定时器


def P9_ISR(t):
    """
    Tilt（P9）主 PWM 定时器回调，作用同 P1_ISR。
    根据 tilt_angle 计算短脉冲宽度并启动短定时器。
    """
    psr = int(tilt_angle * 1000 / 9 + 5000) - 1  # 根据 tilt 角度计算短脉冲 prescaler
    p9.high()  # 开始 tilt 脉冲
    p9_tim_pluse.init(prescaler=psr, period=23)  # 启动短脉冲定时器以控制舵机占空比
    p9_tim_pluse.callback(P9_ISR0)  # 短脉冲结束时回调 P9_ISR0 拉低引脚


def init_servos():
    global p1, p9, p1_tim_pluse, p1_tim_main, p9_tim_pluse, p9_tim_main
    global pan_angle, tilt_angle, servo_init_frames_remaining
    global pan_pwm_started, tilt_pwm_started

    """
    初始化舵机相关的 GPIO 与定时器。

    - 为 pan/tilt 创建 Pin 对象
    - 创建短脉冲定时器和主 PWM 定时器（频率约 50Hz）
    - 设置初始角度与启动帧计数以完成开机回中动作

    该函数不会立即启动 PWM 回调（通过 start_pan_pwm/start_tilt_pwm 分别启动），
    以便控制先后顺序和抖动避免冲突。
    """
    global p1, p9, p1_tim_pluse, p1_tim_main, p9_tim_pluse, p9_tim_main
    global pan_angle, tilt_angle, servo_init_frames_remaining
    global pan_pwm_started, tilt_pwm_started

    if not ENABLE_SERVOS:
        return  # 未启用舵机时直接返回

    pan_angle = PAN_INIT_ANGLE  # 初始化 pan 角度
    tilt_angle = TILT_INIT_ANGLE  # 初始化 tilt 角度
    servo_init_frames_remaining = SERVO_INIT_HOLD_FRAMES  # 设置回中保留帧数
    pan_pwm_started = False  # 标记 PWM 未启动
    tilt_pwm_started = False

    p1 = Pin(PAN_SERVO_PIN, Pin.OUT_PP)  # 配置 pan 引脚为推挽输出
    p9 = Pin(TILT_SERVO_PIN, Pin.OUT_PP)  # 配置 tilt 引脚为推挽输出

    p1_tim_pluse = Timer(12)  # pan 的短脉冲定时器（用于产生脉宽）
    p1_tim_main = Timer(13, freq=50)  # pan 主定时器，50Hz 驱动周期

    p9_tim_pluse = Timer(14)  # tilt 的短脉冲定时器
    p9_tim_main = Timer(15, freq=50)  # tilt 主定时器，50Hz 驱动


def start_pan_pwm():
    global pan_pwm_started
    """
    启动 pan 主 PWM 回调（连接到 P1_ISR）。
    仅在尚未启动并且主定时器存在时生效。
    """
    global pan_pwm_started

    if pan_pwm_started or (p1_tim_main is None):
        return  # 已启动或主定时器不存在则跳过
    p1_tim_main.callback(P1_ISR)  # 将主定时器回调关联到 P1_ISR，开始 PWM
    pan_pwm_started = True  # 标记为已启动


def start_tilt_pwm():
    global tilt_pwm_started
    """
    启动 tilt 主 PWM 回调（连接到 P9_ISR）。
    """
    global tilt_pwm_started

    if tilt_pwm_started or (p9_tim_main is None):
        return
    p9_tim_main.callback(P9_ISR)  # 将 tilt 主定时器回调关联到 P9_ISR
    tilt_pwm_started = True


def init_picker_feedback():
    """
    初始化拾取器（picker）的状态反馈引脚。

    如果没有配置 `PICKER_FEEDBACK_PIN` 则忽略；否则将该引脚配置为输入，后续通过 `read_picker_feedback` 读取。
    """
    global picker_feedback_pin

    if not PICKER_FEEDBACK_PIN:
        picker_feedback_pin = None  # 若未配置引脚则不启用反馈
        return

    picker_feedback_pin = Pin(PICKER_FEEDBACK_PIN, Pin.IN)  # 配置反馈引脚为输入


def read_picker_feedback(consume=True):
    """
    读取 picker 的反馈状态。

    优先级：如果 `picker_uart_done_pending` > 0（由 UART 命令引入），则返回 1 并可选择性消耗计数；
    否则从引脚读取实际电平判断当前反馈状态（与 `PICKER_FEEDBACK_ACTIVE_LEVEL` 比较）。

    参数：
    - consume (bool): 若为 True，且存在 pending，则消费一个 pending 计数。

    返回：0/1 表示反馈是否处于活动状态。
    """
    global picker_feedback_state, picker_uart_done_pending

    if picker_uart_done_pending > 0:  # 若有 UART 上报的待消费完成事件，优先返回活动
        picker_feedback_state = 1
        if consume:
            picker_uart_done_pending -= 1  # 可选地消费一个 pending
        return 1

    if picker_feedback_pin is None:  # 若未配置硬件引脚则视为未就绪
        picker_feedback_state = 0
        return 0

    try:
        # 读取引脚电平并与设定的激活电平比较
        picker_feedback_state = 1 if picker_feedback_pin.value() == PICKER_FEEDBACK_ACTIVE_LEVEL else 0
    except Exception:
        picker_feedback_state = 0  # 出错时安全地认为未激活
    return picker_feedback_state


def uart_bytes_to_text(data):
    """
    将从 UART 读取的 bytes 或 str 转换为干净的文本字符串。

    逻辑：
    - 若为 None 返回空串
    - 若已是 str 直接返回
    - 优先尝试 .decode()（UTF-8），失败则筛选出 ASCII/控制换行字符并拼接

    这能容忍不规范或包含控制字符的串口数据，避免抛异常。
    """
    if data is None:
        return ""  # 无数据时返回空字符串
    if isinstance(data, str):
        return data  # 已经是字符串则直接返回

    try:
        return data.decode()  # 优先用 UTF-8 解码 bytes
    except Exception:
        pass  # 解码失败则走容错路径

    text = ""  # 回退：逐字节筛选可显示字符
    for b in data:
        if b in (10, 13) or (32 <= b <= 126):
            text += chr(b)  # 只保留换行/回车和可打印 ASCII
    return text


def parse_uart_event_count(text):
    """
    从串口文本中解析可能的计数值。

    例如："PICK:3"、"SERVED=2"、"PICK 5" 等形式，函数会尝试从常见分隔符之后解析整数。
    如果无法解析，则默认返回 1（表示一个事件）。
    """
    separators = (":", ",", "=", " ")  # 常见的分隔符
    for sep in separators:
        idx = text.find(sep)  # 查找分隔符位置
        if idx < 0:
            continue  # 未找到则尝试下一个分隔符
        tail = text[idx + 1 :].strip()  # 分隔符后面的内容
        if not tail:
            continue  # 无尾部内容则继续
        try:
            value = int(tail)  # 尝试解析整数
            if value > 0:
                return value  # 返回正整数计数
        except Exception:
            pass  # 解析失败则忽略
    return 1  # 默认返回 1 表示单次事件


def handle_uart_line(line):
    """
    处理从 UART 接收到的一行文本命令/事件。

    支持类型：
    - 握手/链路："HI" -> 将 comm_state 设为 LINKED；"OK" -> COMM_OK
    - 事件计数："SERVED", "SERVE" 增加 balls_served；
      "PICKED", "PICK", "PICKUP", "COLLECT" 增加 balls_picked，并向 picker_uart_done_pending 添加待处理计数

    参数：line（字符串）
    无返回值，通过修改全局计数或通信状态来传达结果。
    """
    global balls_served, balls_picked, picker_uart_done_pending
    global comm_state

    if not line:
        return  # 空行忽略

    text = line.strip()  # 去除两端空白
    if not text:
        return  # 仅空白也忽略

    upper = text.upper()  # 统一转大写便于匹配
    count = parse_uart_event_count(upper)  # 从文本中解析可能的计数

    # 处理握手与链路状态：HI 表示链路建立，OK 表示主机确认
    if upper == "HI" or upper.startswith("HI "):
        try:
            comm_state = COMM_LINKED  # 收到 HI 视为链路已建立
        except Exception:
            pass
        return
    if upper == "OK" or upper.startswith("OK "):
        try:
            comm_state = COMM_OK  # 收到 OK 则链路完全就绪
        except Exception:
            pass
        return

    if upper.startswith("SERVED") or upper.startswith("SERVE"):
        balls_served += count  # 更新发球计数
        return

    if (
        upper.startswith("PICKED")
        or upper.startswith("PICK")
        or upper.startswith("PICKUP")
        or upper.startswith("COLLECT")
    ):
        balls_picked += count  # 更新拾取计数
        picker_uart_done_pending += count  # 将完成事件放入 pending 列表以便被读取


def process_uart_rx():
    """
    非阻塞地读取 UART 接收缓冲区的数据并逐行处理。

    - 读取可用字节数（uart.any）并一次性读出
    - 将 bytes 转为文本并追加到全局缓存 `uart_rx_buffer`
    - 根据换行符拆分完整行并交由 `handle_uart_line` 处理
    - 为避免内存膨胀，超过 `UART_RX_BUFFER_MAX` 时保留尾部
    """
    global uart_rx_buffer

    if uart is None:
        return  # UART 未启用或初始化失败则直接返回

    try:
        waiting = uart.any()  # 查询可读取字节数
    except Exception:
        return  # 出错则忽略本次读取

    if not waiting:
        return  # 无数据可读

    try:
        data = uart.read(waiting)  # 读取全部可用字节
    except Exception:
        return  # 读取失败则放弃

    text = uart_bytes_to_text(data)  # 将 bytes 解码为可显示文本
    if not text:
        return  # 无有效文本则退出

    uart_rx_buffer += text  # 将新文本追加到缓存
    if len(uart_rx_buffer) > UART_RX_BUFFER_MAX:
        uart_rx_buffer = uart_rx_buffer[-UART_RX_BUFFER_MAX:]  # 超长则保留尾部

    while True:
        line_end = uart_rx_buffer.find("\n")  # 查找完整行分隔符
        if line_end < 0:
            break  # 没有完整行则退出循环
        line = uart_rx_buffer[:line_end].strip()  # 提取一行并去除两端空白
        uart_rx_buffer = uart_rx_buffer[line_end + 1 :]
        handle_uart_line(line)  # 交由行处理器解析


def display_frame(img):
    """
    将给定图像输出到 LCD（若已初始化）。

    参数:
    - img: image.Image 要显示的图像对象

    副作用: 写入全局 `lcd` 设备（若存在）。
    """
    if lcd is not None:
        lcd.write(img, hint=LCD_HINT)  # 将图像写到 LCD（如存在）


def try_send_hello():
    """
    在链路未建立时周期性发送 "hello" 握手，触发远端响应以建立通信链路。

    由主循环调用：该函数为非阻塞且只在 comm_state==COMM_NO_LINK 时发送。
    """
    global last_hello_ms, comm_state
    if uart is None:
        return  # 无 UART 时跳过
    if comm_state != COMM_NO_LINK:
        return  # 已有链路则不再发送 hello
    now = time.ticks_ms()  # 获取当前时刻（ms）
    if last_hello_ms <= 0:
        # 发送初始 hello
        uart_write_line("hello")
        last_hello_ms = now  # 记录发送时间
        return
    if time.ticks_diff(now, last_hello_ms) >= HELLO_INTERVAL_MS:
        uart_write_line("hello")  # 超时则重发 hello
        last_hello_ms = now


def show_message(line1, line2=None, color=YELLOW):
    """
    在屏幕上显示两行简短文本（用于启动/错误提示）。

    通过 snapshot 获取帧并直接绘制字符串后写入显示器。
    """
    img = sensor.snapshot()  # 抓取一帧用于显示文字背景
    img.draw_string(2, 2, line1, color=color, mono_space=False)  # 在左上绘制主行文本
    if line2:
        img.draw_string(2, 20, line2, color=color, mono_space=False)  # 可选的第二行
    display_frame(img)  # 输出到 LCD


def halt_with_error(title, err):
    """
    出错时在屏幕上显示错误并进入死循环（阻塞）。

    设计用途：启动阶段或不可恢复错误时调用，便于人工观察错误信息并阻止程序继续运行造成不确定行为。
    """
    try:
        img = sensor.snapshot()  # 捕获帧用于绘制错误信息
        img.draw_string(2, 2, title, color=RED, mono_space=False)  # 标题红色突出
        img.draw_string(2, 20, str(err), color=YELLOW, mono_space=False)  # 显示异常说明
        display_frame(img)  # 输出到屏幕以便人工查看
    except Exception:
        pass  # 如果绘制也失败则忽略，继续打印异常
    sys.print_exception(err)  # 将异常打印到控制台/串口
    while True:
        time.sleep_ms(1000)  # 进入阻塞循环，防止程序继续运行


def threshold_for_class(index):
    """
    根据模型输出通道索引返回对应的置信度阈值。

    约定：index 对应模型输出的类别索引（从 0 开始），本项目把 1-->tennis, 2-->player, 3-->racket
    若未识别到特定类别返回默认 0.5。
    """
    if index == 1:
        return THRESH_TENNIS  # 类别 1 对应 tennis
    if index == 2:
        return THRESH_PLAYER  # 类别 2 对应 player
    if index == 3:
        return THRESH_RACKET  # 类别 3 对应 racket
    return 0.50  # 其他类别默认阈值


def clamp(v, lo, hi):
    """
    将数值限制在区间 [lo, hi] 内的简单工具函数。
    """
    if v < lo:
        return lo  # 低于下界则返回下界
    if v > hi:
        return hi  # 高于上界则返回上界
    return v  # 在区间内则原样返回


def smooth_value(prev_v, curr_v):
    """
    对坐标等数值做指数平滑，避免目标抖动造成舵机频繁调整。

    使用加权平均：older_values 权重比 new_values 更大以保证平稳。
    """
    # 使用加权平均平滑值，旧值权重大以降低抖动
    return ((prev_v * TRACK_SMOOTH_OLD_NUM) + (curr_v * TRACK_SMOOTH_NEW_NUM)) // (
        TRACK_SMOOTH_OLD_NUM + TRACK_SMOOTH_NEW_NUM
    )


def draw_dashed_line(img, x0, y0, x1, y1, color, dash_len=8, gap_len=6):
    """
    在图像上绘制虚线（仅支持水平或垂直直线）。

    参数：起点(x0,y0) 到 终点(x1,y1)，dash_len 为线段长度，gap_len 为间隔长度。
    """
    if x0 == x1:  # 垂直虚线
        y = y0
        while y < y1:
            y_end = min(y + dash_len, y1)
            img.draw_line((x0, y, x1, y_end), color=color)  # 绘制一段实线
            y = y_end + gap_len  # 跳过 gap
    elif y0 == y1:  # 水平虚线
        x = x0
        while x < x1:
            x_end = min(x + dash_len, x1)
            img.draw_line((x, y0, x_end, y1), color=color)
            x = x_end + gap_len


def draw_grid(img, rows, cols, color):
    """
    在图像上绘制用于调试/定位的网格（rows x cols）。

    网格用于将检测结果映射到网格单元，便于选择最近目标或上报位置。
    """
    w = img.width()  # 图像宽度
    h = img.height()  # 图像高度
    for i in range(1, cols):
        x = (w * i) // cols  # 计算列线的 x 坐标
        draw_dashed_line(img, x, 0, x, h, color)  # 绘制垂直虚线
    for j in range(1, rows):
        y = (h * j) // rows  # 计算行线的 y 坐标
        draw_dashed_line(img, 0, y, w, y, color)  # 绘制水平虚线


def get_grid_position(x, y, img_w, img_h, rows, cols):
    """
    将像素坐标 (x,y) 映射到网格 (row, col)。

    返回值保证在合法范围内（0..rows-1, 0..cols-1）。
    """
    col = min(cols - 1, max(0, (x * cols) // img_w))  # 将 x 映射到列（0..cols-1）
    row = min(rows - 1, max(0, (y * rows) // img_h))  # 将 y 映射到行（0..rows-1）
    return row, col


def correct_tennis_distance(distance_cm, radius):
    """
    根据球的半径对估计的距离做经验校正并返回修正后的距离（cm）。

    理由：不同半径的球在像素尺度到实际距离的映射上存在偏差，
    通过经验因子对基于像素的距离估计进行修正以提高实际距离精度。

    参数:
    - distance_cm: float 基于像素估算得到的距离（厘米）
    - radius: int/float 目标球的像素半径

    返回: float 修正后的距离（厘米）
    """
    # 根据球半径选择经验修正因子
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
    """
    基于给定 bbox 在局部区域统计 LAB 分量并返回颜色阈值范围（L_lo,L_hi,A_lo,A_hi,B_lo,B_hi）。

    算法要点：在 bbox 中心取一个较小的种子区域统计均值和标准差，
    并基于全局常量与统计信息生成鲁棒的阈值以用于后续颜色分割。

    参数:
    - img: image.Image 当前帧
    - x,y,w,h: bbox 的坐标与尺寸

    返回: 元组 (L_lo, L_hi, A_lo, A_hi, B_lo, B_hi)
    """
    seed_w = max(6, (w * 2) // 5)  # 在 bbox 中取较小的种子区域宽度
    seed_h = max(6, (h * 2) // 5)  # 种子区域高度
    seed_x = clamp(x + (w - seed_w) // 2, 0, img.width() - 1)  # 种子区域左上 x
    seed_y = clamp(y + (h - seed_h) // 2, 0, img.height() - 1)  # 种子区域左上 y
    seed_w = clamp(seed_w, 1, img.width() - seed_x)  # 限制种子区域不超出图像边界
    seed_h = clamp(seed_h, 1, img.height() - seed_y)

    s = img.get_statistics(roi=(seed_x, seed_y, seed_w, seed_h))  # 在种子区域统计 LAB
    l_mean = s.l_mean()
    a_mean = s.a_mean()
    b_mean = s.b_mean()
    l_std = s.l_stdev()
    a_std = s.a_stdev()
    b_std = s.b_stdev()

    tol_l = int(clamp(COLOR_TOL_L_BASE + COLOR_TOL_L_EXTRA + l_std, 8, 24))  # L 分量容差
    tol_a = int(clamp(COLOR_TOL_A_BASE + COLOR_TOL_A_EXTRA + a_std, 6, 18))  # a 分量容差
    tol_b = int(clamp(COLOR_TOL_B_BASE + COLOR_TOL_B_EXTRA + b_std, 6, 18))  # b 分量容差

    l_lo = int(clamp(l_mean - tol_l, 0, 100))  # L 下界
    l_hi = int(clamp(l_mean + tol_l, 0, 100))  # L 上界
    a_lo = int(clamp(a_mean - tol_a, -128, 127))  # a 下界
    a_hi = int(clamp(a_mean + tol_a, -128, 127))  # a 上界
    b_lo = int(clamp(b_mean - tol_b, -128, 127))  # b 下界
    b_hi = int(clamp(b_mean + tol_b, -128, 127))  # b 上界
    return (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi)


def estimate_color_blob(img, roi, ref_cx, ref_cy, bbox_d):
    """
    在给定 ROI 内使用颜色分割寻找可能的球的颜色斑块，并返回最佳斑块的估计直径与中心。

    算法要点：
    - 基于 bbox 中心附近生成局部颜色阈值（build_tennis_color_threshold）以提高鲁棒性
    - 使用 find_blobs 找出与阈值匹配的连通区域
    - 过滤过大区域（可能为背景或高亮），并按综合评分挑选最佳斑块，评分考虑：
        距离（越中心越好）、形状接近圆（长宽比惩罚）、区域大小（更大更优）、是否覆盖参考点的额外加分

    返回： (color_diameter_estimate, center_x, center_y) 或 (None, ref_cx, ref_cy) 表示未找到。
    """
    rx, ry, rw, rh = roi  # 解包 ROI
    thr = build_tennis_color_threshold(
        img, ref_cx - (rw // 6), ref_cy - (rh // 6), max(6, rw // 3), max(6, rh // 3)
    )  # 基于 bbox 中心附近构建颜色阈值
    blobs = img.find_blobs(
        [thr],
        roi=roi,
        x_stride=1,
        y_stride=1,
        area_threshold=20,
        pixels_threshold=20,
        merge=True,
        margin=3,
    )  # 在 ROI 内查找颜色斑块

    if not blobs:
        return None, ref_cx, ref_cy  # 未找到斑块则返回 None

    best = None
    best_score = None
    max_blob_area = max(36, bbox_d * bbox_d * MAX_COLOR_BLOB_AREA_MULT)  # 过滤过大的斑块
    for b in blobs:
        if b.pixels() > max_blob_area:
            continue  # 过大可能为背景/亮斑，跳过

        dx = b.cx() - ref_cx  # 斑块中心相对于参考中心的偏移
        dy = b.cy() - ref_cy
        inside_ref = (b.x() <= ref_cx <= (b.x() + b.w())) and (b.y() <= ref_cy <= (b.y() + b.h()))

        long_side = max(b.w(), b.h())
        short_side = max(1, min(b.w(), b.h()))
        ratio = (long_side * 100) // short_side  # 长宽比 *100
        shape_penalty = abs(ratio - 100)  # 偏离圆形的惩罚项

        dist_cost = abs(dx) + abs(dy)  # 距离代价（越接近中心越好）
        size_gain = b.pixels() // 6  # 面积带来的奖励
        center_bonus = 60 if inside_ref else 0  # 若覆盖参考点则加分
        score = (dist_cost * 3) + shape_penalty - size_gain - center_bonus  # 综合评分，越小越好
        if (best_score is None) or (score < best_score):
            best_score = score
            best = b

    if best is None:
        return None, ref_cx, ref_cy

    eq_d = int(math.sqrt((4.0 * best.pixels()) / math.pi))  # 基于像素数估算等效直径
    blob_d = max(best.w(), best.h())  # 以最大边作为直径估计
    color_d = max(eq_d, blob_d)  # 取更保守的较大值
    return color_d, best.cx(), best.cy()


def estimate_edge_strength(img, roi):
    """
    使用 Canny 边缘算子评估给定 ROI 的边缘强度（返回灰度均值作为强度指标）。

    该指标用于在 Hough 圆检测中选择阈值（边缘更明显时可提高 Hough 阈值）。
    """
    edge_img = img.copy(roi=roi)  # 复制 ROI 子图以免破坏原图
    edge_img.to_grayscale()  # 转为灰度
    edge_img.find_edges(image.EDGE_CANNY, threshold=(EDGE_LOW_TH, EDGE_HIGH_TH))  # Canny 边缘检测
    return edge_img.get_statistics().l_mean()  # 返回边缘图亮度均值作为强度指标


def estimate_glare_ratio_pct(img, roi):
    """
    估计 ROI 中高光（眩光）像素占比（百分比），用于判断 Hough 参数与结果可信度。

    通过阈值检测非常亮区域并统计像素数，再除以 ROI 像素总数得到占比。
    """
    bright_blobs = img.find_blobs(
        [(GLARE_L_TH, 100, -128, 127, -128, 127)],
        roi=roi,
        x_stride=1,
        y_stride=1,
        area_threshold=1,
        pixels_threshold=1,
        merge=True,
        margin=1,
    )  # 找出亮度很高的斑块
    if not bright_blobs:
        return 0  # 无高光则占比为 0

    bright_pixels = 0
    for b in bright_blobs:
        bright_pixels += b.pixels()  # 累加高光像素数

    roi_pixels = max(1, roi[2] * roi[3])  # ROI 像素总数
    return (bright_pixels * 100) // roi_pixels  # 返回百分比（整数）


def estimate_hough_circle(img, roi, ref_cx, ref_cy, r_guess, edge_strength, glare_ratio_pct):
    """
    在给定 ROI 内运行 Hough 圆检测并返回最佳圆的直径与圆心。

    参数:
    - img: image.Image 当前帧或 ROI 子图
    - roi: (x,y,w,h) 要运行 Hough 的区域
    - ref_cx, ref_cy: 参考圆心（用于评分偏好中心接近的圆）
    - r_guess: 初始半径猜测（像素）
    - edge_strength: ROI 的边缘强度指标
    - glare_ratio_pct: ROI 的眩光像素占比（百分比整数）

    返回: (diameter, center_x, center_y) 或 (None, ref_cx, ref_cy)
    """
    # 根据眩光和边缘强度选择 Hough 的阈值（经验值）
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        hough_threshold = 3300 if edge_strength >= 24 else 2900
    else:
        hough_threshold = 3000 if edge_strength >= 24 else 2650

    r_guess = max(3, r_guess)  # 最小猜测半径限制
    r_min = max(3, (r_guess * 7) // 10)  # r_min 为猜测的 70%
    r_max = min(105, (r_guess * 13) // 10)  # r_max 为猜测的 130%，并限制最大半径
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        r_max = max(r_min, (r_max * 9) // 10)  # 眩光情况下稍微收紧上界

    circles = img.find_circles(
        roi=roi,
        threshold=hough_threshold,
        x_margin=6,
        y_margin=6,
        r_margin=6,
        r_min=r_min,
        r_max=r_max,
        r_step=1,
    )  # 在 ROI 内运行 Hough 圆检测

    if not circles:
        return None, ref_cx, ref_cy  # 未检测到圆

    best = None
    best_score = None
    for c in circles:
        dx = c.x() - ref_cx
        dy = c.y() - ref_cy
        center_cost = abs(dx) + abs(dy)  # 与参考中心的距离代价
        radius_cost = abs(c.r() - r_guess)  # 与猜测半径的差距代价
        # 综合评分：更靠近参考中心、半径更接近猜测并且半径较大更优（减去 r/4 的激励）
        score = (center_cost * 2) + radius_cost - (c.r() // 4)
        if (best_score is None) or (score < best_score):
            best_score = score
            best = c

    if best is None:
        return None, ref_cx, ref_cy

    return best.r() * 2, best.x(), best.y()  # 返回直径与圆心


def estimate_tennis_diameter(img, x, y, w, h, allow_hough=True):
    """
    估计网球的直径（像素）并返回置信度（cue_conf）。

    采用两条线索：基于颜色的连通区域估计与 Hough 圆检测。两者融合后给出
    最终直径估计与置信度指标，用于距离估算与后续跟踪决策。

    参数:
    - img: image.Image 当前帧
    - x,y,w,h: 网球候选的 bbox
    - allow_hough: bool 是否允许使用 Hough 检测（周期性启用以节省计算）

    返回: (diameter_pixels, cue_conf)
    """
    ball_size = max(w, h)  # 以 bbox 的最大边作为球的近似尺寸
    if ball_size < ROI_NEAR_SWITCH:
        pad = max(FAR_ROI_PAD_MIN, (ball_size * 2) // 3)  # 近距离采用较小 pad
    else:
        pad = max(12, ball_size)  # 远距离采用更大 pad
    rx = clamp(x - pad, 0, img.width() - 1)
    ry = clamp(y - pad, 0, img.height() - 1)
    rw = clamp(w + (pad * 2), 1, img.width() - rx)
    rh = clamp(h + (pad * 2), 1, img.height() - ry)
    roi = (rx, ry, rw, rh)  # 扩展 ROI 用于颜色与边缘检测

    cx = x + (w // 2)  # bbox 中心 x
    cy = y + (h // 2)  # bbox 中心 y
    bbox_d = max(w, h)  # bbox 的直径近似

    color_d, color_cx, color_cy = estimate_color_blob(img, roi, cx, cy, bbox_d)  # 颜色法估计

    hough_d = None
    if allow_hough:
        # Hough 圆检测参数会受边缘强度与眩光影响，因此先估计这些指标
        edge_strength = estimate_edge_strength(img, roi)
        glare_ratio_pct = estimate_glare_ratio_pct(img, roi)
        if color_d is not None:
            hough_guess = max(color_d // 2, bbox_d // 2)  # 如果有颜色估计则用它做初始猜测
        else:
            hough_guess = max(5, (bbox_d * 8) // 10)  # 否则用 bbox 值作为猜测
        hough_d, _, _ = estimate_hough_circle(
            img, roi, color_cx, color_cy, hough_guess, edge_strength, glare_ratio_pct
        )  # 运行 Hough 并获取直径
    else:
        glare_ratio_pct = 0

    # 将 color 与 hough 两种测量融合：若两者接近则取平均并给予高置信度；否则取较大的一方并降低置信度
    cue_conf = 0
    if (hough_d is not None) and (color_d is not None):
        if abs(hough_d - color_d) <= 10:
            d = ((hough_d * 5) + (color_d * 5)) // 10  # 两者接近时取平均
            cue_conf = 2
        else:
            d = max(hough_d, color_d)  # 差异较大时取更大的值以防低估
            cue_conf = 1
    elif hough_d is not None:
        d = hough_d
        cue_conf = 1
    elif color_d is not None:
        d = color_d
        cue_conf = 1
    else:
        # 两种方法均失败时基于 bbox 粗估
        d = int((bbox_d * 13) // 10)
        cue_conf = 0

    d = clamp(d, 8, 210)  # 限制直径范围

    # 针对小球且存在较强眩光的情况，对估计进行上界限制以避免被高光误判为大直径
    if (ball_size <= SMALL_BALL_SIZE_TH) and (glare_ratio_pct >= GLARE_RATIO_TH_PCT):
        glare_cap = max(8, (bbox_d * GLARE_DIAMETER_CAP_PCT) // 100)
        if d > glare_cap:
            d = glare_cap
        cue_conf = min(cue_conf, 1)

    return d, cue_conf


def estimate_ball_radius(w, h):
    """
    基于 bbox 宽高粗估球半径（像素）。

    对于较大面积的 bbox 给予额外补偿，最后 clamp 到合理取值范围。
    """
    mx = max(w, h)
    r = ((mx * 13) + 10) // 20
    area = w * h
    if area >= 900:
        r += 3
    elif area >= 400:
        r += 2
    return clamp(r, 4, 55)


def fuse_tennis_radius(w, h, detected_diameter):
    """
    将 Hough/颜色检测得到的直径与 bbox 估计的半径融合为最终绘制/跟踪使用的半径。

    - 对较大检测框取两者较大值，以避免低估；对较小框略微倾向 bbox 估计以防噪声
    - 使用 DRAW_RADIUS_SCALE 做微调并 clamp 到安全范围
    """
    circle_r = detected_diameter // 2
    bbox_r = estimate_ball_radius(w, h)
    if max(w, h) >= 20:
        r = max(circle_r, bbox_r)
    else:
        r = max(circle_r, (bbox_r * 9) // 10)
    r = int((r * DRAW_RADIUS_SCALE) + 0.5)
    return clamp(r, 4, 105)


def smooth_ball_radius(curr_r, prev_r):
    """
    对球半径做速度限制和平滑，防止单帧噪声导致半径突变。

    - 若两帧差异很小直接保留上一帧值
    - 限制上升/下降步长（随当前半径大小调整）
    - 最后做加权平均平滑
    """
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
    """
    将新的测量值追加到滑动窗口历史并保证长度上限。

    返回新的列表副本，不会修改传入的 history 引用（函数式风格）。
    如果 value 为 None 则返回现有历史拷贝或空列表。
    """
    if value is None:
        return list(history) if history else []

    values = list(history) if history else []
    values.append(int(value))
    if len(values) > MEASURE_WINDOW_LEN:
        values.pop(0)
    return values


def trimmed_window_mean(values):
    """
    对测量窗口做截尾均值（trimmed mean）：删除两端的若干极值后求均值。

    - 若数据量未达到窗口长度则直接返回均值
    - MEASURE_TRIM_COUNT 指定两端需要截掉的元素个数
    返回整数近似值
    """
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
    """
    将新样本加入目标的测量历史并返回经过截尾均值过滤后的直径与半径。

    该函数会更新 target 的元数据字段：
    - diameter_history, radius_history
    - refined_diameter, refined_radius
    返回值为 (filtered_diameter, filtered_radius)
    """
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
    """
    分配一个全局唯一的目标 id（自增），用于在多帧跟踪中保持目标一致性。
    """
    global next_target_id

    target_id = next_target_id
    next_target_id += 1
    return target_id


def ensure_target_id(target, forced_id=None):
    """
    确保目标字典包含合法的 `id` 字段。

    - 如果传入 forced_id 则强制覆盖
    - 否则若无 id 则分配新的 id
    返回目标 id 或 None（当 target 为 None 时）。
    """
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
    """
    将 row/col 格式化为字符串 "(r,c)"，用于显示与日志。
    """
    if row is None or col is None:
        return "(?,?)"
    return "(%d,%d)" % (row, col)


def target_distance_cm(target):
    """
    获取目标的距离（cm），若不可用返回 0.0，便于上层统一处理数值。
    """
    if target is None:
        return 0.0
    dist_cm = target.get("dist_cm")
    if dist_cm is None:
        return 0.0
    return float(dist_cm)


def is_live_target(target):
    """
    判断目标是否为“活跃”目标：存在且最近一帧未丢失（miss == 0）。
    """
    return target is not None and target.get("miss", 0) == 0


def is_racket_linked_to_player(player_target, racket_target):
    """
    判断给定的 racket_target 是否可能属于 player_target（球拍是否在球员附近）。

    判定规则：
    - player 和 racket 都需要包含基本 bbox 字段
    - 计算允许的 x/y margin（相对 player bbox 的比例与最小值），并判断 racket 中心是否在扩展 bbox 内
    - 排除过大尺寸的 racket（通常说明误检）
    """
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
    """
    从 racket 候选中选择最合适的目标。

    参数:
    - candidates: list of dict 候选 racket 列表
    - player_target: dict 或 None，如果提供则优先选择与 player 关联的 racket

    返回: 复制后的候选字典或 None
    """
    if not candidates:
        return None  # 没有候选则返回 None

    if is_live_target(player_target):
        linked_candidates = []  # 若存在 player，则优先选择与 player 关联的 racket
        for cand in candidates:
            if is_racket_linked_to_player(player_target, cand):
                linked_candidates.append(cand)  # 判断是否靠近 player
        candidates = linked_candidates
        if not candidates:
            return None  # 若没有与 player 关联的候选则返回 None

    best = None
    best_key = None
    for cand in candidates:
        center_dx = cand["cx"] - 160  # 相对于画面中心的 x 偏差
        center_dy = cand["cy"] - 120  # 相对于画面中心的 y 偏差
        center_d2 = (center_dx * center_dx) + (center_dy * center_dy)  # 距离的平方
        area = cand["w"] * cand["h"]  # 候选 bbox 面积
        if is_live_target(player_target):
            player_dx = abs(cand["cx"] - player_target["cx"])  # 与 player 中心的 x 距离
            player_dy = abs(cand["cy"] - player_target["cy"])  # 与 player 中心的 y 距离
            player_d = player_dx + player_dy  # 与 player 的曼哈顿距离作为惩罚项
        else:
            player_d = 0
        key = (-int(cand["score"] * 100), player_d, center_d2, -area)  # 排序键：优先 score、靠近 player、靠近中心、面积大
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand

    return best.copy()  # 返回候选的拷贝以避免外部修改原对象


def uart_write_line(line):
    """
    向 UART 端口写入一行文本并追加换行符。

    参数:
    - line: str 要发送的文本（不含换行）

    安全性: 如果未初始化或写入失败则捕获异常并打印错误。
    """
    if uart is None:
        return  # UART 未初始化则不发送

    try:
        uart.write(line)  # 写入文本（不含换行）
        uart.write("\n")  # 手动写入换行符作为行分隔
    except Exception as err:
        sys.print_exception(err)  # 写入异常则打印但不抛出


def choose_uart_ball_target(mode_name, tennis_candidates, active_target):
    """
    根据 UART/外部命令需要选择一个待操作的网球目标（仅在 PICK 模式下生效）。

    参数:
    - mode_name: str 当前模式名称（"PLAY"/"SEEK" 等）
    - tennis_candidates: list 候选网球
    - active_target: dict 当前激活目标（可能为 tennis）

    返回: 选定的 tennis 候选或 None
    """
    if mode_name == "PLAY":
        return None  # PLAY 模式不通过 UART 选择球目标

    nearest_target = choose_nearest_tennis(tennis_candidates)  # 优先选择最近的候选
    if nearest_target is not None:
        return nearest_target

    if is_live_target(active_target) and active_target.get("kind") == "tennis":
        return active_target  # 若当前激活目标为 tennis 且有效则返回

    return None


def send_runtime_packets(mode_name, target):
    """
    发送当前运行时数据包（用于外部上报/记录）。

    格式: MODE,ROW,COL,DIST(cm)
    - 若目标为有效 tennis，则填充 row/col/distance，否则使用默认值 -1/0.0
    """
    row = -1
    col = -1
    distance_cm = 0.0

    if is_live_target(target) and target.get("kind") == "tennis":
        row = int(target.get("row", -1))  # 如果目标为网球则填充网格 row
        col = int(target.get("col", -1))  # 填充网格 col
        distance_cm = target_distance_cm(target)  # 获取距离

    uart_write_line("%s,%d,%d,%.1f" % (mode_name, row, col, distance_cm))  # 发送运行时数据行


def send_command_packet(cmd, arg, mode_name, state_name):
    """简单的命令封装发送，避免未定义时崩溃。"""
    try:
        if uart is None:
            return  # 无 UART 则跳过发送
        # 格式：CMD,ARG,MODE,STATE
        line = "%s,%s,%s,%s" % (str(cmd), str(arg), str(mode_name), str(state_name))
        uart_write_line(line)
    except Exception:
        pass  # 忽略异常以提高健壮性


def age_and_prune_tracks():
    global tracked_tennis
    """
    对当前 tracked_tennis 进行老化处理：当连续未匹配到新候选时增加 miss 计数，超过阈值则放弃目标。

    返回: 若仍追踪则返回 tracked_tennis，否则返回 None
    """
    if tracked_tennis is not None:
        tracked_tennis["miss"] += 1  # 未匹配到新候选时增加 miss 计数
        if tracked_tennis["miss"] > TRACK_MAX_MISS:
            tracked_tennis = None  # 超过阈值则放弃跟踪目标
    return tracked_tennis


def match_tennis_track(candidates):
    global tracked_tennis
    """
    将新的 tennis 候选与当前 tracked_tennis 进行匹配与更新。逻辑包括：
    - 若无候选则老化当前 tracked_tennis
    - 若无 tracked_tennis 则选取最优候选并初始化跟踪
    - 否则在候选中寻找匹配（基于距离门限与代价函数），并平滑更新 tracked_tennis

    返回: 更新后的 tracked_tennis 或 None
    """
    if not candidates:
        return age_and_prune_tracks()  # 无新候选则老化并可能丢弃当前跟踪

    if tracked_tennis is None:
        # 初始分配：优先选择最近的目标，否则按面积和 score 选最大值
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
        ensure_target_id(tracked_tennis)  # 分配或确保 id
        tracked_tennis["miss"] = 0  # 命中计数重置
        return tracked_tennis

    best = None
    best_cost = None
    gate = max(TRACK_GATE_MIN, tracked_tennis["radius"] * 3)  # 距离门限（像素）
    gate2 = gate * gate

    for cand in candidates:
        dx = cand["cx"] - tracked_tennis["cx"]
        dy = cand["cy"] - tracked_tennis["cy"]
        d2 = (dx * dx) + (dy * dy)
        if d2 > gate2:
            continue  # 超出门限视为非同一目标
        cost = d2 - (cand["w"] * cand["h"]) - int(cand["score"] * 50)  # 代价函数：靠近且面积大且 score 高更优
        if (best_cost is None) or (cost < best_cost):
            best_cost = cost
            best = cand

    if best is None:
        return age_and_prune_tracks()  # 未找到匹配则老化并可能丢弃

    # 更新 tracked_tennis 的状态，使用平滑函数减少抖动
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
    tracked_tennis["miss"] = 0  # 命中后重置 miss
    return tracked_tennis


def choose_player_target(candidates):
    global tracked_player
    """
    从 player 候选中挑选最合适的目标并保持 tracked_player（支持保持 id 连续性）。

    返回: tracked_player（可能为 None）
    """
    if not candidates:
        if tracked_player is not None:
            tracked_player["miss"] += 1  # 没有候选则老化 tracked_player
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
        key = (-int(cand["score"] * 1000), -area, center_d2)  # 优先 score，再大面积，再靠中心
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand

    previous_id = 0
    if tracked_player is not None:
        previous_id = int(tracked_player.get("id", 0))

    tracked_player = best.copy()  # 记录新的 tracked_player
    if previous_id > 0:
        ensure_target_id(tracked_player, previous_id)  # 保持原有 id
    else:
        ensure_target_id(tracked_player)  # 分配新 id
    tracked_player["miss"] = 0
    return tracked_player


def draw_active_target(img, target):
    """
    在图像上绘制当前激活目标的视觉标记（圆、bbox、十字、距离等信息）。

    参数:
    - img: image.Image 要绘制到的图像
    - target: dict 或 None，包含 cx/cy/radius/w/h/dist_cm/measure_src 等字段
    """
    if target is None or target.get("miss", 0) > 0:
        img.draw_string(2, 74, "target:search", color=YELLOW, mono_space=False)  # 无有效目标
        return

    cx = clamp(target["cx"], 0, img.width() - 1)  # 目标中心 x
    cy = clamp(target["cy"], 0, img.height() - 1)  # 目标中心 y
    radius = clamp(target["radius"], 8, 50)  # 绘制时的半径限制
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
        img.draw_rectangle((target["x"], target["y"], target["w"], target["h"]), color=target_color, thickness=2)  # 绘制 bbox
    else:
        img.draw_circle((cx, cy, radius + 3), color=target_color)  # 绘制标记圆
    img.draw_cross(cx, cy, color=WHITE, size=10, thickness=2)  # 绘制中心十字
    img.draw_line((img.width() // 2, img.height() // 2, cx, cy), color=target_color)  # 从图像中心指向目标

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
    """
    在给定 y 坐标处居中绘制文本（带阴影），scale 用于放大字号。
    """
    char_w = 8 * scale  # 单字符宽度估计（像素）
    text_w = len(text) * char_w  # 文本宽度估计
    x = max(0, (img.width() - text_w) // 2)  # 居中起始 x

    img.draw_string(x + 2, y + 2, text, color=(0, 0, 0), scale=scale, mono_space=True)  # 投影阴影效果
    img.draw_string(x, y, text, color=color, scale=scale, mono_space=True)  # 正文


def draw_seek_tennis_candidates(img, mode_name, state_name, tennis_candidates):
    """
    在 SEEK/SCAN 模式下绘制所有 tennis 候选的圆标记，便于调试观察。
    """
    if mode_name != "SEEK":
        return
    if state_name != "SCAN":
        return

    for cand in tennis_candidates:
        cx = clamp(cand["cx"], 0, img.width() - 1)
        cy = clamp(cand["cy"], 0, img.height() - 1)
        radius = clamp(cand["radius"], 8, 50)
        img.draw_circle((cx, cy, radius), color=GREEN, thickness=2)  # 在扫描模式下绘制每个候选


def draw_player_countdown(img, mode_name, racket_present):
    """
    在 PLAY 模式且 player 被锁定且有 racket 的情况下绘制倒计时大号数字。
    """
    if mode_name != "PLAY":
        return
    if not player_locked:
        return
    if not racket_present:
        return
    if player_lock_start_ms <= 0:
        return

    elapsed_ms = time.ticks_diff(time.ticks_ms(), player_lock_start_ms)  # 已锁定时间
    remain_ms = PLAYER_DETECT_COUNTDOWN_MS - elapsed_ms  # 剩余 ms
    remain_s = max(0, (remain_ms + 999) // 1000)  # 四舍五入到秒
    text = str(remain_s)
    text_h = 10 * COUNTDOWN_FONT_SCALE
    y = max(0, (img.height() - text_h) // 2)
    draw_centered_text(img, text, y, YELLOW, scale=COUNTDOWN_FONT_SCALE)  # 居中绘制倒计时


def update_servo_tracking(target, img):
    """
    根据目标在图像中的偏差计算 PID 输出并更新 pan/tilt 舵机角度。

    副作用: 通过 apply_pan_delta/apply_tilt_delta 修改全局 pan_angle/tilt_angle。
    若未启用舵机或目标无效则不执行任何操作。
    """
    if not ENABLE_SERVOS:
        return
    if target is None:
        return
    if p1 is None or p9 is None:
        return
    if target.get("miss", 0) > 0:
        return

    pan_error = target["cx"] - (img.width() / 2)  # 目标相对于中心的水平误差（像素）
    tilt_error = target["cy"] - (img.height() / 2)  # 垂直误差

    if -SERVO_DEADBAND < pan_error < SERVO_DEADBAND:
        pan_error = 0  # 小于死区则忽略
    if -SERVO_DEADBAND < tilt_error < SERVO_DEADBAND:
        tilt_error = 0

    pan_output = pan_pid.get_pid(pan_error, 1) / 2  # PID 输出（水平），缩放一半以更平滑
    tilt_output = tilt_pid.get_pid(tilt_error, 1)  # PID 输出（垂直）

    apply_pan_delta(-pan_output, TRACK_PAN_MAX_STEP)  # 反向调整 pan（像素到角度映射在 PID 内）
    apply_tilt_delta(tilt_output, TRACK_TILT_MAX_STEP)


def update_tilt_tracking(target, img):
    """
    仅基于垂直误差调整 tilt（用于某些扫描或局部跟踪场景）。
    """
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
    """
    以固定步长朝向目标 tilt_angle 平滑移动，并在末尾 clamp 到机械边界。
    """
    if tilt_angle < (target_angle - step):
        tilt_angle += step  # 向目标角度增加步长
    elif tilt_angle > (target_angle + step):
        tilt_angle -= step  # 向目标角度减少步长
    else:
        tilt_angle = target_angle  # 足够接近时直接置为目标

    tilt_angle = clamp(tilt_angle, tilt_angle_limit[0], tilt_angle_limit[1])  # 限制在机械边界内


def move_pan_toward(target_angle, step=SCAN_PAN_STEP):
    global pan_angle
    """
    以固定步长朝向目标 pan_angle 平滑移动，并在末尾 clamp 到机械边界。
    """
    if pan_angle < (target_angle - step):
        pan_angle += step
    elif pan_angle > (target_angle + step):
        pan_angle -= step
    else:
        pan_angle = target_angle

    pan_angle = clamp(pan_angle, pan_angle_limit[0], pan_angle_limit[1])


def apply_pan_delta(delta, max_step=TRACK_PAN_MAX_STEP):
    global pan_angle
    """
    将限制后的 delta 应用于全局 `pan_angle`，并在应用后 clamp 到安全范围。

    参数:
    - delta: float 希望调整的角度变化
    - max_step: float 单次最大允许变化
    """
    delta = clamp(delta, -max_step, max_step)  # 限制单次调整量
    pan_angle = clamp(pan_angle + delta, pan_angle_limit[0], pan_angle_limit[1])


def apply_tilt_delta(delta, max_step=TRACK_TILT_MAX_STEP):
    global tilt_angle
    """
    将限制后的 delta 应用于全局 `tilt_angle`，并在应用后 clamp 到安全范围。
    """
    delta = clamp(delta, -max_step, max_step)
    tilt_angle = clamp(tilt_angle + delta, tilt_angle_limit[0], tilt_angle_limit[1])


def update_servo_init():
    global servo_init_frames_remaining
    if not ENABLE_SERVOS:
        return False
    if servo_init_frames_remaining <= 0:
        return False
    """
    舵机开机回中逻辑：在启动阶段按时间窗逐步启用 PWM 并缓慢将 pan/tilt 回到初始角度。

    返回: bool 若仍在回中阶段则返回 True，回中结束返回 False。
    """

    start_pan_pwm()  # 启动 pan PWM
    if servo_init_frames_remaining <= (SERVO_INIT_HOLD_FRAMES - SERVO_PWM_STAGGER_FRAMES):
        start_tilt_pwm()  # 延迟启动 tilt PWM 以避免同时启动引起抖动

    move_pan_toward(PAN_INIT_ANGLE, SERVO_INIT_STEP)  # 缓慢回中 pan
    move_tilt_toward(TILT_INIT_ANGLE, SERVO_INIT_STEP)  # 缓慢回中 tilt
    servo_init_frames_remaining -= 1
    return True


def scan_rank_key(target):
    """
    为 scan_rank 排序生成键值：优先距离（近更优），其次面积、score。
    返回一个可用于 list.sort(key=...) 的元组。
    """
    distance = target.get("dist_cm")
    if distance is None:
        distance = 9999.0  # 无距离信息则设为很远
    area = target["w"] * target["h"]
    return (distance, -area, -int(target.get("score", 0) * 1000))  # 排序键：近更优、面积更大更优、score 高更优


def scan_observation_key(target):
    """
    生成单次观察（observation）优先级比较键。

    返回一个三元组用于排序：
    - 第一项：目标中心到画面中心的曼哈顿距离（center_dx + center_dy），越小越优先。
    - 第二项：得分的负值（-score*1000），使得 score 越大排序越靠前。
    - 第三项：调用 `scan_rank_key(target)` 得到的距离指标（作为额外的近距离偏好）。

    该键适用于在同一扫描回合内对观察到的目标按优先级排序（例如选取最值得关注的目标）。
    """
    center_dx = abs(target["cx"] - 160)
    center_dy = abs(target["cy"] - 120)
    return ((center_dx + center_dy), -int(target.get("score", 0) * 1000), scan_rank_key(target)[0])


def is_same_scan_target(saved_target, new_target):
    """
    判断两个记录的扫描目标是否代表同一物理目标，依据网格位置、舵机角度与距离。
    返回 True/False。
    """
    if saved_target is None or new_target is None:
        return False

    row_diff = abs(saved_target["row"] - new_target["row"])  # 网格行差
    col_diff = abs(saved_target["col"] - new_target["col"])  # 网格列差
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

    """
    从 `scan_ranked_tennis` 中选择指定索引的候选作为当前最佳扫描目标。

    副作用：
    - 将全局 `scan_candidate_index` 设为传入的 `index`。
    - 将全局 `best_scan_tennis` 设为所选候选的拷贝并确保其拥有 `id` 字段。

    返回：
    - True 表示选择成功并更新了全局变量。
    - False 表示索引越界，`best_scan_tennis` 被置为 None。
    """

    if index < 0 or index >= len(scan_ranked_tennis):
        best_scan_tennis = None
        return False

    scan_candidate_index = index
    best_scan_tennis = scan_ranked_tennis[index].copy()
    ensure_target_id(best_scan_tennis)
    return True


def remember_scan_target(target):
    """
    将观察到的 target 记录进 scan_ranked_tennis（若为新目标则追加，为已存在则更新优先级）。
    并自动维护排序与选择第 1 个候选。
    """
    global best_scan_tennis

    if target is None:
        return
    if target.get("miss", 0) > 0:
        return
    if target.get("dist_cm") is None:
        return

    scan_target = target.copy()
    scan_target["servo_pan"] = pan_angle  # 记录扫描时的舵机角度
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
                ensure_target_id(scan_ranked_tennis[matched_index], prev_id)  # 保留先前 id
    else:
        scan_ranked_tennis.append(scan_target)  # 新目标加入候选列表

    scan_ranked_tennis.sort(key=scan_rank_key)  # 按优先级排序
    select_scan_candidate(0)  # 选第一个作为当前候选


def begin_scan_round():
    """
    进入一次新的扫描回合，重置与扫描相关的全局状态变量以开始新的观察序列。
    """
    global pick_substate, scan_seek_left, scan_direction, scan_lock_count
    global nearest_switch_count, best_scan_tennis, tracked_tennis
    global scan_ranked_tennis, scan_candidate_index, pick_confirm_start_ms, pick_track_start_ms

    pick_substate = PICK_SCAN  # 进入扫描子状态
    scan_seek_left = True
    scan_direction = 1
    scan_lock_count = 0
    nearest_switch_count = 0
    best_scan_tennis = None
    scan_ranked_tennis = []  # 清空候选
    scan_candidate_index = 0
    tracked_tennis = None
    pick_confirm_start_ms = 0
    pick_track_start_ms = 0


def update_global_scan(img, nearest_tennis):
    """
    更新用于全局扫描的舵机动作逻辑：
    - 在初始阶段将 pan 移到左边界并回中 tilt
    - 在扫描过程中以固定步长横扫并记录观察到的 tennis
    返回: scan 是否完成（到达右边界）
    """
    global scan_seek_left, pan_angle, tilt_angle

    if not ENABLE_SERVOS:
        if nearest_tennis is not None:
            remember_scan_target(nearest_tennis)  # 无舵机时仅记录目标
        return len(scan_ranked_tennis) > 0

    if scan_seek_left:
        move_pan_toward(pan_angle_limit[0], SCAN_PAN_STEP)  # 先把 pan 移到左边界
        move_tilt_toward(SCAN_TILT_TARGET)
        if abs(pan_angle - pan_angle_limit[0]) <= SCAN_ALIGN_MARGIN:
            scan_seek_left = False
            return False
        return False

    apply_pan_delta(SCAN_PAN_STEP, SCAN_PAN_STEP)  # 向右平移小步
    if nearest_tennis is None:
        move_tilt_toward(SCAN_TILT_TARGET)
    else:
        update_tilt_tracking(nearest_tennis, img)  # 若看到目标则用 tilt 跟踪
        if tilt_angle < SCAN_TILT_FLOOR:
            tilt_angle = SCAN_TILT_FLOOR
        remember_scan_target(nearest_tennis)  # 记住观察到的目标

    if pan_angle >= (pan_angle_limit[1] - SCAN_ALIGN_MARGIN):
        pan_angle = pan_angle_limit[1]
        return True  # 扫描完成（到达右边界）
    return False


def update_return_to_saved_target():
    """
    将舵机朝保存的最佳扫描目标返回，直到舵机角度与保存角度对齐。
    返回: 对齐完成则返回 True
    """
    if best_scan_tennis is None:
        return True

    move_pan_toward(best_scan_tennis.get("servo_pan", PAN_INIT_ANGLE), RETURN_PAN_STEP)  # 返回保存角度
    move_tilt_toward(best_scan_tennis.get("servo_tilt", TILT_INIT_ANGLE), RETURN_TILT_STEP)

    pan_ok = abs(pan_angle - best_scan_tennis.get("servo_pan", PAN_INIT_ANGLE)) <= RETURN_LOCK_MARGIN
    tilt_ok = abs(tilt_angle - best_scan_tennis.get("servo_tilt", TILT_INIT_ANGLE)) <= RETURN_LOCK_MARGIN
    return pan_ok and tilt_ok  # 同步到保存角度则返回 True


def choose_confirmed_scan_target(saved_target, candidates):
    """
    在新一帧的候选中寻找与之前保存的 saved_target 相匹配的目标（更严格的门限）。
    若匹配则返回附带原 id/servo 坐标的匹配副本。
    """
    if saved_target is None or not candidates:
        return None

    best = None
    best_key = None
    gate = max(TRACK_GATE_MIN + 8, saved_target["radius"] * 4, 36)  # 更严格的匹配门限
    gate2 = gate * gate
    saved_dist = saved_target.get("dist_cm")

    for cand in candidates:
        dx = cand["cx"] - saved_target["cx"]
        dy = cand["cy"] - saved_target["cy"]
        d2 = (dx * dx) + (dy * dy)
        if d2 > gate2:
            continue  # 超出门限则跳过

        dist_penalty = 0
        cand_dist = cand.get("dist_cm")
        if (saved_dist is not None) and (cand_dist is not None):
            dist_penalty = int(abs(saved_dist - cand_dist) * 8)  # 距离差异惩罚

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

    """
    选择下一个扫描候选（将索引加一）。

    副作用：
    - 清除当前 `tracked_tennis`（设为 None）以便切换到下一个候选。
    - 重置 `pick_confirm_start_ms`（用于重新启动确认计时）。

    返回值：调用 `select_scan_candidate(scan_candidate_index + 1)` 的布尔结果，
    表示是否成功选择到下一个候选。
    """

    tracked_tennis = None
    pick_confirm_start_ms = 0
    return select_scan_candidate(scan_candidate_index + 1)


def choose_nearest_tennis(candidates):
    """
    选择按估计距离最近（优先），若相近再按面积和 score 选择的 tennis 候选。
    返回最优候选或 None。
    """
    best = None
    best_key = None
    for cand in candidates:
        distance = cand["dist_cm"] if cand["dist_cm"] is not None else 9999.0
        key = (distance, -(cand["w"] * cand["h"]), -cand["score"])  # 距离优先，其次面积和 score
        if (best_key is None) or (key < best_key):
            best_key = key
            best = cand
    return best


def should_switch_to_nearest(current_target, nearest_tennis):
    """
    判断是否应从当前跟踪目标切换到最近的候选，依据距离差与面积等启发式规则。
    返回 True/False。
    """
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
        return True  # 最近的明显更近则切换

    current_area = current_target["w"] * current_target["h"]
    nearest_area = nearest_tennis["w"] * nearest_tennis["h"]
    if (nearest_dist <= current_dist) and (nearest_area > current_area * 2):
        return True  # 面积显著更大且不更远则切换

    return False


def update_scan_motion(img, nearest_tennis):
    """
    在扫描阶段根据 scan_direction 更新 pan/tilt 的运动，并在看到目标时优先用 tilt 跟踪。
    """
    global pan_angle, scan_direction, tilt_angle

    if not ENABLE_SERVOS:
        return
    if p1 is None:
        return

    apply_pan_delta(scan_direction * scan_speed, SCAN_PAN_STEP)  # 根据方向滑动 pan
    if pan_angle >= pan_angle_limit[1]:
        pan_angle = pan_angle_limit[1]
        scan_direction = -1  # 到边界后反向
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
    """
    切换到 PICK 模式并重置与 PICK 相关的全局状态（候选、计时器、锁定等）。
    """
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
    """
    切换到 PLAY 模式并重置与 PLAY 相关的全局状态。
    """
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
    """
    主行为状态机：根据当前 `mode`（PICK/PLAY）和子状态管理扫描/返回/确认/跟踪流程，
    并负责与 picker（硬件）交互、发起 capture 命令以及在模式间切换。

    参数:
    - img: 当前帧图像（用于某些跟踪/绘制）
    - tennis_target: 当前基于模型/跟踪匹配的 tennis 目标或 None
    - tennis_candidates: 本帧的所有 tennis 候选
    - player_target: 本帧检测到的 player 目标或 None
    - racket_candidates: 本帧检测到的 racket 候选列表

    返回: (active_target, mode_name, state_name, racket_present, racket_target, command_event)
    - active_target: 最终要跟踪/显示的目标（可能为 tennis/player）
    - mode_name/state_name: 字符串表示当前可显示的模式/子状态
    - racket_present: bool 是否检测到 racket
    - racket_target: racket 的候选或 None
    - command_event: 如需发送命令则为 (cmd,arg) 或 None
    """
    global mode, pick_substate, play_substate, capture_cmd, capture_flash_frames
    global player_locked, balls_served, play_ready_frames, racket_seen_prev
    global scan_lock_count, nearest_switch_count, best_scan_tennis, tracked_tennis
    global scan_ranked_tennis, scan_candidate_index
    global pick_confirm_start_ms, pick_track_start_ms, current_racket_id
    global scan_empty_rounds, player_lock_start_ms

    command_event = None
    racket_target = choose_racket_target(racket_candidates, player_target)  # 选择可能的 racket
    racket_visible = racket_target is not None
    if is_live_target(player_target) and racket_visible:
        play_ready_frames += 1
    else:
        play_ready_frames = 0
    racket_present = play_ready_frames >= PLAY_RACKET_CONFIRM_FRAMES  # 连续帧确认 racket 存在
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

    read_picker_feedback()  # 更新 picker 反馈状态

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
    """
    在屏幕一侧绘制运行状态面板：通信、FPS、模式、捕获/计数、舵机角度及确认/倒计时信息。

    参数:
    - img: image.Image 要绘制到的图像
    - fps: float 当前帧率
    - mode_name/state_name: str 当前运行模式与子状态
    - racket_present: bool 球拍是否被检测到（用于显示倒计时/确认信息）
    """

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
    """
    基于简化的针孔相机模型估计目标距离（cm）。

    仅使用像素直径与经验相机参数进行估算，适合粗略距离判断。
    返回 None 表示无法估算。
    """
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
    """
    对基于像素的距离估计做尺度与半径校正并返回最终估计（cm）。

    参数:
    - pixel_diameter: 像素直径
    - radius: 目标半径（像素），用于选择进一步的校正因子
    返回: float 或 None
    """
    base_distance_cm = estimate_distance(pixel_diameter)
    if base_distance_cm is None:
        return None
    corrected = base_distance_cm * DISTANCE_SCALE
    return correct_tennis_distance(corrected, radius)


def refine_tennis_target(img, target, allow_hough=None):
    global frame_index
    """
    对单个 tennis 目标进行颜色/Hough 融合精化，更新目标的半径/直径/距离等测量字段。

    在允许使用 Hough 的帧周期上运行圆检测并融合颜色法与 Hough 的结果，
    随后使用截尾均值滤波平滑历史测量并返回更新后的 target（就地修改）。
    """
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
    """
    对一组 tennis 候选批量运行精化（颜色+Hough），返回更新后的候选列表。

    若全局未启用精化则直接返回原始候选列表。
    """
    if not ENABLE_TENNIS_REFINEMENT:
        return candidates

    refined = []
    for cand in candidates:
        refined.append(refine_tennis_target(img, cand, allow_hough=allow_hough))
    return refined


def make_grayscale_image(channel, oh, ow):
    """
    将模型输出的浮点通道数据转换为 OpenMV 可用的灰度 image.Image 对象。

    参数:
    - channel: 类似二维数组的浮点数通道访问接口 (oh x ow)
    - oh, ow: 输出通道的高和宽

    返回: image.Image 灰度图对象（copy_to_fb=False）
    """
    # 优先使用 bytearray 构建缓冲以兼顾内存与速度
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
    """
    将模型原始输出转换为每个通道的检测框列表（用于上层处理）。

    参数:
    - model: ml.Model 实例（用于获得输出形状）
    - inputs: 模型输入列表（用于 ROI/缩放信息）
    - outputs: 模型原始输出张量

    返回: 每个通道的检测框列表，格式为 [[(x,y,w,h,score),...], ...]
    """
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
    """
    启动初始化流程：初始化摄像头、LCD、加载模型、初始化 UART/舵机/反馈并进入初始扫描状态。

    该函数在启动阶段执行必要的硬件和模型检查，遇到不可恢复的错误会调用 halt_with_error。
    """
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
    """
    主循环：连续抓取帧、调用模型、构建候选、执行状态机并驱动显示与舵机。

    该函数阻塞运行，包含异常保护逻辑以在出错时显示信息并短暂休眠后继续尝试恢复。
    """
    clock = time.clock()

    while True:
        # 每帧时钟滴答，更新内置的 fps 计算
        clock.tick()
        try:
            # 帧索引用于控制某些周期性运算（如 Hough 检测间隔）
            frame_index += 1

            # 先处理非视觉的外设输入：串口接收与周期性 hello 发送
            process_uart_rx()  # 非阻塞读取并解析串口消息
            try_send_hello()  # 如无链路则周期性发送 hello 握手

            # 捕获当前帧图像用于检测与显示
            img = sensor.snapshot()

            # 可视化辅助：在画面上绘制调试网格（用于定位与上报）
            draw_grid(img, GRID_ROWS, GRID_COLS, GRID_COLOR)

            # 将图像送入模型并通过回调做后处理得到预测列表
            predictions = net.predict([img], callback=fomo_post_process)

            # 初始化候选列表（网球 / player / racket）
            tennis_candidates = []
            player_candidates = []
            racket_candidates = []

            # 根据当前模式决定关心哪类目标
            detect_tennis = mode == MODE_PICK
            detect_play_targets = mode == MODE_PLAY

            # 遍历模型输出的每个通道（每个类别）
            for i, detection_list in enumerate(predictions):
                # 跳过 index 0（通常为背景或无用通道）
                if i == 0:
                    continue

                # 过滤低置信度的检测结果
                conf_th = threshold_for_class(i)
                detection_list = [d for d in detection_list if d[4] >= conf_th]
                if not detection_list:
                    continue

                # 解析标签名字并判定类别（tennis/player/racket）
                label_name = labels[i] if i < len(labels) else ("class_%d" % i)
                label_l = label_name.lower()
                is_tennis = ("tennis" in label_l) and ("player" not in label_l) and ("racket" not in label_l)
                is_player = "player" in label_l
                is_racket = "racket" in label_l

                # 根据运行模式决定是否使用该类别的检测
                if detect_tennis:
                    if not is_tennis:
                        continue
                elif detect_play_targets:
                    if (not is_player) and (not is_racket):
                        continue
                else:
                    continue

                # 遍历该类别的所有检测框并构建候选条目
                for x, y, w, h, score in detection_list:
                    # 计算检测框中心并映射到网格
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    row, col = get_grid_position(
                        center_x, center_y, img.width(), img.height(), GRID_ROWS, GRID_COLS
                    )

                    # 如果是网球候选，准备直径与距离估计的初始值并加入候选表
                    if detect_tennis and is_tennis:
                        radius = max(8, min(40, int(max(w, h) * 0.7)))  # 基于 bbox 的半径粗估
                        dist_cm = estimate_distance(max(w, h), image_width=img.width())  # 粗估距离（cm）
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

                    # player 候选条目，保持必要的字段
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

                    # racket 候选条目，保守估计半径
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

            # 可选：在扫描阶段对所有候选做精化（颜色+Hough 融合）
            refine_all_tennis = (
                ENABLE_TENNIS_REFINEMENT
                and REFINE_ALL_TENNIS_IN_PICK_SCAN
                and (mode == MODE_PICK)
                and (pick_substate == PICK_SCAN)
            )
            if refine_all_tennis:
                tennis_candidates = refine_tennis_candidates(img, tennis_candidates, allow_hough=True)

            # 是否在 SEEK_TRACK 阶段启用基于候选的跟踪匹配
            seek_track_enabled = detect_tennis and (pick_substate == PICK_TRACK)
            tennis_target = match_tennis_track(tennis_candidates) if seek_track_enabled else None
            # 若没有在全量精化阶段，则对当前匹配目标做单帧精化
            if detect_tennis and (not refine_all_tennis):
                tennis_target = refine_tennis_target(img, tennis_target)

            # 选取 player 目标（若在 PLAY 模式）
            player_target = choose_player_target(player_candidates) if detect_play_targets else None

            # 运行主状态机得到要跟踪的最终 active_target 与当前状态名
            active_target, mode_name, state_name, racket_present, racket_target, command_event = run_state_machine(
                img, tennis_target, tennis_candidates, player_target, racket_candidates
            )

            # 绘制调试辅助信息与目标图形
            draw_seek_tennis_candidates(img, mode_name, state_name, tennis_candidates)
            draw_active_target(img, active_target)
            draw_status_panel(img, clock.fps(), mode_name, state_name, racket_present)
            draw_player_countdown(img, mode_name, racket_present)

            # 发送运行时上报包（通过 UART）并处理任何命令事件
            send_runtime_packets(mode_name, active_target)
            if command_event is not None:
                send_command_packet(command_event[0], command_event[1], mode_name, state_name)

            # 最终将图像输出到 LCD（若已初始化）
            display_frame(img)

        except Exception as err:
            # 若主循环出现异常，绘制错误信息并打印异常堆栈，随后短暂休眠恢复循环
            img = sensor.snapshot()
            img.draw_string(2, 2, "RUN FAIL", color=RED, mono_space=False)
            img.draw_string(2, 20, str(err), color=YELLOW, mono_space=False)
            display_frame(img)
            sys.print_exception(err)
            time.sleep_ms(300)


try:
    # 启动流程：初始化模型/外设并进入主循环
    boot()  # 执行启动检查与初始化（摄像头、模型、LCD、UART、舵机等）
    main_loop()  # 启动主循环（阻塞运行，直到异常或重启）
except Exception as err:
    # 若在启动或主循环期间抛出未捕获异常，则打印堆栈并进入安全阻塞状态
    sys.print_exception(err)  # 将异常信息输出到串口/控制台，便于调试
    # 进入无限睡眠循环以防止 MCU 重试导致不确定行为，等待人工干预
    while True:
        time.sleep_ms(1000)
