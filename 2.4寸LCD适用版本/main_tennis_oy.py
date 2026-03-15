# 网球识别 OpenMV 2.4寸LCD版本
# 作者：GitHub Copilot
# 适用于 OPENMV4 H7 PLUS + 龙飞LCD扩展板
# 固件版本建议 >= 4.6.20
# 网球追踪+舵机控制+LCD显示
import sensor
import display, time, image
from pyb import Pin, Timer
from pid import PID

# 舵机引脚定义
p1 = Pin('P1', Pin.OUT_PP)  # 水平舵机
p9 = Pin('P9', Pin.OUT_PP)  # 俯仰舵机

# 网球颜色阈值（浅绿色，需根据实际环境微调）
tennis_threshold = (40, 80, -40, -10, 20, 60)

# PID参数
pan_pid = PID(p=0.07, i=0, imax=90)
tilt_pid = PID(p=0.05, i=0, imax=90)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_auto_whitebal(False)
sensor.skip_frames(10)
clock = time.clock()
lcd = display.SPIDisplay(width=240, height=320)

def find_max(blobs):
    max_size = 0
    max_blob = None
    for blob in blobs:
        if blob[2]*blob[3] > max_size:
            max_blob = blob
            max_size = blob[2]*blob[3]
    return max_blob

# 云台舵机初始回中角度
pan_angle = 90.0
tilt_angle = 90.0

# 云台舵机极限角度限制
pan_angle_limit = [30.0, 150.0]
tilt_angle_limit = [30.0, 150.0]

# P1引脚PWM控制底盘舵机定时器初始化配置
p1_tim_pluse = Timer(12)
p1_tim_main = Timer(13, freq=50)
def P1_ISR0(t):
    p1.low()
    p1_tim_pluse.deinit()
def P1_ISR(t):
    psr = int(pan_angle*1000/9+5000) - 1
    p1.high()
    p1_tim_pluse.init(prescaler=psr, period=23)
    p1_tim_pluse.callback(P1_ISR0)
p1_tim_main.callback(P1_ISR)

# P9引脚PWM控制俯仰舵机定时器初始化配置
p9_tim_pluse = Timer(14)
p9_tim_main = Timer(15, freq=50)
def P9_ISR0(t):
    p9.low()
    p9_tim_pluse.deinit()
def P9_ISR(t):
    psr = int(tilt_angle*1000/9+5000) - 1
    p9.high()
    p9_tim_pluse.init(prescaler=psr, period=23)
    p9_tim_pluse.callback(P9_ISR0)
p9_tim_main.callback(P9_ISR)

TEXT_COLOR = (255, 255, 0)
GREEN = (0, 255, 0)

while True:
    img = sensor.snapshot()
    blobs = img.find_blobs([tennis_threshold], pixels_threshold=30, area_threshold=30, merge=True)
    if blobs:
        # 1. 获取最大色块（目标球）
        max_blob = find_max(blobs)
        # 2. 计算舵机误差：球中心与图像中心的偏差
        pan_error = max_blob.cx() - img.width() / 2
        tilt_error = max_blob.cy() - img.height() / 2

        # 3. 绘制目标球的矩形和十字中心
        img.draw_rectangle(max_blob.rect())
        img.draw_cross(max_blob.cx(), max_blob.cy())
        img.draw_string(2, 2, "Tennis Detected", color=TEXT_COLOR)

        # 4. PID控制器计算舵机输出
        pan_output = pan_pid.get_pid(pan_error, 1) / 2
        tilt_output = tilt_pid.get_pid(tilt_error, 1)

        # 5. 更新舵机角度（底盘水平/俯仰）
        pan_angle -= pan_output
        tilt_angle += tilt_output

        # 6. 舵机角度限制，防止越界
        if pan_angle < pan_angle_limit[0]:
            pan_angle = pan_angle_limit[0]
        elif pan_angle > pan_angle_limit[1]:
            pan_angle = pan_angle_limit[1]
        if tilt_angle < tilt_angle_limit[0]:
            tilt_angle = tilt_angle_limit[0]
        elif tilt_angle > tilt_angle_limit[1]:
            tilt_angle = tilt_angle_limit[1]
    else:
        img.draw_string(2, 2, "No Tennis", color=TEXT_COLOR)

    lcd.write(img, hint=image.ROTATE_270)
