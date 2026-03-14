#作者：淘宝龙飞智能车科技
#硬件：OPENMV4 H7 R1/R2 或 OPENMV4 H7 PLUS 和 龙飞LCD扩展板
#固件版本：4.6.20（仅支持该固件版本及以上，以下版本存在BUG，LCD和P1不能同时使用，参考：https://blog.csdn.net/weixin_45479272/article/details/145203546）
import sensor
import display,time,image
from pyb import Pin, Timer
from pid import PID

#设置P1为底盘舵机，P9为俯仰角舵机
p1 = Pin('P1', Pin.OUT_PP)
p9 = Pin('P9', Pin.OUT_PP)

#乒乓球颜色阈值（可按需更改）
red_threshold  = (67, 100, -20, 23, 32, 69)

pan_pid = PID(p=0.07, i=0, imax=90) #脱机运行或者禁用图像传输，使用这个PID
tilt_pid = PID(p=0.05, i=0, imax=90) #脱机运行或者禁用图像传输，使用这个PID

sensor.reset()  # Initialize the camera sensor.
sensor.set_pixformat(sensor.RGB565)  # or sensor.GRAYSCALE
# 屏幕尺寸由以下两处决定：
# 1. sensor.set_framesize(sensor.QVGA)：设置摄像头采集分辨率为320x240（与LCD显示区域匹配）
sensor.set_framesize(sensor.QVGA)  # 高分辨率320*240，帧数会降低，追踪响应会较缓慢卡顿
sensor.set_auto_whitebal(False)
sensor.skip_frames(10)
clock = time.clock()
# 2. lcd = display.SPIDisplay(width=240,height=320)：设置LCD显示区域为240x320（2.4寸LCD）
lcd = display.SPIDisplay(width=240,height=320)

def find_max(blobs):
    max_size=0
    for blob in blobs:
        if blob[2]*blob[3] > max_size:
            max_blob=blob
            max_size = blob[2]*blob[3]
    return max_blob

#云台舵机初始回中角度
pan_angle = 90.0
tilt_angle = 90.0

#云台舵机极限角度限制
pan_angle_limit = [30.0,150.0]
tilt_angle_limit = [30.0,150.0]

#P1引脚PWM控制底盘舵机定时器初始化配置=============================
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
#===========================================================

#P9引脚PWM控制底盘舵机定时器初始化配置=============================
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
#===========================================================

while True:
    img = sensor.snapshot()
    blobs = img.find_blobs([red_threshold])
    if blobs:
        max_blob = find_max(blobs)
        pan_error = max_blob.cx()-img.width()/2
        tilt_error = max_blob.cy()-img.height()/2

        print("pan_error: ", pan_error)

        img.draw_rectangle(max_blob.rect()) # rect
        img.draw_cross(max_blob.cx(), max_blob.cy()) # cx, cy

        pan_output=pan_pid.get_pid(pan_error,1)/2
        tilt_output=tilt_pid.get_pid(tilt_error,1)
        print("pan_output",pan_output)
        pan_angle-=pan_output
        tilt_angle+=tilt_output
        #角度限制
        if pan_angle < pan_angle_limit[0]:
            pan_angle = pan_angle_limit[0]
        elif pan_angle > pan_angle_limit[1]:
            pan_angle = pan_angle_limit[1]
        if tilt_angle < tilt_angle_limit[0]:
            tilt_angle = tilt_angle_limit[0]
        elif tilt_angle > tilt_angle_limit[1]:
            tilt_angle = tilt_angle_limit[1]


    lcd.write(img, hint=image.ROTATE_270)  # Take a picture and display the image.
