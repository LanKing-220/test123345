# 网球识别 OpenMV 2.4寸LCD版本
# 作者：GitHub Copilot
# 适用于 OPENMV4 H7 PLUS + 龙飞LCD扩展板
# 固件版本建议 >= 4.6.20

import sensor, image, time, display
from pyb import Pin, Timer

tennis_threshold = (40, 80, -40, -10, 20, 60)  # 浅绿色网球阈值，需根据实际环境微调

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.set_auto_whitebal(False)
sensor.skip_frames(10)
lcd = display.SPIDisplay(width=240, height=320)
clock = time.clock()

TEXT_COLOR = (255, 255, 0)
GREEN = (0, 255, 0)

while True:
    clock.tick()
    img = sensor.snapshot()
    blobs = img.find_blobs([tennis_threshold], pixels_threshold=30, area_threshold=30, merge=True)
    if blobs:
        max_blob = max(blobs, key=lambda b: b.pixels())
        r = int((max_blob.w() + max_blob.h()) / 4)
        img.draw_circle((max_blob.cx(), max_blob.cy(), r), color=GREEN)
        img.draw_string(2, 2, f"Tennis Detected", color=TEXT_COLOR)
    else:
        img.draw_string(2, 2, "No Tennis", color=TEXT_COLOR)
    lcd.write(img, hint=image.ROTATE_270)
