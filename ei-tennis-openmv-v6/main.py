# --- 网球半径历史平滑缓存 ---
BALL_RADIUS_HISTORY_LEN = 10
ball_radius_history = []
# 分段经验修正的网球距离校正函数
def correct_tennis_distance(distance_cm, radius):
    """
    对distance_cm进行分段经验修正，radius为像素半径。
    """
    if radius >= 50:
        factor = 1.10
    if radius >= 42:
        factor = 1.15
    elif radius >= 36:
        factor = 1.20
    else:
        factor = 1.25
    return distance_cm * factor

# Edge Impulse - OpenMV FOMO 目标检测示例
# 本代码用于网球场景的目标检测、距离估算与可视化，所有注释均为中文。


import sensor, image, time, ml, math, uos, gc
import display
from pyb import Pin, Timer
from pid import PID

# 舵机引脚定义
p1 = Pin('P1', Pin.OUT_PP)   # 水平舵机
p9 = Pin('P9', Pin.OUT_PP)   # 垂直舵机
# 舵机引脚定义（可根据实际硬件调整）
pan_servo = Pin('P1', Pin.OUT_PP)   # 水平舵机
tilt_servo = Pin('P9', Pin.OUT_PP)  # 垂直舵机

# 舵机角度变量
pan_angle = 90.0   # 水平舵机初始角度
tilt_angle = 130.0  # 垂直舵机初始角度

# 舵机极限角度
pan_angle_limit = [-1800.0, 180.0]
tilt_angle_limit = [80.0, 150.0]

# PWM定时器初始化

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
#===========================================================

# P9引脚PWM控制底盘舵机定时器初始化配置
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

sensor.reset()                         # 复位并初始化摄像头
sensor.set_pixformat(sensor.RGB565)    # 设置像素格式为RGB565（或灰度）
sensor.set_framesize(sensor.QVGA)      # 设置分辨率为QVGA（240x320）
sensor.skip_frames(time=2000)          # 等待摄像头自动调整

lcd = display.SPIDisplay(width=240,height=320)
net = None
labels = None
min_confidence = 0.5


try:
    # 加载模型，若内存充足则分配到堆上
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('模型加载失败，请确认.tflite和labels.txt已复制到设备 (' + str(e) + ')')


try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('标签文件加载失败，请确认labels.txt已复制到设备 (' + str(e) + ')')


# 目标类别颜色列表（按类别顺序，RGB格式）。如需支持更多类别可扩展。
colors = [
    (255,   0,   0),   # 红色
    (  0, 255,   0),   # 绿色
    (255, 255,   0),   # 黄色
    (  0,   0, 255),   # 蓝色
    (255,   0, 255),   # 品红
    (  0, 255, 255),   # 青色
    (255, 255, 255),   # 白色
]



# 各类别置信度阈值（可根据实际模型表现调整）
THRESH_TENNIS = 0.5   # 网球置信度阈值
THRESH_PLAYER = 0.4   # 球员置信度阈值
THRESH_RACKET = 0.7   # 球拍置信度阈值
threshold_list = [(math.ceil(THRESH_TENNIS * 255), 255)]  # 用于二值化的亮度阈值

# 常用颜色常量
GREEN = (0, 255, 0)   # 绿色（网球圈用）
RED = (255, 0, 0)     # 红色（球拍框用）

# 网球目标跟踪相关
tennis_tracks = {}     # 记录每个网球的轨迹信息
next_track_id = 1      # 下一个可用的轨迹ID


# LAB颜色容差参数（影响颜色分割灵敏度）
# 光照变化大时可适当增大，误检多时可减小
COLOR_TOL_L_BASE = 12      # L通道基础容差
COLOR_TOL_A_BASE = 10      # A通道基础容差
COLOR_TOL_B_BASE = 10      # B通道基础容差
COLOR_TOL_L_EXTRA = 3      # L通道额外自适应容差
COLOR_TOL_A_EXTRA = 3      # A通道额外自适应容差
COLOR_TOL_B_EXTRA = 3.5    # B通道额外自适应容差


# 网球圆圈半径最终放大系数
# 若圈偏小可适当增大（如1.20->1.30）
DRAW_RADIUS_SCALE = 1.15   # 半径放大比例


# 远距离网球检测的自适应参数
ROI_NEAR_SWITCH = 26           # ROI切换阈值，近远场分界
FAR_ROI_PAD_MIN = 6            # 远距离时ROI最小扩展像素
MAX_COLOR_BLOB_AREA_MULT = 6   # 颜色分割最大面积系数
TRACK_MAX_MISS = 6             # 轨迹最大丢失帧数


# 边缘与圆检测参数
EDGE_LOW_TH = 55   # Canny边缘检测低阈值
EDGE_HIGH_TH = 110 # Canny边缘检测高阈值


# 强光/反光地面下小球的特殊处理参数
GLARE_L_TH = 88                # LAB L通道高亮阈值
GLARE_RATIO_TH_PCT = 18        # 高亮像素比例阈值（%）
SMALL_BALL_SIZE_TH = 24        # 小球像素尺寸阈值
GLARE_DIAMETER_CAP_PCT = 115   # 强光下直径上限百分比


# 半径锁定参数：当检测不稳定时锁定半径若干帧，抑制突变
RADIUS_LOCK_HOLD = 4       # 锁定帧数
RADIUS_JUMP_ABS = 6        # 半径突变绝对阈值
RADIUS_JUMP_RATIO_PCT = 35 # 半径突变相对百分比

def clamp(v, lo, hi):
    # 【通用工具函数】将输入v限制在[lo, hi]闭区间内，防止越界。
    # 参数：
    #   v  —— 需要限制的数值
    #   lo —— 区间下界
    #   hi —— 区间上界
    # 返回：
    #   若v小于下界，返回lo；若v大于上界，返回hi；否则返回v本身
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def match_tennis_track(cx, cy, used_track_ids, hint_size):
    # 【轨迹匹配函数】将当前检测到的网球中心(cx, cy)与历史轨迹进行最近邻匹配，返回最优轨迹ID。
    # 参数：
    #   cx, cy         —— 当前检测到的网球中心坐标（像素）
    #   used_track_ids —— 本帧已分配的轨迹ID列表，避免重复分配
    #   hint_size      —— 当前检测框的尺寸，用于动态调整匹配门限
    # 返回：
    #   best_id —— 匹配到的轨迹ID（若无合适轨迹则为None）
    best_id = None  # 最优轨迹ID
    best_d2 = None  # 最小距离平方
    gate = max(18, hint_size * 2)  # 匹配门限，防止远距离误匹配
    gate2 = gate * gate

    for tid in tennis_tracks:
        if tid in used_track_ids:
            continue  # 跳过本帧已分配的轨迹
        tx, ty, tr, miss, lock_count = tennis_tracks[tid]
        dx = cx - tx
        dy = cy - ty
        d2 = (dx * dx) + (dy * dy)  # 欧氏距离平方
        local_gate = max(gate2, (tr * tr * 4))  # 动态门限，近距离更严格
        if d2 > local_gate:
            continue  # 距离过远不匹配
        if (best_d2 is None) or (d2 < best_d2):
            best_d2 = d2
            best_id = tid

    return best_id  # 若无合适轨迹则返回None


def age_and_prune_tracks(used_track_ids):
    # 【轨迹老化与清理】对未被当前帧匹配到的轨迹，增加丢失计数，超限后删除。
    # 参数：
    #   used_track_ids —— 本帧已分配的轨迹ID列表
    # 作用：防止轨迹无限增长，及时清理丢失目标
    stale = []  # 记录需清理的轨迹ID
    for tid in list(tennis_tracks.keys()):
        if tid in used_track_ids:
            continue  # 本帧已匹配到的轨迹不处理
        tx, ty, tr, miss, lock_count = tennis_tracks[tid]
        miss += 1  # 丢失帧数+1
        if miss > TRACK_MAX_MISS:
            stale.append(tid)  # 超过最大丢失帧数，标记为过期
        else:
            tennis_tracks[tid] = (tx, ty, tr, miss, lock_count)  # 更新丢失计数

    for tid in stale:
        tennis_tracks.pop(tid)  # 删除过期轨迹


def build_tennis_color_threshold(img, x, y, w, h):
    # 【动态颜色阈值生成】根据检测框中心区域的LAB均值和方差，自适应生成颜色分割阈值。
    # 参数：
    #   img —— 当前帧图像
    #   x, y, w, h —— 检测框左上角坐标及宽高
    # 返回：
    #   (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi) —— LAB颜色空间的上下界元组
    # 步骤：
    # 1. 选取检测框中心区域作为种子区域，避免边缘干扰
    seed_w = max(6, (w * 2) // 5)  # 种子区域宽度，最小6像素
    seed_h = max(6, (h * 2) // 5)  # 种子区域高度，最小6像素
    seed_x = clamp(x + (w - seed_w) // 2, 0, img.width() - 1)  # 居中
    seed_y = clamp(y + (h - seed_h) // 2, 0, img.height() - 1)
    seed_w = clamp(seed_w, 1, img.width() - seed_x)
    seed_h = clamp(seed_h, 1, img.height() - seed_y)

    # 2. 计算种子区域的LAB均值和标准差
    s = img.get_statistics(roi=(seed_x, seed_y, seed_w, seed_h))
    l_mean = s.l_mean()
    a_mean = s.a_mean()
    b_mean = s.b_mean()
    l_std = s.l_stdev()
    a_std = s.a_stdev()
    b_std = s.b_stdev()

    # 3. 根据基础容差+自适应分量，动态调整容差范围，防止过分分割
    tol_l = int(clamp(COLOR_TOL_L_BASE + COLOR_TOL_L_EXTRA + l_std, 8, 24))
    tol_a = int(clamp(COLOR_TOL_A_BASE + COLOR_TOL_A_EXTRA + a_std, 6, 18))
    tol_b = int(clamp(COLOR_TOL_B_BASE + COLOR_TOL_B_EXTRA + b_std, 6, 18))

    # 4. 计算LAB各通道上下界，防止越界
    l_lo = int(clamp(l_mean - tol_l, 0, 100))
    l_hi = int(clamp(l_mean + tol_l, 0, 100))
    a_lo = int(clamp(a_mean - tol_a, -128, 127))
    a_hi = int(clamp(a_mean + tol_a, -128, 127))
    b_lo = int(clamp(b_mean - tol_b, -128, 127))
    b_hi = int(clamp(b_mean + tol_b, -128, 127))
    return (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi)  # 返回LAB阈值元组


def estimate_color_blob(img, roi, ref_cx, ref_cy, bbox_d):
    # 【颜色分割辅助检测】利用动态颜色阈值，在FOMO中心附近ROI内寻找最可能的网球色块。
    # 参数：
    #   img      —— 当前帧图像
    #   roi      —— 感兴趣区域(左上x, 左上y, 宽, 高)
    #   ref_cx, ref_cy —— FOMO检测中心点
    #   bbox_d   —— 检测框最大边长
    # 返回：
    #   color_d  —— 估算的等效直径（像素），若无则None
    #   best.cx(), best.cy() —— 色块中心坐标
    rx, ry, rw, rh = roi
    # 1. 以FOMO中心为种子，生成自适应颜色阈值
    thr = build_tennis_color_threshold(img, ref_cx - (rw // 6), ref_cy - (rh // 6), rw // 3, rh // 3)
    # 2. 在ROI内查找色块
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
        return None, ref_cx, ref_cy  # 未找到色块，返回原中心

    # 3. 选择最优色块：面积不过大、中心靠近FOMO中心、形状接近正方形
    best = None
    best_score = None
    max_blob_area = max(36, bbox_d * bbox_d * MAX_COLOR_BLOB_AREA_MULT)  # 最大允许面积
    for b in blobs:
        if b.pixels() > max_blob_area:
            continue  # 面积过大，通常为背景

        dx = b.cx() - ref_cx
        dy = b.cy() - ref_cy
        inside_ref = (b.x() <= ref_cx <= (b.x() + b.w())) and (b.y() <= ref_cy <= (b.y() + b.h()))

        # 形状惩罚：越接近正方形越优
        long_side = max(b.w(), b.h())
        short_side = max(1, min(b.w(), b.h()))
        ratio = (long_side * 100) // short_side
        shape_penalty = abs(ratio - 100)

        dist_cost = abs(dx) + abs(dy)  # 距离惩罚
        size_gain = b.pixels() // 6    # 面积奖励
        center_bonus = 60 if inside_ref else 0  # 中心包含奖励
        score = (dist_cost * 3) + shape_penalty - size_gain - center_bonus
        if (best_score is None) or (score < best_score):
            best_score = score
            best = b

    if best is None:
        return None, ref_cx, ref_cy

    # 4. 用等效圆直径（面积反推）和最大边长取最大，防止低估
    eq_d = int(math.sqrt((4.0 * best.pixels()) / math.pi))
    blob_d = max(best.w(), best.h())
    color_d = max(eq_d, blob_d)
    return color_d, best.cx(), best.cy()


def estimate_edge_strength(img, roi):
    # 【边缘强度估算】对ROI区域做Canny边缘检测，返回亮度均值作为边缘强度。
    # 参数：
    #   img —— 当前帧图像
    #   roi —— 感兴趣区域(左上x, 左上y, 宽, 高)
    # 返回：
    #   边缘图像的亮度均值，数值越大边缘越明显
    edge_img = img.copy(roi=roi)
    edge_img.to_grayscale()  # 转灰度
    edge_img.find_edges(image.EDGE_CANNY, threshold=(EDGE_LOW_TH, EDGE_HIGH_TH))
    return edge_img.get_statistics().l_mean()


def estimate_glare_ratio_pct(img, roi):
    # 【高亮比例估算】统计ROI区域内高亮像素占比，用于判断强光/反光干扰。
    # 参数：
    #   img —— 当前帧图像
    #   roi —— 感兴趣区域(左上x, 左上y, 宽, 高)
    # 返回：
    #   高亮像素占ROI总像素的百分比（0~100）
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
        return 0  # 无高亮像素

    bright_pixels = 0
    for b in bright_blobs:
        bright_pixels += b.pixels()

    roi_pixels = max(1, roi[2] * roi[3])
    return (bright_pixels * 100) // roi_pixels


def estimate_hough_circle(img, roi, ref_cx, ref_cy, r_guess, edge_strength, glare_ratio_pct):
    # 【霍夫圆辅助检测】在ROI内以FOMO中心为引导，利用边缘和高亮信息自适应参数，检测最优圆。
    # 参数：
    #   img —— 当前帧图像
    #   roi —— 感兴趣区域(左上x, 左上y, 宽, 高)
    #   ref_cx, ref_cy —— FOMO检测中心点
    #   r_guess —— 预估半径
    #   edge_strength —— 边缘强度
    #   glare_ratio_pct —— 高亮比例
    # 返回：
    #   (直径, 圆心x, 圆心y)，若无则None, ref_cx, ref_cy
    # 1. 根据高亮和边缘自适应调整霍夫阈值
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        hough_threshold = 3300 if edge_strength >= 24 else 2900
    else:
        hough_threshold = 3000 if edge_strength >= 24 else 2650

    r_guess = max(3, r_guess)
    r_min = max(3, (r_guess * 7) // 10)
    r_max = min(105, (r_guess * 13) // 10)
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        # 反光地面易出大圆，收紧上界
        r_max = max(r_min, (r_max * 9) // 10)

    # 2. 查找所有圆
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

    # 3. 选择最优圆：中心靠近FOMO中心，半径接近预估，略偏大优先
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


def estimate_tennis_diameter(img, x, y, w, h):
    # 【网球直径多线索估算】融合FOMO检测框、颜色分割、边缘、圆检测等多种信息，鲁棒估计网球像素直径。
    # 参数：
    #   img —— 当前帧图像
    #   x, y, w, h —— FOMO检测框左上角及宽高
    # 返回：
    #   d —— 估算的像素直径
    #   cue_conf —— 置信度（2:双线索一致，1:单线索，0:仅检测框）
    # 步骤：
    # 1. ROI自适应：远距离小球用更小ROI，近距离适当放大
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

    # 2. 颜色分割辅助估算
    color_d, color_cx, color_cy = estimate_color_blob(img, roi, cx, cy, bbox_d)
    # 3. 边缘强度与高亮比例
    edge_strength = estimate_edge_strength(img, roi)
    glare_ratio_pct = estimate_glare_ratio_pct(img, roi)

    # 4. 霍夫圆辅助估算
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
    )

    # 5. 多线索融合：
    cue_conf = 0
    if (hough_d is not None) and (color_d is not None):
        # 两线索接近则加权平均，否则取较大值
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
        d = int((bbox_d * 13) // 10)  # 仅用检测框估算
        cue_conf = 0

    d = clamp(d, 8, 210)  # 防止极端值

    # 6. 强光/远距离保护：防止反光导致半径过大
    if (ball_size <= SMALL_BALL_SIZE_TH) and (glare_ratio_pct >= GLARE_RATIO_TH_PCT):
        glare_cap = max(8, (bbox_d * GLARE_DIAMETER_CAP_PCT) // 100)
        if d > glare_cap:
            d = glare_cap
        cue_conf = min(cue_conf, 1)

    # 仅估算直径和置信度，中心点仍以FOMO为准
    return d, cue_conf


def fuse_tennis_radius(w, h, detected_diameter):
    # 【半径融合】结合圆拟合和检测框尺寸，获得更稳健的网球半径估算。
    # 参数：
    #   w, h —— 检测框宽高
    #   detected_diameter —— 多线索估算的直径
    # 返回：
    #   r —— 最终用于绘制和输出的半径
    circle_r = detected_diameter // 2  # 圆拟合半径
    bbox_r = estimate_ball_radius(w, h)  # 检测框推算半径

    # 近距离（大框）优先取较大半径，远距离以圆拟合为主但保底
    if max(w, h) >= 20:
        r = max(circle_r, bbox_r)
    else:
        r = max(circle_r, (bbox_r * 9) // 10)

    # 最终放大系数，补偿边缘/纹理导致的低估
    r = int((r * DRAW_RADIUS_SCALE) + 0.5)
    return clamp(r, 4, 105)  # 限制合理范围


def estimate_ball_radius(w, h):
    # 【检测框半径估算】仅根据检测框尺寸推算半径，近大远小。
    # 参数：w, h —— 检测框宽高
    # 返回：r —— 推算半径
    mx = max(w, h)
    r = ((mx * 13) + 10) // 20  # 约0.65倍最大边长

    # 大面积（近距离）适当补偿
    area = w * h
    if area >= 900:
        r += 3
    elif area >= 400:
        r += 2

    if r < 4:
        return 4
    if r > 55:
        return 55
    return r


def smooth_ball_radius(curr_r, prev_r):
    # 【半径平滑】对半径变化做自适应平滑，抑制抖动和突变，提升显示稳定性。
    # 参数：
    #   curr_r —— 当前帧估算半径
    #   prev_r —— 上一帧半径
    # 返回：
    #   平滑后的半径
    if prev_r is None:
        return curr_r  # 首帧直接返回

    diff = curr_r - prev_r

    # 1. 忽略微小变化，防止闪烁
    if -2 <= diff <= 2:
        return prev_r

    # 2. 自适应步长限制：靠近时增长快，远离时收缩慢
    up_step = 8 if prev_r < 24 else 12
    down_step = 8 if prev_r >= 20 else 6
    if diff > up_step:
        curr_r = prev_r + up_step
    elif diff < -down_step:
        curr_r = prev_r - down_step

    # Adaptive EMA: faster response on growth, smoother on shrink.
    if curr_r >= prev_r:
        # 65% previous + 35% current
        return ((prev_r * 13) + (curr_r * 7)) // 20

    # 65% previous + 35% current for faster shrink when ball moves farther.
    return ((prev_r * 13) + (curr_r * 7)) // 20

def fomo_post_process(model, inputs, outputs):
    # 【FOMO输出后处理】将模型输出的检测框坐标还原到原图坐标系，便于后续可视化和分析。
    # 参数：
    #   model   —— FOMO模型对象
    #   inputs  —— 输入图像及ROI信息
    #   outputs —— 模型输出张量
    # 返回：
    #   l —— 按类别分组的检测框列表，每项为(x, y, w, h, score)
    ob, oh, ow, oc = model.output_shape[0]  # 输出张量维度

    x_scale = inputs[0].roi[2] / ow  # x方向缩放
    y_scale = inputs[0].roi[3] / oh  # y方向缩放
    scale = min(x_scale, y_scale)    # 保持比例

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]  # x偏移
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]  # y偏移

    l = [[] for i in range(oc)]  # 按类别分组

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            # 坐标还原到原图
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

clock = time.clock()

GRID_ROWS = 9  # 网格行数
GRID_COLS = 12  # 网格列数
GRID_COLOR = (128, 128, 128)
TEXT_COLOR = (255, 255, 0)

def draw_dashed_line(img, x0, y0, x1, y1, color, dash_len=8, gap_len=6):
    # 【虚线绘制】仅支持水平或垂直虚线，用于网格线美化。
    # 参数：
    #   img —— 当前帧图像
    #   x0, y0, x1, y1 —— 起止坐标
    #   color —— 线条颜色
    #   dash_len —— 虚线段长度
    #   gap_len  —— 虚线间隔长度
    if x0 == x1:
        # 垂直线
        y = y0
        while y < y1:
            y_end = min(y + dash_len, y1)
            img.draw_line((x0, y, x1, y_end), color=color)
            y = y_end + gap_len
    elif y0 == y1:
        # 水平线
        x = x0
        while x < x1:
            x_end = min(x + dash_len, x1)
            img.draw_line((x, y0, x_end, y1), color=color)
            x = x_end + gap_len

def draw_grid(img, rows, cols, color):
    # 【网格绘制】在图像上绘制rows×cols的虚线网格，用于辅助定位。
    # 参数：
    #   img —— 当前帧图像
    #   rows, cols —— 网格行列数
    #   color —— 网格线颜色
    w = img.width()
    h = img.height()
    # 竖线
    for i in range(1, cols):
        x = (w * i) // cols
        draw_dashed_line(img, x, 0, x, h, color)
    # 横线
    for j in range(1, rows):
        y = (h * j) // rows
        draw_dashed_line(img, 0, y, w, y, color)

def get_grid_position(x, y, img_w, img_h, rows, cols):
    # 【网格坐标换算】将像素坐标(x, y)映射到网格(row, col)编号。
    # 参数：
    #   x, y —— 像素坐标
    #   img_w, img_h —— 图像宽高
    #   rows, cols —— 网格行列数
    # 返回：
    #   row, col —— 网格行列编号（从0开始）
    col = min(cols - 1, max(0, (x * cols) // img_w))
    row = min(rows - 1, max(0, (y * rows) // img_h))
    return row, col

FOCAL_LENGTH_MM = 2.8  # OpenMV H7 Plus镜头典型焦距（可查具体镜头参数）
TENNIS_DIAMETER_MM = 67  # 标准网球直径
SENSOR_WIDTH_MM = 4.896  # OpenMV H7 Plus OV5640传感器宽度（mm）
IMAGE_WIDTH = 320  # 你的windowing宽度

def estimate_distance(pixel_diameter, sensor_width=SENSOR_WIDTH_MM, image_width=IMAGE_WIDTH):
    # 【距离估算】根据成像原理，利用像素直径反推网球到摄像头的距离。
    # 参数：
    #   pixel_diameter —— 检测到的网球像素直径
    #   sensor_width   —— 传感器宽度（mm）
    #   image_width    —— 图像宽度（像素）
    # 返回：
    #   D —— 估算距离（mm），若输入异常返回-1
    mm_per_pixel = sensor_width / image_width  # 单像素对应的实际长度
    h_mm = pixel_diameter * mm_per_pixel       # 网球在传感器上的实际成像长度
    if h_mm == 0:
        return -1  # 防止除零
    D = (FOCAL_LENGTH_MM * TENNIS_DIAMETER_MM) / h_mm  # 成像公式
    return D  # 单位：mm

# ==================== 新增：状态机定义 ====================
# 模式
MODE_PICK = 0          # 捡球模式
MODE_PLAY = 1          # 对打模式
mode = MODE_PICK

# 捡球模式子状态
PICK_SCAN = 0          # 扫描
PICK_TRACK = 1         # 跟踪锁定网球
pick_substate = PICK_SCAN

# 对打模式子状态
PLAY_TRACK_PLAYER = 0  # 追踪人物
PLAY_WAIT_SERVE = 1    # 等待发球（检测到球拍）
play_substate = PLAY_TRACK_PLAYER

# 控制标志
capture_cmd = 0                # 捕获指令（捡球模式）
player_locked = False          # 人物是否锁定
balls_served = 0               # 已发球计数（模拟）
target_balls = 5               # 假设需要发出5个球（可动态调整）
locked_player_cx = 0            # 锁定人物的中心x（用于跟踪）
locked_player_cy = 0            # 锁定人物的中心y
locked_tennis_id = None         # 当前锁定的网球轨迹ID

# 扫描相关
scan_direction = 1              # 1: 增大角度， -1: 减小角度
scan_speed = 2.0                 # 扫描速度（度/帧）
scan_phase = 0                   # 扫描阶段（0:未完成，1:完成一次扫描）
best_scan_tennis = None          # 扫描过程中发现的最佳网球（距离最小）
# 格式: {'distance': dist, 'track_id': tid, 'cx':cx, 'cy':cy}

# PID控制器（从简单示例中移植）
pan_pid = PID(p=0.07, i=0, imax=90)   # 水平PID
tilt_pid = PID(p=0.05, i=0, imax=90)  # 俯仰PID
# =========================================================

while(True):
    clock.tick()

    img = sensor.snapshot()
    # 先画网格线，保证任何情况下都显示
    draw_grid(img, GRID_ROWS, GRID_COLS, GRID_COLOR)
    used_track_ids = []

    output_info = []
    tennis_id_counter = 1  # Tennis球id递增

    # 检测结果分类存储
    tennis_detections = []   # 存放 (x,y,w,h,score,cx,cy,distance_cm,track_id) 用于后续处理
    player_detections = []   # 存放 (x,y,w,h,score,cx,cy)
    racket_detections = []   # 存放 (x,y,w,h,score,cx,cy)

    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        # 动态调整各类别置信度
        if i == 1:
            conf_th = THRESH_TENNIS
        elif i == 2:
            conf_th = THRESH_PLAYER
        elif i == 3:
            conf_th = THRESH_RACKET
        else:
            conf_th = 0.5
        detection_list = [d for d in detection_list if d[4] >= conf_th]
        if i == 0: continue  # background class
        if len(detection_list) == 0: continue  # no detections for this class?

        print("********** %s **********" % labels[i])
        for x, y, w, h, score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            # 计算球的网格位置
            row, col = get_grid_position(center_x, center_y, img.width(), img.height(), GRID_ROWS, GRID_COLS)
            pos_text = f"Grid: ({row},{col})"
            info = {}
            info['kind'] = labels[i]
            label_l = labels[i].lower()  # 修复未定义
            # 对Tennis球id递增编号，其它类别保持原样
            if ("tennis" in label_l) and ("racket" not in label_l) and ("player" not in label_l):
                info['id'] = f"{tennis_id_counter:02d}"
                tennis_id_counter += 1
            else:
                info['id'] = f"{i:02d}"

            if ("tennis" in label_l) and ("racket" not in label_l) and ("player" not in label_l):
                detected_diameter, cue_conf = estimate_tennis_diameter(img, x, y, w, h)
                radius = fuse_tennis_radius(w, h, detected_diameter)
                # 半径历史平滑：每10帧去极值后取均值
                ball_radius_history.append(radius)
                if len(ball_radius_history) > BALL_RADIUS_HISTORY_LEN:
                    ball_radius_history.pop(0)
                smooth_radius = radius
                if len(ball_radius_history) == BALL_RADIUS_HISTORY_LEN:
                    sorted_r = sorted(ball_radius_history)
                    trimmed = sorted_r[1:-1]  # 去掉最大最小
                    if trimmed:
                        smooth_radius = int(sum(trimmed) / len(trimmed))
                else:
                    smooth_radius = int(sum(ball_radius_history) / len(ball_radius_history))

                # 距离估算
                distance_mm = estimate_distance(detected_diameter)
                if distance_mm > 0:
                    distance_cm = (distance_mm / 10) * 2.15
                    # 分段经验修正
                    distance_cm = correct_tennis_distance(distance_cm, radius)
                    dist_text = "Dist: %.1fcm" % distance_cm
                    pos_cm = f"({center_x},{center_y})"
                    info['pos_cm'] = pos_cm
                    info['distance_cm'] = round(distance_cm, 1)
                else:
                    dist_text = "Dist: -"
                    info['pos_cm'] = 0
                    info['distance_cm'] = 0
                info['radius'] = radius
                info['g_id'] = f"({row},{col})"

                # FOMO is the only source for tennis recognition and center.
                assist_cx = center_x
                assist_cy = center_y

                # 用平滑后的半径画圈，保证显示更稳定
                # 限制圆心和半径，防止超出边界
                safe_cx = clamp(assist_cx, smooth_radius, img.width() - smooth_radius)
                safe_cy = clamp(assist_cy, smooth_radius, img.height() - smooth_radius)
                safe_radius = clamp(smooth_radius, 4, min(safe_cx, img.width()-safe_cx, safe_cy, img.height()-safe_cy))
                img.draw_circle((safe_cx, safe_cy, safe_radius), color=GREEN)

                # 仍然保留轨迹管理和半径平滑用于后续跟踪，但不影响当前圈的显示
                tid = match_tennis_track(assist_cx, assist_cy, used_track_ids, max(w, h))
                if tid is None:
                    tid = next_track_id
                    next_track_id += 1
                    prev_radius = None
                    lock_count = 0
                else:
                    prev_radius = tennis_tracks[tid][2]
                    lock_count = tennis_tracks[tid][4]

                if prev_radius is None:
                    smooth_radius = radius
                    lock_count = 0
                else:
                    jump = abs(radius - prev_radius)
                    jump_ratio = (jump * 100) // max(1, prev_radius)
                    unstable = (cue_conf == 0 and jump >= RADIUS_JUMP_ABS) or \
                               (cue_conf <= 1 and jump_ratio >= RADIUS_JUMP_RATIO_PCT)

                    if lock_count > 0:
                        smooth_radius = prev_radius
                        lock_count -= 1
                    elif unstable:
                        smooth_radius = prev_radius
                        lock_count = RADIUS_LOCK_HOLD
                    else:
                        smooth_radius = smooth_ball_radius(radius, prev_radius)

                tennis_tracks[tid] = (assist_cx, assist_cy, smooth_radius, 0, lock_count)
                used_track_ids.append(tid)
                # 显示网格位置信息和距离
                # 限制文字显示坐标，防止超出边界
                text_x1 = clamp(safe_cx + 5, 0, img.width() - 1)
                text_y1 = clamp(safe_cy - 10, 0, img.height() - 1)
                text_x2 = clamp(safe_cx + 5, 0, img.width() - 1)
                text_y2 = clamp(safe_cy + 10, 0, img.height() - 1)
                img.draw_string(text_x1, text_y1, pos_text, color=TEXT_COLOR, mono_space=False)
                img.draw_string(text_x2, text_y2, dist_text, color=TEXT_COLOR, mono_space=False)
                # 输出也用平滑后的半径
                info['radius'] = smooth_radius
                output_info.append(info)

                # 存储网球检测信息供状态机使用
                if distance_cm > 0:
                    tennis_detections.append({
                        'tid': tid,
                        'cx': assist_cx,
                        'cy': assist_cy,
                        'distance': distance_cm,
                        'x': x, 'y': y, 'w': w, 'h': h
                    })

            elif ("player" in label_l):
                # Tennis player: 用蓝色固定大小圆圈标记
                fixed_radius = 20  # 可根据实际调整
                BLUE = (0, 0, 255)
                # 限制圆心和半径，防止超出边界
                safe_cx = clamp(center_x, fixed_radius, img.width() - fixed_radius)
                safe_cy = clamp(center_y, fixed_radius, img.height() - fixed_radius)
                safe_radius = clamp(fixed_radius, 4, min(safe_cx, img.width()-safe_cx, safe_cy, img.height()-safe_cy))
                img.draw_circle((safe_cx, safe_cy, safe_radius), color=BLUE)
                # 限制文字显示坐标，防止超出边界
                text_x1 = clamp(safe_cx + 5, 0, img.width() - 1)
                text_y1 = clamp(safe_cy - 10, 0, img.height() - 1)
                img.draw_string(text_x1, text_y1, pos_text, color=TEXT_COLOR, mono_space=False)
                info['radius'] = fixed_radius
                info['pos_cm'] = 0
                info['distance_cm'] = 0
                info['g_id'] = f"({row},{col})"
                output_info.append(info)

                # 存储人物检测
                player_detections.append({
                    'cx': center_x,
                    'cy': center_y,
                    'x': x, 'y': y, 'w': w, 'h': h
                })

            elif "racket" in label_l:
                img.draw_rectangle((x, y, w, h), color=RED)
                info['radius'] = 0
                info['pos_cm'] = 0
                info['distance_cm'] = 0
                info['g_id'] = f"({row},{col})"
                output_info.append(info)

                # 存储球拍检测
                racket_detections.append({
                    'cx': center_x,
                    'cy': center_y,
                    'x': x, 'y': y, 'w': w, 'h': h
                })

            else:
                radius = 12
                # 限制圆心和半径，防止超出边界
                safe_cx = clamp(center_x, radius, img.width() - radius)
                safe_cy = clamp(center_y, radius, img.height() - radius)
                safe_radius = clamp(radius, 4, min(safe_cx, img.width()-safe_cx, safe_cy, img.height()-safe_cy))
                img.draw_circle((safe_cx, safe_cy, safe_radius), color=colors[i])
                # 限制文字显示坐标，防止超出边界
                text_x1 = clamp(safe_cx + 5, 0, img.width() - 1)
                text_y1 = clamp(safe_cy - 10, 0, img.height() - 1)
                img.draw_string(text_x1, text_y1, pos_text, color=TEXT_COLOR, mono_space=False)
                info['radius'] = radius
                info['pos_cm'] = 0
                info['distance_cm'] = 0
                info['g_id'] = f"({row},{col})"
                output_info.append(info)

    age_and_prune_tracks(used_track_ids)

    # ==================== 状态机与舵机控制 ====================
    # 先确定当前要跟踪的目标中心（如果有）
    target_cx = None
    target_cy = None

    if mode == MODE_PICK:
        # 捡球模式
        if pick_substate == PICK_SCAN:
            # 扫描：水平舵机往复运动
            pan_angle += scan_direction * scan_speed
            if pan_angle >= pan_angle_limit[1]:
                pan_angle = pan_angle_limit[1]
                scan_direction = -1
                # 完成一次扫描（从一端到另一端），可以认为扫描完成，准备选取最佳网球
                # 但为了确保能看到各个方向，我们可以在每次到达边界时评估一次，或者设定扫描时间
                # 简化：每次到达边界时，如果存在最佳候选，则锁定
                if best_scan_tennis is not None:
                    # 锁定该网球
                    locked_tennis_id = best_scan_tennis['tid']
                    pick_substate = PICK_TRACK
                    best_scan_tennis = None  # 清空
                    print("[PICK] 锁定网球 ID:", locked_tennis_id)
            elif pan_angle <= pan_angle_limit[0]:
                pan_angle = pan_angle_limit[0]
                scan_direction = 1
                if best_scan_tennis is not None:
                    locked_tennis_id = best_scan_tennis['tid']
                    pick_substate = PICK_TRACK
                    best_scan_tennis = None
                    print("[PICK] 锁定网球 ID:", locked_tennis_id)

            # 在扫描过程中，记录检测到的网球，选择距离最小的
            for det in tennis_detections:
                if best_scan_tennis is None or det['distance'] < best_scan_tennis['distance']:
                    best_scan_tennis = det

            # ====== 垂直舵机追踪（追踪当前帧最近的网球）======
            if tennis_detections:
                # 找到当前帧中距离最近的网球
                nearest_tennis = min(tennis_detections, key=lambda det: det['distance'])
                ref_cx, ref_cy = nearest_tennis['cx'], nearest_tennis['cy']
                # 计算垂直误差（图像中心Y坐标）
                tilt_error = ref_cy - img.height() / 2
                tilt_output = tilt_pid.get_pid(tilt_error, 1)
                tilt_angle += tilt_output
                tilt_angle = clamp(tilt_angle, tilt_angle_limit[0], tilt_angle_limit[1])
            # =============================================

            # 如果有锁定网球，则跟踪它，否则不更新水平舵机（由扫描控制）
            if locked_tennis_id is not None:
                # 如果锁定ID还在跟踪中，则进入TRACK状态
                if locked_tennis_id in tennis_tracks:
                    pick_substate = PICK_TRACK
                else:
                    # 锁定丢失，回到扫描
                    locked_tennis_id = None
                    pick_substate = PICK_SCAN
                    best_scan_tennis = None

        elif pick_substate == PICK_TRACK:
            # 跟踪锁定网球
            if locked_tennis_id is not None and locked_tennis_id in tennis_tracks:
                # 获取锁定网球的当前信息
                track = tennis_tracks[locked_tennis_id]
                tx, ty, tr, miss, lock = track
                # 检查距离是否≤10cm
                # 我们需要找到该网球对应的距离信息（从当前帧的检测中匹配）
                dist_cm = None
                for det in tennis_detections:
                    if det['tid'] == locked_tennis_id:
                        dist_cm = det['distance']
                        break
                if dist_cm is not None and dist_cm <= 10.0:
                    # 触发捕获指令
                    capture_cmd = 1
                    print("[PICK] 距离<=10cm，捕获指令=1")
                    # 捕获完成后（这里假设立即完成）清除指令，解锁，回到扫描
                    capture_cmd = 0
                    locked_tennis_id = None
                    pick_substate = PICK_SCAN
                    best_scan_tennis = None
                else:
                    # 正常跟踪：设置目标中心为网球中心
                    target_cx = tx
                    target_cy = ty
            else:
                # 锁定丢失，回到扫描
                locked_tennis_id = None
                pick_substate = PICK_SCAN
                best_scan_tennis = None

    elif mode == MODE_PLAY:
        # 对打模式
        if play_substate == PLAY_TRACK_PLAYER:
            # 追踪人物
            if player_detections:
                # 简单选取第一个（或最大）人物
                # 这里选择面积最大的
                max_area = 0
                best_player = None
                for p in player_detections:
                    area = p['w'] * p['h']
                    if area > max_area:
                        max_area = area
                        best_player = p
                if best_player:
                    target_cx = best_player['cx']
                    target_cy = best_player['cy']
                    player_locked = True
                    # 检查是否出现球拍
                    if racket_detections:
                        print("[PLAY] 检测到球拍，进入等待发球状态")
                        play_substate = PLAY_WAIT_SERVE
                else:
                    player_locked = False
            else:
                player_locked = False

        elif play_substate == PLAY_WAIT_SERVE:
            # 等待发球，同时继续追踪人物（不能追踪网球）
            if player_detections:
                # 仍然追踪人物
                max_area = 0
                best_player = None
                for p in player_detections:
                    area = p['w'] * p['h']
                    if area > max_area:
                        max_area = area
                        best_player = p
                if best_player:
                    target_cx = best_player['cx']
                    target_cy = best_player['cy']
                    # 发球模拟：如果检测到球拍，认为一次发球完成
                    if racket_detections:
                        balls_served += 1
                        print("[PLAY] 发球计数:", balls_served)
                        if balls_served >= target_balls:
                            # 发球完毕，解除锁定，回到捡球模式
                            print("[PLAY] 发球完毕，返回捡球模式")
                            mode = MODE_PICK
                            pick_substate = PICK_SCAN
                            player_locked = False
                            balls_served = 0
                            # 记录打出去的网球个数（这里用balls_served作为伪参考）
                            # 可以存储到某个变量，但本次暂不实现
                else:
                    # 人物丢失，可能回到追踪状态
                    play_substate = PLAY_TRACK_PLAYER
            else:
                # 人物丢失，回到追踪
                play_substate = PLAY_TRACK_PLAYER

    # 如果存在目标中心（target_cx, target_cy），计算误差并调整舵机
    if target_cx is not None and target_cy is not None:
        # 计算误差（图像中心为期望位置）
        pan_error = target_cx - img.width() / 2
        tilt_error = target_cy - img.height() / 2

        # PID计算输出
        pan_output = pan_pid.get_pid(pan_error, 1) / 2   # 比例因子可根据实际情况调整
        tilt_output = tilt_pid.get_pid(tilt_error, 1)

        # 更新舵机角度
        pan_angle -= pan_output   # 根据正负方向可能需要调整符号
        tilt_angle += tilt_output

        # 限幅
        pan_angle = clamp(pan_angle, pan_angle_limit[0], pan_angle_limit[1])
        tilt_angle = clamp(tilt_angle, tilt_angle_limit[0], tilt_angle_limit[1])

    # 可选：打印状态信息
    print("Mode:", "PICK" if mode==MODE_PICK else "PLAY",
          "Substate:", pick_substate if mode==MODE_PICK else play_substate,
          "Pan:", pan_angle, "Tilt:", tilt_angle,
          "Capture:", capture_cmd, "PlayerLocked:", player_locked)

    # 原有输出信息
    for obj in output_info:
        print(obj)
    print(f"{clock.fps():.5f} fps\n")

    lcd.write(img, hint=image.ROTATE_270)  # Take a picture and display the image.
