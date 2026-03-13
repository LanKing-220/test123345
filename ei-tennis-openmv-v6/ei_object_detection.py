# Edge Impulse - OpenMV FOMO Object Detection Example
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE

import sensor, image, time, ml, math, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.5

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]


# 各类别置信度阈值
THRESH_TENNIS = 0.5
THRESH_PLAYER = 0.4
THRESH_RACKET = 0.7
threshold_list = [(math.ceil(THRESH_TENNIS * 255), 255)]
GREEN = (0, 255, 0)
RED = (255, 0, 0)
tennis_tracks = {}
next_track_id = 1

# Color tolerance controls (LAB):
# Increase these values when lighting changes a lot, decrease to reduce false positives.
COLOR_TOL_L_BASE = 12
COLOR_TOL_A_BASE = 10
COLOR_TOL_B_BASE = 10
COLOR_TOL_L_EXTRA = 3
COLOR_TOL_A_EXTRA = 3
COLOR_TOL_B_EXTRA = 3.5

# Final draw scale for the tennis circle radius.
# Increase slightly (e.g. 1.20 -> 1.30) if circles still look too small.
DRAW_RADIUS_SCALE = 1.15

# Distance-adaptive controls for far-ball stability.
ROI_NEAR_SWITCH = 26
FAR_ROI_PAD_MIN = 6
MAX_COLOR_BLOB_AREA_MULT = 6
TRACK_MAX_MISS = 6

# Edge and circle constraints.
EDGE_LOW_TH = 55
EDGE_HIGH_TH = 110

# Strong-light handling for small balls on reflective floor.
GLARE_L_TH = 88
GLARE_RATIO_TH_PCT = 18
SMALL_BALL_SIZE_TH = 24
GLARE_DIAMETER_CAP_PCT = 115

# Radius lock controls: freeze radius for a few frames when cues become unstable.
RADIUS_LOCK_HOLD = 4
RADIUS_JUMP_ABS = 6
RADIUS_JUMP_RATIO_PCT = 35

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def match_tennis_track(cx, cy, used_track_ids, hint_size):
    # Match current detection to an existing tennis track (nearest valid center).
    best_id = None
    best_d2 = None
    gate = max(18, hint_size * 2)
    gate2 = gate * gate

    for tid in tennis_tracks:
        if tid in used_track_ids:
            continue
        tx, ty, tr, miss, lock_count = tennis_tracks[tid]
        dx = cx - tx
        dy = cy - ty
        d2 = (dx * dx) + (dy * dy)
        local_gate = max(gate2, (tr * tr * 4))
        if d2 > local_gate:
            continue
        if (best_d2 is None) or (d2 < best_d2):
            best_d2 = d2
            best_id = tid

    return best_id


def age_and_prune_tracks(used_track_ids):
    # Increase miss count for unmatched tracks and prune stale tracks.
    stale = []
    for tid in list(tennis_tracks.keys()):
        if tid in used_track_ids:
            continue
        tx, ty, tr, miss, lock_count = tennis_tracks[tid]
        miss += 1
        if miss > TRACK_MAX_MISS:
            stale.append(tid)
        else:
            tennis_tracks[tid] = (tx, ty, tr, miss, lock_count)

    for tid in stale:
        tennis_tracks.pop(tid)


def build_tennis_color_threshold(img, x, y, w, h):
    # Build dynamic LAB threshold from the center patch of the FOMO tennis bbox.
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

    # Adaptive tolerance: base allowance + texture/lighting variation from stdev.
    l_std = s.l_stdev()
    a_std = s.a_stdev()
    b_std = s.b_stdev()

    # Tightened tolerance window to reduce color over-segmentation.
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
    # Use dynamic color prior to find the tennis-colored blob near FOMO center.
    rx, ry, rw, rh = roi
    thr = build_tennis_color_threshold(img, ref_cx - (rw // 6), ref_cy - (rh // 6), rw // 3, rh // 3)
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
        # Reject oversized color regions that usually come from background when ball is far.
        if b.pixels() > max_blob_area:
            continue

        dx = b.cx() - ref_cx
        dy = b.cy() - ref_cy
        inside_ref = (b.x() <= ref_cx <= (b.x() + b.w())) and (b.y() <= ref_cy <= (b.y() + b.h()))

        # Blob roundness proxy: prefer near-square blobs for tennis balls.
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

    # Use equivalent-circle diameter from area for a less under-sized estimate.
    eq_d = int(math.sqrt((4.0 * best.pixels()) / math.pi))
    blob_d = max(best.w(), best.h())
    color_d = max(eq_d, blob_d)
    return color_d, best.cx(), best.cy()


def estimate_edge_strength(img, roi):
    # Edge map confidence for circle fit reliability.
    edge_img = img.copy(roi=roi)
    edge_img.to_grayscale()
    edge_img.find_edges(image.EDGE_CANNY, threshold=(EDGE_LOW_TH, EDGE_HIGH_TH))
    return edge_img.get_statistics().l_mean()


def estimate_glare_ratio_pct(img, roi):
    # Estimate % of very bright pixels in ROI using LAB L channel threshold.
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
    # Local Hough circle fit guided by FOMO center and optional color center.
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        hough_threshold = 3300 if edge_strength >= 24 else 2900
    else:
        hough_threshold = 3000 if edge_strength >= 24 else 2650

    r_guess = max(3, r_guess)
    r_min = max(3, (r_guess * 7) // 10)
    r_max = min(105, (r_guess * 13) // 10)
    if glare_ratio_pct >= GLARE_RATIO_TH_PCT:
        # Reflective floor often creates over-sized circles: shrink search upper bound.
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

        # Keep slight large-circle preference, but weaker to avoid over-sized circles.
        score = (center_cost * 2) + radius_cost - (c.r() // 4)
        if (best_score is None) or (score < best_score):
            best_score = score
            best = c

    if best is None:
        return None, ref_cx, ref_cy

    return best.r() * 2, best.x(), best.y()


def estimate_tennis_diameter(img, x, y, w, h):
    # Multi-cue diameter estimation:
    # FOMO bbox -> color prior -> edge confidence -> Hough circle.
    # Distance-adaptive ROI:
    # far balls use smaller ROI to avoid background color pollution.
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
    )

    cue_conf = 0
    if (hough_d is not None) and (color_d is not None):
        # Avoid tiny circles: keep the larger cue when they differ too much.
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

    # Strong-light guard for far/small balls: avoid over-sized radius from reflections.
    if (ball_size <= SMALL_BALL_SIZE_TH) and (glare_ratio_pct >= GLARE_RATIO_TH_PCT):
        glare_cap = max(8, (bbox_d * GLARE_DIAMETER_CAP_PCT) // 100)
        if d > glare_cap:
            d = glare_cap
        cue_conf = min(cue_conf, 1)

    # Auxiliary path only estimates diameter/confidence.
    # Center/identity must come from FOMO detection, not auxiliary cues.
    return d, cue_conf


def fuse_tennis_radius(w, h, detected_diameter):
    # Near-field robust radius:
    # combine circle fit with bbox-based estimate to avoid under-sized circles.
    circle_r = detected_diameter // 2
    bbox_r = estimate_ball_radius(w, h)

    # When ball is close (large bbox), trust the larger radius more.
    if max(w, h) >= 20:
        r = max(circle_r, bbox_r)
    else:
        # Far/mid range keeps circle fit dominant while retaining a safety floor.
        r = max(circle_r, (bbox_r * 9) // 10)

    # Final inflation compensates for partial edges/texture that underestimate radius.
    r = int((r * DRAW_RADIUS_SCALE) + 0.5)
    return clamp(r, 4, 105)


def estimate_ball_radius(w, h):
    # Make radius follow distance more clearly:
    # near ball => larger bbox => larger circle.
    mx = max(w, h)
    r = ((mx * 13) + 10) // 20  # ~0.65 * max(w, h)

    # Add a small boost for large nearby balls.
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
    # Radius smoothing for stable overlay on noisy detections:
    # 1) deadband suppresses tiny jitter
    # 2) step limit avoids sudden jumps
    # 3) EMA keeps motion smooth
    if prev_r is None:
        return curr_r

    diff = curr_r - prev_r

    # Ignore tiny changes to prevent flicker.
    if -2 <= diff <= 2:
        return prev_r

    # Adaptive step limit:
    # grow faster when object comes near, shrink slower for stability.
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
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh

    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

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
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

clock = time.clock()

GRID_ROWS = 12  # 可根据需要调整精度
GRID_COLS = 12
GRID_COLOR = (128, 128, 128)
TEXT_COLOR = (255, 255, 0)

def draw_dashed_line(img, x0, y0, x1, y1, color, dash_len=8, gap_len=6):
    # 只支持水平或垂直线
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
    col = min(cols - 1, max(0, (x * cols) // img_w))
    row = min(rows - 1, max(0, (y * rows) // img_h))
    return row, col

FOCAL_LENGTH_MM = 2.8  # OpenMV H7 Plus镜头典型焦距（可查具体镜头参数）
TENNIS_DIAMETER_MM = 67  # 标准网球直径
SENSOR_WIDTH_MM = 4.896  # OpenMV H7 Plus OV5640传感器宽度（mm）
IMAGE_WIDTH = 240  # 你的windowing宽度

def estimate_distance(pixel_diameter, sensor_width=SENSOR_WIDTH_MM, image_width=IMAGE_WIDTH):
    # pixel_diameter: 检测到的像素直径
    # sensor_width: 传感器宽度（mm）
    # image_width: 图像宽度（像素）
    mm_per_pixel = sensor_width / image_width
    h_mm = pixel_diameter * mm_per_pixel
    if h_mm == 0:
        return -1
    D = (FOCAL_LENGTH_MM * TENNIS_DIAMETER_MM) / h_mm
    return D  # 单位：mm

while(True):
    clock.tick()

    img = sensor.snapshot()
    used_track_ids = []
    # 画网格线
    draw_grid(img, GRID_ROWS, GRID_COLS, GRID_COLOR)

    output_info = []
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
            info['id'] = f"{i:02d}"

            label_l = labels[i].lower()
            if ("tennis" in label_l) and ("racket" not in label_l) and ("player" not in label_l):
                detected_diameter, cue_conf = estimate_tennis_diameter(img, x, y, w, h)
                radius = fuse_tennis_radius(w, h, detected_diameter)

                # 距离估算
                distance_mm = estimate_distance(detected_diameter)
                if distance_mm > 0:
                    distance_cm = (distance_mm / 10) * 2  # 距离结果乘以2
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

                # 直接用最新的radius画圈，保证与输出一致
                img.draw_circle((assist_cx, assist_cy, radius), color=GREEN)

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
                img.draw_string(assist_cx + 5, assist_cy - 10, pos_text, color=TEXT_COLOR, mono_space=False)
                img.draw_string(assist_cx + 5, assist_cy + 10, dist_text, color=TEXT_COLOR, mono_space=False)
                output_info.append(info)
            elif ("player" in label_l):
                # Tennis player: 用蓝色固定大小圆圈标记
                fixed_radius = 20  # 可根据实际调整
                BLUE = (0, 0, 255)
                img.draw_circle((center_x, center_y, fixed_radius), color=BLUE)
                # 显示网格位置信息
                img.draw_string(center_x + 5, center_y - 10, pos_text, color=TEXT_COLOR, mono_space=False)
                info['radius'] = fixed_radius
                info['pos_cm'] = 0
                info['distance_cm'] = 0
                info['g_id'] = f"({row},{col})"
                output_info.append(info)
            elif "racket" in label_l:
                img.draw_rectangle((x, y, w, h), color=RED)
                info['radius'] = 0
                info['pos_cm'] = 0
                info['distance_cm'] = 0
                info['g_id'] = f"({row},{col})"
                output_info.append(info)
            else:
                radius = 12
                img.draw_circle((center_x, center_y, radius), color=colors[i])
                # 显示网格位置信息
                img.draw_string(center_x + 5, center_y - 10, pos_text, color=TEXT_COLOR, mono_space=False)
                info['radius'] = radius
                info['pos_cm'] = 0
                info['distance_cm'] = 0
                info['g_id'] = f"({row},{col})"
                output_info.append(info)

    age_and_prune_tracks(used_track_ids)

    print(output_info)
    print(clock.fps(), "fps", end="\n\n")
