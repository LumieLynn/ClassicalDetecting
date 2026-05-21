"""Task 2 — 道路坐标追踪（Canny + 中心线拟合）。

用 Canny 边缘 + HoughLinesP 检测左右道路边界，
拟合道路中心线，采样固定点，基于中心线做 (lateral, arc_length) 坐标映射。
道路模型每 N 帧刷新（不受遮挡影响）。

输出路径: output/improved/task2_constrained/
"""

import os
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from classical.motion import KalmanTracker
from classical.scene_motion import SceneMotionEstimator
from classical.ncc import fast_ncc_match

# ======================== 模板提取 ========================
TM_HSV_LOWER = np.array([75, 20, 20])
TM_HSV_UPPER = np.array([150, 255, 255])
TM_BORDER_MASK_RATIO = 0.1
TM_MORPH_KERNEL = (3, 3)
TM_MORPH_ITER = 1
TM_MIN_CONTOUR_AREA = 30
TM_POS_BIAS_X = -0.1
TM_POS_DECAY_W = 0.2
TM_TEMPLATE_PAD = 2

# ======================== 锚点扫描 ========================
ANCHOR_MAX_SCAN = 600
ANCHOR_SCAN_STEP = 2
ANCHOR_MIN_SCORE = 0.4
ANCHOR_ROI_RATIO = 2

# ======================== 反向追踪 ========================
REV_PAD = 50
REV_SCALES = [0.92, 0.96, 1.0, 1.04]
REV_MIN_SCORE = 0.4
REV_MIN_SIZE = 8

# ======================== 道路检测（Otsu 分割）========================
ROAD_DETECT_INTERVAL = 1
ROAD_SAMPLE_N = 60  # 中心线采样点数
ROAD_POLY_DEG = 3  # 多项式阶数

# ======================== 道路坐标约束 ========================
ROAD_SPEED_EMA = 0.5
ROAD_X_THRESH = 25.0  # 横向一致性阈值（像素）
ROAD_Z_THRESH = 50.0  # 纵向一致性阈值（像素）

# ======================== 跳变检测 ========================
JUMP_WINDOW = 8  # delta 历史窗口
JUMP_LAT_MULT = 2.5  # lateral delta > median * MULT -> 跳变
JUMP_ARC_MULT = 2.5  # arc delta > median * MULT -> 跳变
JUMP_LAT_ABS_MIN = 25.0  # lateral 跳变绝对下限
JUMP_ARC_ABS_MIN = 50.0  # arc 跳变绝对下限

# ======================== 中心线抖动抑制 ========================
RD_JITTER_THRESH = 8.0
RD_LOSS_THRESH = 25.0
RD_JITTER_EMA = 0.6  # 抖动时旧权重（降低，更快响应视角变化）
RD_NORMAL_EMA = 0.3  # 正常时旧权重（0.3旧 + 0.7新）
RD_COOLDOWN = 5  # 路检失败后冷却帧数

# ======================== 正向追踪 ========================
FWD_SCALES = [0.50, 0.80, 0.88, 0.94, 1.0, 1.06, 1.12]
FWD_PAD = 60
FWD_MIN_SIZE = 8
FWD_NCC_GOOD = 0.32
FWD_NCC_MARGINAL = 0.20
FWD_LOST_MAX = 40

# ======================== 前景加权 ========================
FG_ALPHA_HEALTHY = 0.6
FG_ALPHA_REDUCED = 0.25

# ======================== 模板更新 ========================
TPL_INTERVAL = 8
TPL_BLEND = 0.15
TPL_SCALE_TRIGGER = 0.80
TPL_TRIGGER_BLEND = 0.35
TPL_SCALE_BAND = 0.6
TPL_MIN_PATCH = 8

# ======================== 通用 ========================
TRAJ_MAXLEN = 2000
PROG_INTERVAL = 50

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_BLUE = (255, 0, 0)
C_ORANGE = (255, 165, 0)
C_YELLOW = (0, 255, 255)


# ================================================================
#  道路路径模型（Canny -> 中心线拟合 -> 坐标映射）
# ================================================================
class RoadPathModel:
    """道路中心线模型，由 RoadPathDetector 构建。

    存储沿道路中心线的采样点、累积弧长、多项式拟合。
    提供 image ↔ road-path (lateral, arc_length) 坐标映射。
    """

    def __init__(self):
        self.pts = []  # [(x, y), ...] 中心线采样点
        self.arc = []  # 对应累积弧长（从底部开始）
        self.poly = None  # np.polyfit: x = f(y)
        self.ready = False

    def image_to_path(self, x, y):
        """图像坐标 -> (lateral_offset, arc_length)。"""
        if not self.ready or len(self.pts) < 2:
            return None
        # 找最近中心线点
        pts_arr = np.array(self.pts, dtype=np.float32)
        dx = pts_arr[:, 0] - x
        dy = pts_arr[:, 1] - y
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))
        cx, cy = self.pts[idx]
        arc_len = self.arc[idx]
        # 带符号的横向偏移（cross product 判左右）
        if idx < len(self.pts) - 1:
            tx = self.pts[idx + 1][0] - cx
            ty = self.pts[idx + 1][1] - cy
        else:
            tx = cx - self.pts[idx - 1][0]
            ty = cy - self.pts[idx - 1][1]
        cross = tx * (y - cy) - ty * (x - cx)
        lateral = np.sqrt(dist2[idx]) * (1.0 if cross >= 0 else -1.0)
        return lateral, arc_len

    def path_to_image(self, lateral, arc_len):
        """道路坐标 -> 图像坐标。"""
        if not self.ready or len(self.arc) < 2:
            return None
        arc_arr = np.array(self.arc)
        idx = int(np.searchsorted(arc_arr, arc_len))
        idx = max(1, min(idx, len(self.pts) - 1))
        # 在相邻采样点间线性插值
        a0, a1 = self.arc[idx - 1], self.arc[idx]
        t = (arc_len - a0) / max(a1 - a0, 1e-8)
        t = max(0.0, min(1.0, t))
        cx = self.pts[idx - 1][0] + t * (self.pts[idx][0] - self.pts[idx - 1][0])
        cy = self.pts[idx - 1][1] + t * (self.pts[idx][1] - self.pts[idx - 1][1])
        # 法向量偏移
        tx = self.pts[idx][0] - self.pts[idx - 1][0]
        ty = self.pts[idx][1] - self.pts[idx - 1][1]
        norm = np.sqrt(tx * tx + ty * ty) + 1e-8
        nx = -ty / norm
        ny = tx / norm
        x = cx + lateral * nx
        y = cy + lateral * ny
        return x, y


# ================================================================
#  道路路径检测器
# ================================================================
class RoadPathDetector:
    """Otsu 分割道路区域 -> 左右边界 -> 中心线拟合。"""

    def __init__(self, img_w, img_h):
        self.w = img_w
        self.h = img_h
        self.model = RoadPathModel()
        self._prev_pts = None
        self._ema = 0.5
        self._road_mask = None  # 最近一次检测的路面蒙版

    def detect(self, gray):
        """Otsu 分割亮色道路区域 -> 左右边界 -> 中心线。返回 True 表示成功。"""
        h, w = gray.shape
        y0 = int(h * 0.03)
        y1 = int(h * 0.97)
        roi = gray[y0:y1, :]

        # Otsu 阈值找亮色路面
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学清理
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 最大连通域 = 道路
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels < 2:
            return False
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        road_mask = (labels == largest_idx).astype(np.uint8) * 255

        # 逐行取左右边界中点 -> 中心线
        def _extract_centerline(mask):
            pts = []
            for row in range(mask.shape[0]):
                cols = np.where(mask[row, :] > 0)[0]
                if len(cols) > 20:
                    pts.append(((cols[0] + cols[-1]) / 2.0, float(row + y0)))
            return pts

        raw = _extract_centerline(road_mask)

        # 跨平台校验：底部中心线如果跑到画面边缘，说明前景/背景反了
        if len(raw) >= 10:
            bottom_x = raw[-1][0]  # 最后一行的 x 坐标
            if bottom_x < w * 0.1 or bottom_x > w * 0.9:
                # 尝试反相
                binary_inv = cv2.bitwise_not(binary)
                num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
                    binary_inv, connectivity=8
                )
                if num_labels2 >= 2:
                    areas2 = stats2[1:, cv2.CC_STAT_AREA]
                    idx2 = int(np.argmax(areas2)) + 1
                    mask2 = (labels2 == idx2).astype(np.uint8) * 255
                    raw_inv = _extract_centerline(mask2)
                    if len(raw_inv) >= 10:
                        bx2 = raw_inv[-1][0]
                        if w * 0.1 <= bx2 <= w * 0.9:
                            raw = raw_inv  # 反相后中心线合理，采用

        if len(raw) < 10:
            return False

        # 均匀采样
        n = min(ROAD_SAMPLE_N, len(raw))
        step = (len(raw) - 1) / max(n - 1, 1)
        pts = [raw[int(round(i * step))] for i in range(n)]

        # 帧间抖动检测 + EMA 平滑
        if self._prev_pts is not None and len(self._prev_pts) == n:
            diffs = [abs(px - qx) for (px, _), (qx, _) in zip(pts, self._prev_pts)]
            mean_diff = sum(diffs) / len(diffs)
            if mean_diff > RD_LOSS_THRESH:
                # 检测严重失败，沿用旧中心线
                return False
            wt = RD_NORMAL_EMA if mean_diff <= RD_JITTER_THRESH else RD_JITTER_EMA
            smooth = []
            for (px, py), (qx, qy) in zip(pts, self._prev_pts):
                smooth.append((wt * qx + (1 - wt) * px, wt * qy + (1 - wt) * py))
            pts = smooth
        self._prev_pts = [(p[0], p[1]) for p in pts]

        # 累积弧长
        arc = [0.0]
        for i in range(1, len(pts)):
            d = np.sqrt(
                (pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2
            )
            arc.append(arc[-1] + d)

        # 多项式拟合
        pts_arr = np.array(pts)
        deg = min(ROAD_POLY_DEG, len(pts) - 1)
        poly = np.polyfit(pts_arr[:, 1], pts_arr[:, 0], deg)

        model = RoadPathModel()
        model.pts = pts
        model.arc = arc
        model.poly = poly
        model.ready = True
        self.model = model
        self._road_mask = road_mask
        self._road_mask_y0 = y0  # ROI 顶部偏移（padding 用）
        return True


# ================================================================
#  道路运动模型
# ================================================================
class RoadMotionModel:
    """在道路坐标 (X, Z) 中维持速度估计 + 跳变检测。"""

    def __init__(self):
        self.X = 0.0
        self.Z = 0.0
        self.vx = 0.0
        self.vz = 0.0
        self.initialized = False
        self._prev_X = None
        self._prev_Z = None
        self._lat_history = deque(maxlen=JUMP_WINDOW + 2)
        self._arc_history = deque(maxlen=JUMP_WINDOW + 2)
        self._lat_deltas = deque(maxlen=JUMP_WINDOW)
        self._arc_deltas = deque(maxlen=JUMP_WINDOW)
        self.jump_count = 0

    def update(self, X, Z, dt=1.0):
        if not self.initialized:
            self.X, self.Z = X, Z
            self._prev_X, self._prev_Z = X, Z
            self._lat_history.append(X)
            self._arc_history.append(Z)
            self.initialized = True
            return

        raw_vx = (X - self._prev_X) / dt
        raw_vz = (Z - self._prev_Z) / dt
        a = ROAD_SPEED_EMA
        self.vx = a * self.vx + (1 - a) * raw_vx
        self.vz = a * self.vz + (1 - a) * raw_vz

        self.X, self.Z = X, Z
        self._prev_X, self._prev_Z = X, Z
        self._lat_history.append(X)
        self._arc_history.append(Z)
        self._lat_deltas.append(abs(raw_vx))
        self._arc_deltas.append(abs(raw_vz))

    def predict(self, dt=1.0):
        if not self.initialized:
            return self.X, self.Z
        return self.X + self.vx * dt, self.Z + self.vz * dt

    def reset_at(self, X, Z):
        """重设位置，清零速度。"""
        self.X, self.Z = X, Z
        self._prev_X, self._prev_Z = X, Z
        self.vx, self.vz = 0.0, 0.0
        self.initialized = True
        self._lat_history.clear()
        self._arc_history.clear()
        self._lat_deltas.clear()
        self._arc_deltas.clear()
        self._lat_history.append(X)
        self._arc_history.append(Z)

    def check_jump(self, X, Z):
        """检测道路坐标是否跳变。

        基于最近帧的 delta 中位数做自适应阈值。
        返回 (is_jump, lat_err, arc_err, lat_thresh, arc_thresh)。
        """
        if len(self._lat_history) < 3:
            return False, 0.0, 0.0, 0.0, 0.0

        prev_X = self._lat_history[-1]
        prev_Z = self._arc_history[-1]
        dX = abs(X - prev_X)
        dZ = abs(Z - prev_Z)

        # 自适应阈值：median delta * MULT，但不低于绝对下限
        med_lat = max(
            np.median(self._lat_deltas) if self._lat_deltas else 0,
            JUMP_LAT_ABS_MIN / JUMP_LAT_MULT,
        )
        med_arc = max(
            np.median(self._arc_deltas) if self._arc_deltas else 0,
            JUMP_ARC_ABS_MIN / JUMP_ARC_MULT,
        )

        lat_th = max(JUMP_LAT_ABS_MIN, med_lat * JUMP_LAT_MULT)
        arc_th = max(JUMP_ARC_ABS_MIN, med_arc * JUMP_ARC_MULT)

        is_jump = dX > lat_th or dZ > arc_th
        if is_jump:
            self.jump_count += 1

        return is_jump, dX, dZ, lat_th, arc_th


# ================================================================
#  前景加权 NCC
# ================================================================
def foreground_weighted_ncc(search_img, template, fg_prob=None, alpha=0.4):
    H, W = search_img.shape
    h, w = template.shape
    if H < h or W < w:
        return -1.0, (0, 0)

    img = search_img.astype(np.float64)
    tpl = template.astype(np.float64)
    N = h * w

    integral = cv2.integral(img)
    integral_sq = cv2.integral(img * img)

    rows, cols = H - h + 1, W - w + 1
    i_idx = np.arange(rows)[:, None]
    j_idx = np.arange(cols)[None, :]

    sum_I = (
        integral[i_idx + h, j_idx + w]
        - integral[i_idx + h, j_idx]
        - integral[i_idx, j_idx + w]
        + integral[i_idx, j_idx]
    )
    sum_sq = (
        integral_sq[i_idx + h, j_idx + w]
        - integral_sq[i_idx + h, j_idx]
        - integral_sq[i_idx, j_idx + w]
        + integral_sq[i_idx, j_idx]
    )

    mean_I = sum_I / N
    var_I = sum_sq / N - mean_I * mean_I
    var_I = np.maximum(var_I, 0.0)
    std_I = np.sqrt(var_I * N)

    t_mean = tpl.mean()
    t_std = np.sqrt(np.sum((tpl - t_mean) ** 2))
    if t_std < 1e-5:
        return -1.0, (0, 0)
    t_sum = tpl.sum()

    corr = cv2.filter2D(img, cv2.CV_64F, tpl, anchor=(0, 0))
    corr = corr[:rows, :cols]

    numerator = corr - mean_I * t_sum
    denominator = std_I * t_std + 1e-8
    ncc_map = np.full((rows, cols), -1.0, dtype=np.float64)
    valid = std_I > 1e-5
    ncc_map[valid] = numerator[valid] / denominator[valid]

    if fg_prob is not None and fg_prob.shape[0] >= rows and fg_prob.shape[1] >= cols:
        cy_off, cx_off = h // 2, w // 2
        fg_cropped = fg_prob[cy_off : cy_off + rows, cx_off : cx_off + cols]
        if fg_cropped.shape == (rows, cols):
            weight = 1.0 - alpha + alpha * fg_cropped
            ncc_map = ncc_map * weight

    max_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    return float(ncc_map[max_idx]), (int(max_idx[1]), int(max_idx[0]))


# ================================================================
#  统一目标搜索（正向 + 反向共用）
# ================================================================
def search_target(
    gray,
    fg_prob,
    road_mask,
    pred_cx,
    pred_cy,
    curr_tmpl,
    base_w,
    base_h,
    curr_scale,
    img_w,
    img_h,
    pad,
    scales,
    min_size,
):
    """路面约束 + 前景加权 NCC -> 统一搜索。"""
    if road_mask is not None and road_mask.shape[0] < img_h:
        rp = np.zeros((img_h, img_w), dtype=np.float32)
        mh = road_mask.shape[0]
        y_off = int(img_h * 0.03)  # Otsu ROI 从 3% 开始
        rp[y_off : y_off + mh, :] = road_mask.astype(np.float32) / 255.0
    elif road_mask is not None:
        rp = road_mask.astype(np.float32) / 255.0
    else:
        rp = np.ones((img_h, img_w), dtype=np.float32)
    combined = rp * (0.3 + 0.7 * fg_prob)

    x1 = max(0, int(pred_cx - pad))
    y1 = max(0, int(pred_cy - pad))
    x2 = min(img_w, int(pred_cx + pad))
    y2 = min(img_h, int(pred_cy + pad))
    search = gray[y1:y2, x1:x2]

    best_score = -1.0
    best_match = None
    for sf in scales:
        s = curr_scale * sf
        sw = int(base_w * s)
        sh = int(base_h * s)
        if (
            sw < min_size
            or sh < min_size
            or sw > search.shape[1]
            or sh > search.shape[0]
        ):
            continue
        scaled = cv2.resize(curr_tmpl, (sw, sh))
        fg_roi = combined[y1:y2, x1:x2]
        score, (lx, ly) = foreground_weighted_ncc(search, scaled, fg_roi, alpha=0.8)
        if score > best_score:
            best_score = score
            best_match = (lx + x1, ly + y1, sw, sh, s)

    return best_score, best_match


# ================================================================
#  模板提取
# ================================================================
def extract_template_from_ref(ref_path):
    print(f"从参考图提取模板: {ref_path}")
    img = cv2.imread(ref_path)
    if img is None:
        raise FileNotFoundError(f"无法读取参考图: {ref_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TM_HSV_LOWER, TM_HSV_UPPER)
    mask[: int(h * TM_BORDER_MASK_RATIO), :] = 0
    mask[int(h * (1.0 - TM_BORDER_MASK_RATIO)) :, :] = 0

    kernel = np.ones(TM_MORPH_KERNEL, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=TM_MORPH_ITER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到彩色区域")

    best_contour = None
    best_score = -1
    img_center = (w // 2, h // 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TM_MIN_CONTOUR_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = np.sqrt(
            (cx - img_center[0] + TM_POS_BIAS_X * w) ** 2 + (cy - img_center[1]) ** 2
        )
        pos_score = np.exp(-dist / (TM_POS_DECAY_W * w))
        score = area * circularity * pos_score
        if score > best_score:
            best_score = score
            best_contour = (cnt, (cx, cy))

    if best_contour is None:
        raise ValueError("没有找到合适的轮廓")

    cnt, _ = best_contour
    # 拟合椭圆 -> 内接矩形（切掉四角，比包围盒更紧）
    ellipse = cv2.fitEllipse(cnt)
    (ecx, ecy), (major, minor), angle = ellipse
    a, b = major / 2, minor / 2
    # 椭圆内接矩形半宽/半高: a/√2, b/√2
    iw = a * np.sqrt(2)
    ih = b * np.sqrt(2)
    # 内接矩形四角（旋转前）
    half = np.array(
        [[-iw / 2, -ih / 2], [iw / 2, -ih / 2], [iw / 2, ih / 2], [-iw / 2, ih / 2]]
    )
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = half @ R.T
    rotated[:, 0] += ecx
    rotated[:, 1] += ecy
    pad = TM_TEMPLATE_PAD
    x1 = max(0, int(np.min(rotated[:, 0])) - pad)
    y1 = max(0, int(np.min(rotated[:, 1])) - pad)
    x2 = min(w, int(np.max(rotated[:, 0])) + pad)
    y2 = min(h, int(np.max(rotated[:, 1])) + pad)
    patch = img[y1:y2, x1:x2]
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    out_dir = "output/improved/task2_constrained"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "auto_template.png"), gray_patch)
    print(f"模板: {gray_patch.shape[1]}x{gray_patch.shape[0]}")
    return gray_patch


# ================================================================
#  主程序
# ================================================================
def main():
    # 固定随机种子，确保跨平台 RANSAC / SIFT 结果一致
    np.random.seed(0)
    cv2.setRNGSeed(0)

    video_path = "data/task2/大疆无人机航拍视频.mp4"
    ref_image_path = "data/task2/大疆无人机航拍视频目标.png"
    output_dir = "output/improved/task2_constrained"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 {video_path}")
        return
    if not os.path.exists(ref_image_path):
        print(f"错误: 参考图不存在 {ref_image_path}")
        return

    try:
        base_template = extract_template_from_ref(ref_image_path)
    except Exception as e:
        print(f"模板提取失败: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {width}x{height}, {fps}fps, ~{total_frames} 帧")

    base_w, base_h = base_template.shape[1], base_template.shape[0]

    # ==================== 阶段一：锚点扫描 ====================
    print("阶段一：锚点扫描...")
    max_scan = min(ANCHOR_MAX_SCAN, total_frames)
    anchor_idx = -1
    anchor_box = None
    best_anchor_score = ANCHOR_MIN_SCORE
    scan_frames = []
    scan_grays = []
    frame_idx = 0

    while frame_idx < max_scan:
        ret, frame = cap.read()
        if not ret:
            break
        scan_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scan_grays.append(gray)
        if frame_idx % ANCHOR_SCAN_STEP == 0:
            roi = gray[height // ANCHOR_ROI_RATIO :, :]
            off_y = height // ANCHOR_ROI_RATIO
            score, (lx, ly) = fast_ncc_match(roi, base_template)
            if score > best_anchor_score:
                best_anchor_score = score
                anchor_idx = frame_idx
                anchor_box = (lx, ly + off_y, base_w, base_h)
                print(f"  锚点: 帧 {anchor_idx}, NCC={score:.3f}")
        frame_idx += 1

    if anchor_idx == -1:
        print("错误: 未找到目标。")
        cap.release()
        return
    print(f"锚点: 帧 {anchor_idx}, NCC={best_anchor_score:.3f}")

    # ==================== 道路检测初始化 ====================
    print("检测道路路径...")
    road_detector = RoadPathDetector(width, height)
    anchor_gray = scan_grays[anchor_idx]
    road_ok = road_detector.detect(anchor_gray)
    print(
        f"道路检测: {'成功' if road_ok else '失败，稍后重试'}，"
        f"中心线点数={len(road_detector.model.pts)}"
    )

    road_motion = RoadMotionModel()

    # ==================== 阶段二：反向追踪 ====================
    print("阶段二：反向追踪...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    history = {anchor_idx: anchor_box}
    curr_scale = 1.0
    prev_gray_rev = scan_grays[anchor_idx]  # 锚点帧作为帧差起点

    for i in range(anchor_idx - 1, -1, -1):
        gray = scan_grays[i]
        bx, by, bw, bh = history.get(i + 1, anchor_box)
        cx, cy = bx + bw // 2, by + bh // 2

        # 帧差前景 + 路面约束
        frame_diff = cv2.absdiff(gray, prev_gray_rev)
        _, fg_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        fg_prob_rev = cv2.GaussianBlur(fg_mask.astype(np.float32) / 255.0, (21, 21), 0)
        prev_gray_rev = gray
        road_mask_rev = road_detector._road_mask
        best_score, best_match = search_target(
            gray,
            fg_prob_rev,
            road_mask_rev,
            cx,
            cy,
            base_template,
            base_w,
            base_h,
            curr_scale,
            width,
            height,
            REV_PAD,
            REV_SCALES,
            REV_MIN_SIZE,
        )
        if best_score > REV_MIN_SCORE:
            _, _, _, _, curr_scale = best_match
            history[i] = (best_match[0], best_match[1], best_match[2], best_match[3])
        else:
            history[i] = history.get(i + 1, anchor_box)

    print("反向追踪完成。")

    # ==================== 阶段三：正向追踪（道路坐标约束）====================
    print("阶段三：正向追踪（Otsu 道路 + 坐标约束）...")
    out = cv2.VideoWriter(
        os.path.join(output_dir, "tracked_result.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # 道路坐标轨迹图视频
    ROAD_VIDEO_W = 500
    ROAD_VIDEO_H = 500
    road_video_out = cv2.VideoWriter(
        os.path.join(output_dir, "road_trajectory.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ROAD_VIDEO_W, ROAD_VIDEO_H),
    )

    scene_motion = SceneMotionEstimator()
    kalman = KalmanTracker()
    trajectory = deque(maxlen=TRAJ_MAXLEN)
    road_traj = deque(maxlen=TRAJ_MAXLEN)  # (X, Z) 列表

    curr_tmpl = base_template.astype(np.float32)
    curr_scale = 1.0
    last_tpl_update = anchor_idx
    cx = anchor_box[0] + anchor_box[2] // 2
    cy = anchor_box[1] + anchor_box[3] // 2
    target_w, target_h = anchor_box[2], anchor_box[3]
    kalman.init(cx, cy)

    consecutive_lost = 0
    in_occlusion = False
    is_tracking = True
    occ_good = [0]  # 遮挡退出连续确认计数器
    ncc_cooldown = 0  # NCC 冷却期（退出遮挡后 N 帧不更新 road_motion）
    rd_cooldown = 0
    frame_idx = 0
    frames_written = 0
    _prev_cx, _prev_cy = cx, cy

    # ---- 道路轨迹渲染 ----
    def _render_road_frame(rtraj, current_rpos, pred_rpos, occ, fridx):
        canvas = np.ones((ROAD_VIDEO_H, ROAD_VIDEO_W, 3), dtype=np.uint8) * 240

        # 自动缩放
        if len(rtraj) > 1:
            all_x = [p[0] for p in rtraj]
            all_z = [p[1] for p in rtraj]
            x_min, x_max = min(all_x), max(all_x)
            z_min, z_max = min(all_z), max(all_z)
            x_margin = max(0.01, (x_max - x_min) * 0.2)
            z_margin = max(0.001, (z_max - z_min) * 0.2)
            x_min -= x_margin
            x_max += x_margin
            z_min -= z_margin
            z_max += z_margin
        else:
            x_min, x_max = -0.1, 0.1
            z_min, z_max = 0.0, 0.02

        # 防除零
        if x_max - x_min < 1e-8:
            x_max = x_min + 0.01
        if z_max - z_min < 1e-8:
            z_max = z_min + 0.001

        def to_px(rx, rz):
            px = int((rx - x_min) / (x_max - x_min) * (ROAD_VIDEO_W - 40) + 20)
            # Z: near(small) -> bottom, far(large) -> top
            py = int(
                ROAD_VIDEO_H - 20 - (rz - z_min) / (z_max - z_min) * (ROAD_VIDEO_H - 40)
            )
            return px, py

        # 轨迹线
        if len(rtraj) > 1:
            pts = [to_px(p[0], p[1]) for p in rtraj]
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, (0, 0, 200), 1)
            for i, (px, py) in enumerate(pts):
                alpha = i / max(1, len(pts) - 1)
                cv2.circle(
                    canvas,
                    (px, py),
                    2,
                    (int(255 * (1 - alpha)), 0, int(255 * alpha)),
                    -1,
                )

        # 当前位置
        if current_rpos is not None:
            px, py = to_px(current_rpos[0], current_rpos[1])
            cv2.circle(canvas, (px, py), 7, (0, 200, 0), -1)
            cv2.circle(canvas, (px, py), 9, (0, 150, 0), 2)

        # 预测位置
        if occ and pred_rpos is not None:
            ppx, ppy = to_px(pred_rpos[0], pred_rpos[1])
            cv2.circle(canvas, (ppx, ppy), 6, (0, 200, 255), -1)
            if current_rpos is not None:
                cv2.line(canvas, (px, py), (ppx, ppy), (0, 200, 255), 1)

        cv2.putText(canvas, f"F:{fridx}", (10, 20), FONT, 0.45, (50, 50, 50), 1)
        cv2.putText(
            canvas,
            "X (lateral) ->",
            (ROAD_VIDEO_W - 120, ROAD_VIDEO_H - 8),
            FONT,
            0.35,
            (100, 100, 100),
            1,
        )
        cv2.putText(
            canvas, "^ Z (depth)", (5, ROAD_VIDEO_H - 8), FONT, 0.35, (100, 100, 100), 1
        )
        if occ:
            cv2.putText(
                canvas, "OCC", (ROAD_VIDEO_W - 50, 20), FONT, 0.5, (0, 0, 255), 2
            )
        return canvas

    # ====================== 主循环 ======================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 反向帧 ---
        if frame_idx < anchor_idx and frame_idx in history:
            bx, by, bw, bh = history[frame_idx]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_GREEN, 2)
            center = (bx + bw // 2, by + bh // 2)
            trajectory.append(center)
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, C_RED, 2)
                cv2.circle(frame, center, 4, C_BLUE, -1)
            out.write(frame)
            road_video_out.write(
                _render_road_frame(road_traj, None, None, False, frame_idx)
            )
            frames_written += 1
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 道路检测更新（带冷却）---
        road_lost = False
        if rd_cooldown > 0:
            rd_cooldown -= 1
        elif frame_idx % ROAD_DETECT_INTERVAL == 0:
            if road_detector.detect(gray):
                pass  # last_road_detect unused
            else:
                road_lost = road_detector.model.ready
                rd_cooldown = RD_COOLDOWN

        # --- 前景概率图 ---
        fg_prob, _, _ = scene_motion.align_and_diff(gray)

        # --- 锚点帧 ---
        if frame_idx == anchor_idx:
            bx, by, bw, bh = anchor_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_RED, 2)
            cv2.putText(frame, "ANCHOR", (bx, by - 10), FONT, 0.7, C_RED, 2)
            trajectory.append((bx + bw // 2, by + bh // 2))

            if road_detector.model.ready:
                rp = road_detector.model.image_to_path(cx, cy)
                if rp is not None:
                    road_motion.update(rp[0], rp[1])
                    road_traj.append(rp)

            scene_motion._update_prev(gray)
            out.write(frame)
            road_video_out.write(
                _render_road_frame(road_traj, rp, None, False, frame_idx)
            )
            frames_written += 1
            frame_idx += 1
            continue

        # ============================================================
        #  正常追踪帧
        # ============================================================

        # --- 预测搜索中心（卡尔曼 + 道路预测融合）---
        kf_cx, kf_cy, std_x, std_y = kalman.predict()

        if road_motion.initialized and road_detector.model.ready and not in_occlusion:
            rpx, rpz = road_motion.predict(dt=1)
            ri = road_detector.model.path_to_image(rpx, rpz)
            if ri is not None and 0 <= ri[0] < width and 0 <= ri[1] < height:
                pred_cx = 0.5 * kf_cx + 0.5 * ri[0]
                pred_cy = 0.5 * kf_cy + 0.5 * ri[1]
            else:
                pred_cx, pred_cy = kf_cx, kf_cy
        else:
            pred_cx, pred_cy = kf_cx, kf_cy

        # --- 丢失：不再尝试恢复（车太小 NCC 无意义）---
        if not is_tracking:
            cv2.putText(frame, "LOST", (30, 50), FONT, 0.9, C_RED, 2)
            out.write(frame)
            road_video_out.write(
                _render_road_frame(road_traj, None, None, False, frame_idx)
            )
            frames_written += 1
            if frame_idx % PROG_INTERVAL == 0:
                rd_status = "RD-OK" if not road_lost else "RD-LOSS"
                print(
                    f"  帧 {frame_idx}: LOST | {rd_status} "
                    f"| 中心线={len(road_detector.model.pts)}pts"
                )
            frame_idx += 1
            continue

        # --- 搜索区域（遮挡时缩小，防止跳到远处）---
        curr_w = int(base_w * curr_scale)
        curr_h = int(base_h * curr_scale)
        if in_occlusion:
            pad = max(FWD_PAD // 2, int(max(curr_w, curr_h) * 0.6))
        else:
            pad = max(FWD_PAD, int(max(curr_w, curr_h) * 1.2))
        # --- 统一搜索 ---
        road_mask_fwd = road_detector._road_mask
        best_ncc, best_match = search_target(
            gray,
            fg_prob,
            road_mask_fwd,
            pred_cx,
            pred_cy,
            curr_tmpl,
            base_w,
            base_h,
            curr_scale,
            width,
            height,
            pad,
            FWD_SCALES,
            FWD_MIN_SIZE,
        )

        if best_match is None:
            consecutive_lost += 1
            if consecutive_lost > FWD_LOST_MAX:
                is_tracking = False
            out.write(frame)
            road_video_out.write(
                _render_road_frame(road_traj, None, None, False, frame_idx)
            )
            frames_written += 1
            frame_idx += 1
            continue

        x, y, w, h, matched_scale = best_match
        ncc_cx, ncc_cy = x + w // 2, y + h // 2

        # ============================================================
        #  道路坐标验证 + 跳变检测
        # ============================================================
        road_ok = False
        road_jump = False
        ncc_rp = None
        pred_rp = None

        if road_motion.initialized and road_detector.model.ready:
            ncc_rp = road_detector.model.image_to_path(ncc_cx, ncc_cy)
            if ncc_rp is not None:
                pred_dt = min(consecutive_lost if in_occlusion else 1, 20)
                pred_rp = road_motion.predict(dt=pred_dt)

                dx = abs(ncc_rp[0] - pred_rp[0])
                dz = abs(ncc_rp[1] - pred_rp[1])

                gap_factor = 1.0 + consecutive_lost * 0.3
                x_thresh = ROAD_X_THRESH * gap_factor
                z_thresh = ROAD_Z_THRESH * gap_factor

                road_ok = dx < x_thresh and dz < z_thresh

                # 跳变检测
                road_jump, jx, jz, jx_th, jz_th = road_motion.check_jump(
                    ncc_rp[0], ncc_rp[1])
                if road_jump:
                    road_ok = False  # 跳变强制否决 road_ok

                if not road_ok:
                    # 用道路预测作为备选图像位置
                    ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                    if ri is not None:
                        ncc_cx, ncc_cy = ri

        # ============================================================
        #  跟踪决策
        # ============================================================
        # 健康跟踪：NCC 够高，或 NCC 边缘高 + 道路坐标一致
        track_healthy = best_ncc > FWD_NCC_GOOD or (
            best_ncc > FWD_NCC_MARGINAL and road_ok
        )
        if track_healthy and (road_ok or not road_motion.initialized):
            if in_occlusion:
                # 遮挡中：需连续 5 帧确认才退出
                occ_good[0] += 1
                if occ_good[0] >= 5:
                    in_occlusion = False
                    consecutive_lost = 0
                    occ_good[0] = 0
                    ncc_cooldown = 5
                # 仍用道路预测过渡
                final_cx, final_cy = x + w // 2, y + h // 2
                kalman.update(final_cx, final_cy)
                cx, cy = final_cx, final_cy
                _prev_cx, _prev_cy = cx, cy
                target_w, target_h = w, h
                status_color = (0, 255, 255) if in_occlusion else C_GREEN
            else:
                in_occlusion = False
                consecutive_lost = 0
                final_cx, final_cy = x + w // 2, y + h // 2
                kalman.update(final_cx, final_cy)
                cx, cy = final_cx, final_cy
                _prev_cx, _prev_cy = cx, cy
                target_w, target_h = w, h
                status_color = C_GREEN

            if ncc_rp is not None:
                if ncc_cooldown > 0:
                    ncc_cooldown -= 1
                    if ncc_cooldown == 0:
                        # 冷却结束：强制更新模板到当前外观
                        bx_t = int(cx - w // 2)
                        by_t = int(cy - h // 2)
                        bx_t = max(0, bx_t)
                        by_t = max(0, by_t)
                        ew = min(w, gray.shape[1] - bx_t)
                        eh = min(h, gray.shape[0] - by_t)
                        if ew >= TPL_MIN_PATCH and eh >= TPL_MIN_PATCH:
                            patch = gray[by_t : by_t + eh, bx_t : bx_t + ew]
                            curr_tmpl = cv2.resize(patch, (base_w, base_h)).astype(
                                np.float32
                            )
                            last_tpl_update = frame_idx
                else:
                    road_motion.update(ncc_rp[0], ncc_rp[1])
                road_traj.append(ncc_rp)

        elif best_ncc > FWD_NCC_GOOD and road_motion.initialized and not road_ok:
            if road_jump:
                # NCC 高但道路坐标跳变 -> 限高架/遮挡导致的错误匹配
                if not in_occlusion:
                    in_occlusion = True
                    consecutive_lost = 1
                    occ_good[0] = 0
                else:
                    consecutive_lost += 1
                # 道路预测（黄点）跟着车走 -> 用它主导框位置
                ncc_x, ncc_y = x + w // 2, y + h // 2
                if pred_rp is not None:
                    ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                    if ri is not None:
                        final_cx = 0.10 * ncc_x + 0.60 * ri[0] + 0.30 * kf_cx
                        final_cy = 0.10 * ncc_y + 0.60 * ri[1] + 0.30 * kf_cy
                    else:
                        final_cx, final_cy = kf_cx, kf_cy
                else:
                    final_cx, final_cy = kf_cx, kf_cy
                cx, cy = final_cx, final_cy
                _prev_cx, _prev_cy = cx, cy
                status_color = C_ORANGE
                if consecutive_lost > FWD_LOST_MAX:
                    is_tracking = False
                    in_occlusion = False
            else:
                # NCC 高但道路不一致（非跳变）
                if in_occlusion:
                    # 遮挡期间不信任 NCC，继续道路预测
                    consecutive_lost += 1
                    if pred_rp is not None:
                        ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                        if ri is not None:
                            final_cx = 0.2 * (x + w // 2) + 0.5 * ri[0] + 0.3 * kf_cx
                            final_cy = 0.2 * (y + h // 2) + 0.5 * ri[1] + 0.3 * kf_cy
                        else:
                            final_cx, final_cy = kf_cx, kf_cy
                    else:
                        final_cx, final_cy = kf_cx, kf_cy
                    cx, cy = final_cx, final_cy
                    _prev_cx, _prev_cy = cx, cy
                    status_color = C_ORANGE
                else:
                    # NCC 高但道路不一致（非跳变）：NCC 定位 + 道路模型不重置
                    # road_motion 保持原样，防止被后车坐标污染
                    in_occlusion = False
                    consecutive_lost = 0
                    final_cx, final_cy = x + w // 2, y + h // 2
                    kalman.update(final_cx, final_cy)
                    cx, cy = final_cx, final_cy
                    _prev_cx, _prev_cy = cx, cy
                    target_w, target_h = w, h
                    # prev_ncc = best_ncc  (unused)
                    status_color = (0, 255, 255)
                    # 不调用 road_motion.reset_at — 保持旧状态

        else:
            # NCC 低 -> 遮挡，用道路预测
            if not in_occlusion:
                in_occlusion = True
                consecutive_lost = 1
                occ_good[0] = 0
            else:
                consecutive_lost += 1

            if pred_rp is not None:
                ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                if ri is not None:
                    final_cx = 0.5 * ri[0] + 0.5 * kf_cx
                    final_cy = 0.5 * ri[1] + 0.5 * kf_cy
                else:
                    final_cx, final_cy = kf_cx, kf_cy
            else:
                final_cx, final_cy = kf_cx, kf_cy

            cx, cy = final_cx, final_cy
            _prev_cx, _prev_cy = cx, cy
            status_color = C_ORANGE

            if consecutive_lost > FWD_LOST_MAX:
                is_tracking = False
                in_occlusion = False
                print(f"帧 {frame_idx}: 连续 NCC 低 ({consecutive_lost})，全局搜索")

        # 出界检测
        if int(cx) < 0 or int(cx) >= width or int(cy) < 0 or int(cy) >= height:
            is_tracking = False
            in_occlusion = False
            print(f"帧 {frame_idx}: 目标出界，停止跟踪")

        trajectory.append((int(cx), int(cy)))

        # ============================================================
        #  模板更新
        # ============================================================
        scale_dropped = matched_scale < curr_scale * TPL_SCALE_TRIGGER
        time_to_update = (frame_idx - last_tpl_update) >= TPL_INTERVAL

        if (
            (time_to_update or scale_dropped)
            and not in_occlusion
            and best_ncc > FWD_NCC_MARGINAL
        ):
            bx_t = int(cx - w // 2)
            by_t = int(cy - h // 2)
            bx_t = max(0, bx_t)
            by_t = max(0, by_t)
            ew = min(w, gray.shape[1] - bx_t)
            eh = min(h, gray.shape[0] - by_t)
            if ew >= TPL_MIN_PATCH and eh >= TPL_MIN_PATCH:
                patch = gray[by_t : by_t + eh, bx_t : bx_t + ew]
                if matched_scale < TPL_SCALE_BAND and base_w > 20 and base_h > 15:
                    base_w = max(12, base_w // 2)
                    base_h = max(10, base_h // 2)
                    matched_scale *= 2.0
                    curr_tmpl = cv2.resize(curr_tmpl, (base_w, base_h))
                    blend = 0.5
                else:
                    blend = TPL_TRIGGER_BLEND if scale_dropped else TPL_BLEND
                new_tmpl = cv2.resize(patch, (base_w, base_h)).astype(np.float32)
                curr_tmpl = (1.0 - blend) * curr_tmpl + blend * new_tmpl
                last_tpl_update = frame_idx

        curr_scale = matched_scale

        # ============================================================
        #  可视化
        # ============================================================
        bx, by = int(cx - target_w // 2), int(cy - target_h // 2)
        cv2.rectangle(frame, (bx, by), (bx + target_w, by + target_h), status_color, 2)

        status_parts = [f"NCC:{best_ncc:.2f}", f"s={curr_scale:.2f}"]
        if road_lost:
            status_parts.append("RD-LOSS")
        elif rd_cooldown > 0:
            status_parts.append(f"RD-cd={rd_cooldown}")
        if in_occlusion:
            status_parts.append(f"OCC({consecutive_lost})")
        if road_jump:
            status_parts.append(f"JUMP!(j={road_motion.jump_count})")
        if road_motion.initialized:
            status_parts.append(f"X={road_motion.X:.1f}")
        cv2.putText(
            frame, " ".join(status_parts), (bx, by - 8), FONT, 0.4, status_color, 1
        )

        # 道路中心线
        if road_detector.model.ready and len(road_detector.model.pts) > 1:
            cpts = np.array(road_detector.model.pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [cpts], False, (255, 255, 0), 1)

        # 道路预测点
        if in_occlusion and pred_rp is not None:
            ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
            if ri is not None:
                cv2.circle(frame, (int(ri[0]), int(ri[1])), 6, C_YELLOW, -1)
                cv2.line(
                    frame, (int(cx), int(cy)), (int(ri[0]), int(ri[1])), C_YELLOW, 1
                )

        # 轨迹
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, C_RED, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, C_BLUE, -1)

        out.write(frame)
        road_video_out.write(
            _render_road_frame(
                road_traj,
                ncc_rp if ncc_rp is not None else (road_motion.X, road_motion.Z),
                pred_rp if in_occlusion else None,
                in_occlusion,
                frame_idx,
            )
        )
        frames_written += 1
        frame_idx += 1

        if frame_idx % PROG_INTERVAL == 0:
            rstr = ""
            if road_motion.initialized:
                rstr = (
                    f" road=({road_motion.X:.1f},{road_motion.Z:.1f})"
                    f" v=({road_motion.vx:.2f},{road_motion.vz:.2f})"
                )
            ocstr = f" OCC({consecutive_lost})" if in_occlusion else ""
            jstr = (
                f" JUMPS={road_motion.jump_count}" if road_motion.jump_count > 0 else ""
            )
            print(
                f"  帧 {frame_idx}: NCC={best_ncc:.2f} s={curr_scale:.2f}"
                f"{ocstr}{jstr}{rstr}"
            )

    # ==================== 道路轨迹图 ====================
    if len(road_traj) > 2:
        _save_road_trajectory_plot(road_traj, output_dir)

    cap.release()
    out.release()
    road_video_out.release()
    print("完成。")
    print(f"  跟踪视频: {os.path.join(output_dir, 'tracked_result.mp4')}")
    print(f"  道路轨迹: {os.path.join(output_dir, 'road_trajectory.mp4')}")


def _save_road_trajectory_plot(road_traj, output_dir):
    """保存静态道路坐标轨迹图"""
    canvas_w, canvas_h = 600, 500
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

    xs = [p[0] for p in road_traj]
    zs = [p[1] for p in road_traj]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    mx = max(0.01, (x_max - x_min) * 0.1)
    mz = max(0.001, (z_max - z_min) * 0.1)
    x_min -= mx
    x_max += mx
    z_min -= mz
    z_max += mz

    def tp(rx, rz):
        px = int((rx - x_min) / (x_max - x_min) * (canvas_w - 60) + 30)
        # Z: near(small) -> bottom, far(large) -> top
        py = int(canvas_h - 30 - (rz - z_min) / (z_max - z_min) * (canvas_h - 60))
        return px, py

    pts = [tp(p[0], p[1]) for p in road_traj]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (0, 0, 200), 1)
    for i, (px, py) in enumerate(pts):
        alpha = i / max(1, len(pts) - 1)
        cv2.circle(
            canvas, (px, py), 3, (int(255 * (1 - alpha)), 0, int(255 * alpha)), -1
        )
    if pts:
        cv2.circle(canvas, pts[0], 7, (0, 255, 0), -1)
        cv2.circle(canvas, pts[-1], 7, (0, 0, 255), -1)

    cv2.putText(
        canvas,
        "X (lateral)",
        (canvas_w // 2 - 40, canvas_h - 10),
        FONT,
        0.4,
        (100, 100, 100),
        1,
    )
    cv2.putText(
        canvas, "^ Z (depth)", (5, canvas_h - 10), FONT, 0.4, (100, 100, 100), 1
    )
    cv2.putText(canvas, "START", pts[0] if pts else (10, 30), FONT, 0.4, (0, 200, 0), 1)

    path = os.path.join(output_dir, "road_trajectory.png")
    cv2.imwrite(path, canvas)
    print(f"  轨迹图: {path}")


if __name__ == "__main__":
    main()
