"""Task 3 — 骑车人追踪（道路约束 + 场景运动 + 卡尔曼）。

白天道路检测（亮度分位阈值），建立道路坐标系约束骑车人运动。
场景运动估计（SIFT + 单应矩阵）用于前景加权和相机运动补偿。
卡尔曼滤波平滑轨迹。

输出路径: output/improved/task3_auto_road/
"""

import cv2
import numpy as np
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from classical.ncc import fast_ncc_match
from classical.scene_motion import SceneMotionEstimator
from classical.motion import KalmanTracker

# ======================== 模板提取 ========================
TM_HSV_LOWER = np.array([15, 40, 40])
TM_HSV_UPPER = np.array([45, 255, 255])
TM_BORDER_MASK_RATIO = 0.1
TM_MORPH_KERNEL = (3, 3)
TM_MORPH_ITER = 1
TM_MIN_CONTOUR_AREA = 20
TM_INSCRIBED_SCALE = 0.75      # 比 ms3 的 0.85 更小，减少背景车辆污染
TM_TEMPLATE_PAD = 2

# ======================== 锚点扫描 ========================
ANCHOR_SCAN_START_SEC = 9
ANCHOR_SCAN_END_SEC = 13
ANCHOR_MIN_SCORE = 0.4
ANCHOR_SCAN_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0]
ANCHOR_MIN_TEMPLATE_SIZE = 10

# ======================== 反向追踪 ========================
REV_PAD = 50
REV_SCALES = [0.92, 0.96, 1.0, 1.04]
REV_MIN_SCORE = 0.4
REV_MIN_SIZE = 8

# ======================== 道路检测（梯形蒙版 + SIFT）========================
ROAD_DETECT_INTERVAL = 1
ROAD_SAMPLE_N = 60
ROAD_POLY_DEG = 3
ROAD_MIN_CENTERLINE_PTS = 10
ROAD_SIFT_MIN_MATCHES = 8
ROAD_SIFT_MIN_INLIERS = 6

def auto_detect_road(gray, vehicle_cx, vehicle_cy):
    """以车辆位置为引导，自动检测路面梯形 4 角点。

    车辆坐标 -> ROI 直方图峰值 -> 亮度+方差掩码 -> RANSAC 拟合左右边线 -> 4 角点。
    """
    h, w = gray.shape
    margin_x, margin_y = 250, 150

    # 1. 车辆周围 ROI 灰度直方图 -> 路面峰值
    rx0 = max(0, vehicle_cx - margin_x)
    rx1 = min(w, vehicle_cx + margin_x)
    ry0 = max(0, vehicle_cy - margin_y)
    ry1 = min(h, vehicle_cy + margin_y)
    roi = gray[ry0:ry1, rx0:rx1]
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_s = cv2.GaussianBlur(hist, (7, 1), 0).flatten()
    peak = int(np.argmax(hist_s[30:230]) + 30)

    # 2. 局部方差图
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    var_map = cv2.GaussianBlur((gray.astype(np.float32) - blurred) ** 2,
                               (15, 15), 0)
    var_thresh = np.percentile(var_map[ry0:ry1, rx0:rx1], 55)

    # 3. 亮度 + 低方差 + 车辆周围空间约束
    mask = cv2.inRange(gray, max(0, peak - 25), min(255, peak + 25))
    mask = cv2.bitwise_and(mask, (var_map < var_thresh).astype(np.uint8) * 255)
    mask[:, :max(0, vehicle_cx - margin_x)] = 0

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        mask = (labels == int(np.argmax(areas)) + 1).astype(np.uint8) * 255

    # 4. 每行左右边界 -> RANSAC 拟合左右边线
    y_top = int(h * 0.03)
    y_bot = int(h * 0.97)
    left_pts, right_pts = [], []
    for y in range(y_top, y_bot, 2):
        cols = np.where(mask[y, :] > 0)[0]
        if len(cols) > 15:
            left_pts.append((cols[0], y))
            right_pts.append((cols[-1], y))

    if len(left_pts) < 20 or len(right_pts) < 20:
        return None

    def _ransac_line(pts):
        pts = np.float32(pts)
        best_inl, best_line = 0, None
        for _ in range(100):
            i, j = np.random.choice(len(pts), 2, replace=False)
            p1, p2 = pts[i], pts[j]
            if abs(p2[1] - p1[1]) < 1e-6:
                continue
            a = (p2[0] - p1[0]) / (p2[1] - p1[1] + 1e-8)
            b = p1[0] - a * p1[1]
            inliers = np.sum(np.abs(pts[:, 0] - (a * pts[:, 1] + b)) < 5)
            if inliers > best_inl:
                best_inl = inliers
                best_line = (a, b)
        return best_line

    ll = _ransac_line(left_pts)
    rl = _ransac_line(right_pts)
    if ll is None or rl is None:
        return None

    a_l, b_l = ll
    a_r, b_r = rl
    pts = [(int(a_l * y_top + b_l), y_top),
           (int(a_l * y_bot + b_l), y_bot),
           (int(a_r * y_top + b_r), y_top),
           (int(a_r * y_bot + b_r), y_bot)]
    cx_c = sum(p[0] for p in pts) / 4
    cy_c = sum(p[1] for p in pts) / 4
    return sorted(pts, key=lambda p: np.arctan2(p[1] - cy_c, p[0] - cx_c))

# ======================== 道路坐标约束 ========================
ROAD_SPEED_EMA = 0.5
ROAD_X_THRESH = 25.0
ROAD_Z_THRESH = 50.0

# ======================== 跳变检测 ========================
JUMP_WINDOW = 8
JUMP_LAT_MULT = 2.5
JUMP_ARC_MULT = 2.5
JUMP_LAT_ABS_MIN = 15.0
JUMP_ARC_ABS_MIN = 30.0

# ======================== 中心线抖动抑制 ========================
RD_JITTER_THRESH = 8.0
RD_LOSS_THRESH = 25.0
RD_JITTER_EMA = 0.85
RD_NORMAL_EMA = 0.5
RD_COOLDOWN = 0

# ======================== 正向追踪 ========================
FWD_SCALES = [0.80, 0.88, 0.94, 1.0, 1.06, 1.12]
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
#  道路路径模型
# ================================================================
class RoadPathModel:
    """道路中心线模型。

    存储沿道路中心线的采样点、累积弧长、多项式拟合。
    提供 image ↔ road-path (lateral, arc_length) 坐标映射。
    """

    def __init__(self):
        self.pts = []              # [(x, y), ...] 中心线采样点
        self.arc = []              # 对应累积弧长（从底部开始）
        self.poly = None           # np.polyfit: x = f(y)
        self.ready = False

    def image_to_path(self, x, y):
        """图像坐标 -> (lateral_offset, arc_length)。"""
        if not self.ready or len(self.pts) < 2:
            return None
        pts_arr = np.array(self.pts, dtype=np.float32)
        dx = pts_arr[:, 0] - x
        dy = pts_arr[:, 1] - y
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))
        cx, cy = self.pts[idx]
        arc_len = self.arc[idx]
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
        a0, a1 = self.arc[idx - 1], self.arc[idx]
        t = (arc_len - a0) / max(a1 - a0, 1e-8)
        t = max(0.0, min(1.0, t))
        cx = self.pts[idx - 1][0] + t * (self.pts[idx][0] - self.pts[idx - 1][0])
        cy = self.pts[idx - 1][1] + t * (self.pts[idx][1] - self.pts[idx - 1][1])
        tx = self.pts[idx][0] - self.pts[idx - 1][0]
        ty = self.pts[idx][1] - self.pts[idx - 1][1]
        norm = np.sqrt(tx * tx + ty * ty) + 1e-8
        nx = -ty / norm
        ny = tx / norm
        x = cx + lateral * nx
        y = cy + lateral * ny
        return x, y


# ================================================================
#  道路检测器（NCC 独立跟踪 4 角点）
# ================================================================
class RoadDetector:
    """梯形蒙版 + 逐帧 SIFT 链式跟踪 -> 重建中心线。

    锚点帧：梯形蒙版内 SIFT 关键点作为初始「上一帧」。
    后续帧：匹配上一帧->当前帧（小位移）-> 单应矩阵累乘 -> warp 梯形四角。
    大步横移被拆成 86 个小步，SIFT 不会丢失跟踪。
    """

    def __init__(self, img_w, img_h, manual_pts=None):
        self.w = img_w
        self.h = img_h
        self.model = RoadPathModel()
        self._prev_pts = None
        self._anchor_pts = list(manual_pts) if manual_pts else None
        self._road_pts = list(manual_pts) if manual_pts else None
        self._refs = []        # [(kp, des, gray, H_acc_from_anchor), ...] 参考帧栈
        self._prev_gray = None
        self._H_acc = np.eye(3, dtype=np.float64)

        self._sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def _build_road_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.int32(self._road_pts).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def init_anchor(self, gray):
        """锚点帧 -> 参考帧栈 -> 路面模型。"""
        h, w = gray.shape
        self._road_pts = list(self._anchor_pts)
        kp, des = self._sift.detectAndCompute(gray, None)
        self._prev_gray = gray.copy()
        self._H_acc = np.eye(3, dtype=np.float64)
        self._refs = [(kp, des, gray.copy(), self._H_acc.copy())]
        return self._build_model(gray)

    def track(self, gray):
        """多参考帧 SIFT 匹配 -> H 累乘 -> warp 锚点。"""
        h, w = gray.shape
        if not self._refs:
            return False

        kp_curr, des_curr = self._sift.detectAndCompute(gray, None)
        if des_curr is None or len(kp_curr) < 4:
            return False

        H = None
        # 先试锚点（无漂移），再试最近的非锚点参考
        refs_try = [self._refs[0]] + list(reversed(self._refs[1:]))
        for ri, (ref_kp, ref_des, ref_gray, ref_H) in enumerate(refs_try):
            matches = self._matcher.knnMatch(ref_des, des_curr, k=2)
            strict = [m for m, n in matches if m.distance < 0.65 * n.distance]
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            use = strict if len(strict) >= 10 else good
            if len(use) < ROAD_SIFT_MIN_MATCHES:
                continue
            src = np.float32([ref_kp[m.queryIdx].pt for m in use]
                           ).reshape(-1, 1, 2)
            dst = np.float32([kp_curr[m.trainIdx].pt for m in use]
                           ).reshape(-1, 1, 2)
            H_ref2curr, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
            if H_ref2curr is not None and inl is not None \
               and inl.sum() >= ROAD_SIFT_MIN_INLIERS:
                # H from anchor = ref->curr @ anchor->ref
                H = H_ref2curr @ ref_H
                # used_ref = ri  (unused)
                break

        # 回退：相位相关
        if H is None and self._prev_gray is not None:
            try:
                shift, _ = cv2.phaseCorrelate(
                    np.float32(self._prev_gray), np.float32(gray))
                dx, dy = shift
                if abs(dx) < w * 0.6 and abs(dy) < h * 0.4:
                    H = np.eye(3, dtype=np.float64)
                    H[0, 2] = dx
                    H[1, 2] = dy
            except Exception:
                pass

        if H is None:
            return False

        self._prev_gray = gray.copy()
        self._H_acc = H

        # Warp 锚点
        pts = np.float32(self._anchor_pts).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        new_pts = [(float(p[0][0]), float(p[0][1])) for p in warped]

        xs, ys = [p[0] for p in new_pts], [p[1] for p in new_pts]
        if (max(xs) < -w * 0.3 or min(xs) > w * 1.3
            or max(ys) < -h * 0.3 or min(ys) > h * 1.3):
            return False

        self._road_pts = new_pts

        # 每 8 帧压入新参考帧；锚点(_refs[0])永久保留，最多保留 4 个
        if not hasattr(self, '_track_cnt'):
            self._track_cnt = 0
        self._track_cnt += 1
        if self._track_cnt % 8 == 0:
            self._refs.append((kp_curr, des_curr, gray.copy(), H.copy()))
            if len(self._refs) > 4:
                self._refs.pop(1)  # 删最旧的非锚点参考

        return self._build_model(gray)

    def _snap_to_edges(self, gray, pts, window=7):
        """每个角点在小窗口内搜最强梯度位置，吸附到真实边缘。"""
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(grad * grad + grady * grady)
        h, w = gray.shape
        half = window // 2
        snapped = []
        for (px, py) in pts:
            x0 = max(half, int(px) - half)
            y0 = max(half, int(py) - half)
            x1 = min(w - half, int(px) + half + 1)
            y1 = min(h - half, int(py) + half + 1)
            if x1 <= x0 or y1 <= y0:
                snapped.append((px, py))
                continue
            patch = mag[y0:y1, x0:x1]
            _, _, _, max_loc = cv2.minMaxLoc(patch)
            snapped.append((x0 + max_loc[0], y0 + max_loc[1]))
        return snapped

    def _build_anchor_mask(self):
        """构建锚点帧路面掩码（只算一次，缓存）。"""
        if not hasattr(self, '_cached_anchor_mask'):
            h, w = self.h, self.w
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.int32(self._anchor_pts).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            self._cached_anchor_mask = mask
        return self._cached_anchor_mask

    def _build_model(self, gray):
        """Warp 锚点掩码到当前帧 -> 提取中心线 -> RoadPathModel。"""
        if self._anchor_pts is None:
            return False
        h, w = gray.shape
        anchor_mask = self._build_anchor_mask()
        mask = cv2.warpPerspective(anchor_mask, self._H_acc, (w, h),
                                   flags=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 取最大连通域（排除噪声碎片）
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if n_labels < 2:
            return False
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        mask = (labels == largest).astype(np.uint8) * 255

        # 逐行加权质心 = mask 的精确中线
        raw = []
        col_idx = np.arange(w, dtype=np.float32)
        for row in range(h):
            row_data = mask[row, :]
            total = int(row_data.sum())
            if total == 0:
                continue
            cx = float(np.dot(col_idx, row_data.astype(np.float32)) / total)
            raw.append((cx, float(row)))

        if len(raw) < ROAD_MIN_CENTERLINE_PTS:
            return False

        n = min(ROAD_SAMPLE_N, len(raw))
        step = (len(raw) - 1) / max(n - 1, 1)
        pts = [raw[int(round(i * step))] for i in range(n)]

        # SIFT warp 蒙版本身稳定，不做 EMA 平滑（急摇时反而会锁死旧位置）
        self._prev_pts = [(p[0], p[1]) for p in pts]

        arc = [0.0]
        for i in range(1, len(pts)):
            d = np.sqrt((pts[i][0] - pts[i - 1][0]) ** 2
                        + (pts[i][1] - pts[i - 1][1]) ** 2)
            arc.append(arc[-1] + d)

        pts_arr = np.array(pts)
        deg = min(ROAD_POLY_DEG, len(pts) - 1)
        poly = np.polyfit(pts_arr[:, 1], pts_arr[:, 0], deg)

        model = RoadPathModel()
        model.pts = pts
        model.arc = arc
        model.poly = poly
        model.ready = True
        self.model = model
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
        """检测道路坐标是否跳变。"""
        if len(self._lat_history) < 3:
            return False, 0.0, 0.0, 0.0, 0.0

        prev_X = self._lat_history[-1]
        prev_Z = self._arc_history[-1]
        dX = abs(X - prev_X)
        dZ = abs(Z - prev_Z)

        med_lat = max(np.median(self._lat_deltas) if self._lat_deltas else 0,
                      JUMP_LAT_ABS_MIN / JUMP_LAT_MULT)
        med_arc = max(np.median(self._arc_deltas) if self._arc_deltas else 0,
                      JUMP_ARC_ABS_MIN / JUMP_ARC_MULT)

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

    sum_I = (integral[i_idx + h, j_idx + w]
             - integral[i_idx + h, j_idx]
             - integral[i_idx, j_idx + w]
             + integral[i_idx, j_idx])
    sum_sq = (integral_sq[i_idx + h, j_idx + w]
              - integral_sq[i_idx + h, j_idx]
              - integral_sq[i_idx, j_idx + w]
              + integral_sq[i_idx, j_idx])

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
        fg_cropped = fg_prob[cy_off:cy_off + rows, cx_off:cx_off + cols]
        if fg_cropped.shape == (rows, cols):
            weight = 1.0 - alpha + alpha * fg_cropped
            ncc_map = ncc_map * weight

    max_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    return float(ncc_map[max_idx]), (int(max_idx[1]), int(max_idx[0]))


# ================================================================
#  统一目标搜索（正向 + 反向共用）
# ================================================================
def search_target(gray, fg_prob, road_prob, pred_cx, pred_cy,
                  curr_tmpl, base_w, base_h, curr_scale,
                  img_w, img_h, pad, scales, min_size):
    """路面约束 + 前景加权 NCC -> 统一的正向/反向搜索。

    Returns: (best_score, best_match) where best_match = (x, y, w, h, scale)
    """
    combined = road_prob * (0.3 + 0.7 * fg_prob)

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
        if (sw < min_size or sh < min_size
                or sw > search.shape[1] or sh > search.shape[0]):
            continue
        scaled = cv2.resize(curr_tmpl, (sw, sh))
        fg_roi = combined[y1:y2, x1:x2]
        score, (lx, ly) = foreground_weighted_ncc(
            search, scaled, fg_roi, alpha=0.8)
        if score > best_score:
            best_score = score
            best_match = (lx + x1, ly + y1, sw, sh, s)

    return best_score, best_match


# ================================================================
#  模板提取
# ================================================================
def extract_template_from_ref(ref_path, output_dir="output/improved/task3_auto_road"):
    print(f"从参考图提取模板: {ref_path}")
    img = cv2.imread(ref_path)
    if img is None:
        raise FileNotFoundError(f"无法读取参考图: {ref_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TM_HSV_LOWER, TM_HSV_UPPER)
    mask[:int(h * TM_BORDER_MASK_RATIO), :] = 0
    mask[int(h * (1.0 - TM_BORDER_MASK_RATIO)):, :] = 0

    kernel = np.ones(TM_MORPH_KERNEL, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=TM_MORPH_ITER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "debug_mask.png"), mask)
        raise ValueError("未找到黄色区域，已保存 debug_mask.png")

    # 取面积最大的黄色区域
    best_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best_cnt) < TM_MIN_CONTOUR_AREA:
        raise ValueError("黄色区域面积太小")

    # 椭圆拟合 -> 内接矩形 -> 比最小外接圆法更紧致
    try:
        ellipse = cv2.fitEllipse(best_cnt)
        (ecx, ecy), (major, minor), angle = ellipse
        a, b = major / 2, minor / 2
        iw = a * np.sqrt(2) * TM_INSCRIBED_SCALE
        ih = b * np.sqrt(2) * TM_INSCRIBED_SCALE
        half = np.array([[-iw / 2, -ih / 2], [iw / 2, -ih / 2],
                         [iw / 2, ih / 2], [-iw / 2, ih / 2]])
        theta = np.radians(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        rotated = half @ R.T
        rotated[:, 0] += ecx
        rotated[:, 1] += ecy
        pad = TM_TEMPLATE_PAD
        x1 = max(0, int(np.min(rotated[:, 0])) - pad)
        y1 = max(0, int(np.min(rotated[:, 1])) - pad)
        x2 = min(w, int(np.max(rotated[:, 0])) + pad)
        y2 = min(h, int(np.max(rotated[:, 1])) + pad)
    except Exception:
        # 椭圆拟合失败 -> 退化为外接圆内接正方形（ms3 原逻辑）
        (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
        side = int(radius * TM_INSCRIBED_SCALE)
        side = min(side, w, h)
        x1 = max(0, int(cx - side // 2))
        y1 = max(0, int(cy - side // 2))
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

    patch = img[y1:y2, x1:x2]
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "auto_template.png"), gray_patch)
    print(f"模板: {gray_patch.shape[1]}x{gray_patch.shape[0]}")
    return gray_patch


# ================================================================
#  主程序
# ================================================================
def main():
    np.random.seed(0)
    cv2.setRNGSeed(0)

    video_path = "data/task3/大疆无人机航拍骑车人.mp4"
    ref_image_path = "data/task3/大疆无人机航拍骑车人目标.png"
    output_dir = "output/improved/task3_auto_road"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 {video_path}")
        return
    if not os.path.exists(ref_image_path):
        print(f"错误: 参考图不存在 {ref_image_path}")
        return

    try:
        base_template = extract_template_from_ref(ref_image_path, output_dir)
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
    start_scan_frame = int(fps * ANCHOR_SCAN_START_SEC)
    max_scan = min(int(fps * ANCHOR_SCAN_END_SEC), total_frames)
    print(f"阶段一：锚点扫描（{ANCHOR_SCAN_START_SEC}～{ANCHOR_SCAN_END_SEC} 秒）...")

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

        if frame_idx >= start_scan_frame:
            best_local_score = -1.0
            best_local_match = None
            for s in ANCHOR_SCAN_SCALES:
                sw = int(base_w * s)
                sh = int(base_h * s)
                if (sw < ANCHOR_MIN_TEMPLATE_SIZE or sh < ANCHOR_MIN_TEMPLATE_SIZE
                        or sw > gray.shape[1] or sh > gray.shape[0]):
                    continue
                scaled = cv2.resize(base_template, (sw, sh))
                score, (lx, ly) = fast_ncc_match(gray, scaled)
                if score > best_local_score:
                    best_local_score = score
                    best_local_match = (lx, ly, sw, sh, s)
            if best_local_match is not None and best_local_score > best_anchor_score:
                lx, ly, sw, sh, s = best_local_match
                best_anchor_score = best_local_score
                anchor_idx = frame_idx
                anchor_box = (lx, ly, base_w, base_h)
                print(f"  锚点: 帧 {anchor_idx} ({anchor_idx/fps:.1f}s), "
                      f"NCC={best_anchor_score:.3f}")
        frame_idx += 1

    if anchor_idx == -1:
        print("错误: 未找到目标。")
        cap.release()
        return
    print(f"锚点: 帧 {anchor_idx}, NCC={best_anchor_score:.3f}")

    # ==================== 道路检测初始化 ====================
    print("自动检测路面...")
    acx = anchor_box[0] + anchor_box[2] // 2
    acy = anchor_box[1] + anchor_box[3] // 2
    anchor_gray = scan_grays[anchor_idx]
    road_pts = auto_detect_road(anchor_gray, acx, acy)
    if road_pts is None:
        print("错误: 自动路面检测失败")
        cap.release()
        return
    print(f"  路面 4 点: {road_pts}")
    road_detector = RoadDetector(width, height, manual_pts=road_pts)
    road_ok = road_detector.init_anchor(anchor_gray)
    print(f"道路检测: {'成功' if road_ok else '失败，稍后重试'}，"
          f"中心线点数={len(road_detector.model.pts)}")

    road_motion = RoadMotionModel()

    # ==================== 阶段二：反向追踪（SIFT 路面跟踪 + 道路约束）====================
    print("阶段二：反向追踪（SIFT 路面跟踪 + 道路约束）...")

    # 从锚点初始化道路运动模型
    if road_detector.model.ready:
        acx = anchor_box[0] + anchor_box[2] // 2
        acy = anchor_box[1] + anchor_box[3] // 2
        arp = road_detector.model.image_to_path(acx, acy)
        if arp is not None:
            road_motion.update(arp[0], arp[1])

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    history = {anchor_idx: anchor_box}
    curr_scale = 1.0

    # 存储每帧的 H_acc（反向帧用，避免主循环重复跟踪）
    rev_H_acc = {}

    for i in range(anchor_idx - 1, -1, -1):
        gray = scan_grays[i]

        # SIFT 跟踪路面 -> 蒙版 + 中线
        road_detector.track(gray)
        rev_H_acc[i] = road_detector._H_acc.copy()

        # 路面蒙版
        anchor_mask = road_detector._build_anchor_mask()
        road_mask = cv2.warpPerspective(
            anchor_mask, road_detector._H_acc, (width, height),
            flags=cv2.INTER_LINEAR)
        _, road_mask = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)
        road_prob = road_mask.astype(np.float32) / 255.0

        # 帧差前景：骑车人在动，路面静止
        diff_src = scan_grays[anchor_idx] if i == anchor_idx - 1 \
                   else scan_grays[i + 1]
        frame_diff = cv2.absdiff(gray, diff_src)
        _, fg_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        fg_prob = cv2.GaussianBlur(
            fg_mask.astype(np.float32) / 255.0, (21, 21), 0)

        # 合并：路面内 + 运动处 = 最高权重
        # combined_prob = road_prob  (unused) * (0.3 + 0.7 * fg_prob)

        # 道路坐标预测搜索中心
        bx, by, bw, bh = history.get(i + 1, anchor_box)
        if road_motion.initialized and road_detector.model.ready:
            pred_rp = road_motion.predict(dt=1)
            ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
            if ri is not None:
                pred_cx, pred_cy = int(ri[0]), int(ri[1])
            else:
                pred_cx, pred_cy = bx + bw // 2, by + bh // 2
        else:
            pred_cx, pred_cy = bx + bw // 2, by + bh // 2

        # 统一搜索
        pad = max(REV_PAD, int(max(bw, bh) * 1.5))
        best_score, best_match = search_target(
            gray, fg_prob, road_prob, pred_cx, pred_cy,
            base_template, base_w, base_h, curr_scale,
            width, height, pad, REV_SCALES, REV_MIN_SIZE)

        accepted = False
        if best_score > REV_MIN_SCORE and best_match is not None:
            ncc_cx = best_match[0] + best_match[2] // 2
            ncc_cy = best_match[1] + best_match[3] // 2
            if road_detector.model.ready and road_motion.initialized:
                ncc_rp = road_detector.model.image_to_path(ncc_cx, ncc_cy)
                if ncc_rp is not None:
                    pred_rp = road_motion.predict(dt=1)
                    dlat = abs(ncc_rp[0] - pred_rp[0])
                    darc = abs(ncc_rp[1] - pred_rp[1])
                    # 收紧横向阈值 + 跳变检测
                    is_jump, _, _, _, _ = road_motion.check_jump(ncc_rp[0], ncc_rp[1])
                    if not is_jump and dlat < ROAD_X_THRESH * 1.5 \
                       and darc < ROAD_Z_THRESH * 2.5:
                        accepted = True
                        road_motion.update(ncc_rp[0], ncc_rp[1])
            else:
                accepted = True

        if accepted:
            bx2, by2, bw2, bh2 = best_match[0], best_match[1], best_match[2], best_match[3]
            # 出界检测：框必须在画面内
            if (bx2 + bw2 // 2 < 0 or bx2 + bw2 // 2 >= width
                or by2 + bh2 // 2 < 0 or by2 + bh2 // 2 >= height):
                history[i] = history.get(i + 1, anchor_box)
            else:
                _, _, _, _, curr_scale = best_match
                history[i] = (bx2, by2, bw2, bh2)
        else:
            history[i] = history.get(i + 1, anchor_box)

    # 卡尔曼正向平滑反向轨迹（消除帧间 NCC 随机抖动）
    print("平滑反向轨迹...")
    kf_rev = KalmanTracker()
    # 从帧 0 开始
    b0 = history[0]
    kf_rev.init(b0[0] + b0[2] // 2, b0[1] + b0[3] // 2)
    for i in range(1, anchor_idx):
        if i not in history:
            continue
        bx, by, bw, bh = history[i]
        cx, cy = bx + bw // 2, by + bh // 2
        kf_rev.predict()
        kf_rev.update(cx, cy)
        # 用滤波后的位置替换
        scx = int(kf_rev.kf.statePost[0, 0])
        scy = int(kf_rev.kf.statePost[1, 0])
        history[i] = (scx - bw // 2, scy - bh // 2, bw, bh)

    # 反向追踪完成后完全重置 road_motion（逆向数据对正向无意义）
    road_motion = RoadMotionModel()

    print("反向追踪完成。")

    # ==================== 阶段三：正向追踪（道路坐标约束）====================
    print("阶段三：正向追踪（道路 + 场景运动 + 卡尔曼）...")
    out = cv2.VideoWriter(
        os.path.join(output_dir, "tracked_result.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height),
    )

    ROAD_VIDEO_W = 500
    ROAD_VIDEO_H = 500
    road_video_out = cv2.VideoWriter(
        os.path.join(output_dir, "road_trajectory.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (ROAD_VIDEO_W, ROAD_VIDEO_H),
    )

    scene_motion = SceneMotionEstimator()
    kalman = KalmanTracker()
    trajectory = deque(maxlen=TRAJ_MAXLEN)
    road_traj = deque(maxlen=TRAJ_MAXLEN)

    curr_tmpl = base_template.astype(np.float32)
    curr_scale = 1.0
    last_tpl_update = anchor_idx
    cx = anchor_box[0] + anchor_box[2] // 2
    cy = anchor_box[1] + anchor_box[3] // 2
    target_w, target_h = anchor_box[2], anchor_box[3]
    kalman.init(cx, cy)

    consecutive_lost = 0
    is_tracking = True
    rd_cooldown = 0
    frame_idx = 0
    frames_written = 0
    _prev_cx, _prev_cy = cx, cy

    # ---- 道路轨迹渲染 ----
    def _render_road_frame(rtraj, current_rpos, pred_rpos, occ, fridx):
        canvas = np.ones((ROAD_VIDEO_H, ROAD_VIDEO_W, 3), dtype=np.uint8) * 240

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

        if x_max - x_min < 1e-8:
            x_max = x_min + 0.01
        if z_max - z_min < 1e-8:
            z_max = z_min + 0.001

        def to_px(rx, rz):
            px = int((rx - x_min) / (x_max - x_min) * (ROAD_VIDEO_W - 40) + 20)
            py = int(ROAD_VIDEO_H - 20
                     - (rz - z_min) / (z_max - z_min) * (ROAD_VIDEO_H - 40))
            return px, py

        if len(rtraj) > 1:
            pts = [to_px(p[0], p[1]) for p in rtraj]
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, (0, 0, 200), 1)
            for i, (px, py) in enumerate(pts):
                alpha = i / max(1, len(pts) - 1)
                cv2.circle(canvas, (px, py), 2,
                           (int(255 * (1 - alpha)), 0, int(255 * alpha)), -1)

        if current_rpos is not None:
            px, py = to_px(current_rpos[0], current_rpos[1])
            cv2.circle(canvas, (px, py), 7, (0, 200, 0), -1)
            cv2.circle(canvas, (px, py), 9, (0, 150, 0), 2)

        if occ and pred_rpos is not None:
            ppx, ppy = to_px(pred_rpos[0], pred_rpos[1])
            cv2.circle(canvas, (ppx, ppy), 6, (0, 200, 255), -1)
            if current_rpos is not None:
                cv2.line(canvas, (px, py), (ppx, ppy), (0, 200, 255), 1)

        cv2.putText(canvas, f"F:{fridx}", (10, 20), FONT, 0.45, (50, 50, 50), 1)
        cv2.putText(canvas, "X (lateral) ->", (ROAD_VIDEO_W - 120, ROAD_VIDEO_H - 8),
                    FONT, 0.35, (100, 100, 100), 1)
        cv2.putText(canvas, "^ Z (depth)", (5, ROAD_VIDEO_H - 8),
                    FONT, 0.35, (100, 100, 100), 1)
        if occ:
            cv2.putText(canvas, "OCC", (ROAD_VIDEO_W - 50, 20),
                        FONT, 0.5, (0, 0, 255), 2)
        return canvas

    # ====================== 主循环 ======================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 道路检测更新 ---
        road_lost = False
        if rd_cooldown > 0:
            rd_cooldown -= 1
        elif frame_idx >= anchor_idx and frame_idx % ROAD_DETECT_INTERVAL == 0:
            # 正向帧：逐帧 SIFT 链式跟踪
            if road_detector.track(gray):
                pass  # last_road_detect unused
            else:
                road_lost = road_detector.model.ready
                rd_cooldown = RD_COOLDOWN

        # --- 反向帧：用阶段二预计算的角点 ---
        if frame_idx < anchor_idx and frame_idx in history:
            if frame_idx in rev_H_acc:
                road_detector._H_acc = rev_H_acc[frame_idx]
                road_detector._build_model(gray)
            bx, by, bw, bh = history[frame_idx]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_GREEN, 2)
            center = (bx + bw // 2, by + bh // 2)
            trajectory.append(center)
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, C_RED, 2)
                cv2.circle(frame, center, 4, C_BLUE, -1)

            # 反向帧：记录道路坐标轨迹（road_motion 已在阶段二更新）
            rp = None
            if road_detector.model.ready:
                rp = road_detector.model.image_to_path(center[0], center[1])
                if rp is not None:
                    road_traj.append(rp)

            # 反向帧也画道路中心线
            if road_detector.model.ready and len(road_detector.model.pts) > 1:
                cpts = np.array(road_detector.model.pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [cpts], False, (255, 255, 0), 1)

            out.write(frame)
            road_video_out.write(_render_road_frame(
                road_traj, rp, None, False, frame_idx))
            frames_written += 1
            frame_idx += 1
            continue

        # --- 前景概率图 + 路面蒙版 ---
        fg_prob, _, _ = scene_motion.align_and_diff(gray)
        anchor_mask_fwd = road_detector._build_anchor_mask()
        road_mask_fwd = cv2.warpPerspective(
            anchor_mask_fwd, road_detector._H_acc, (width, height),
            flags=cv2.INTER_LINEAR)
        _, road_mask_fwd = cv2.threshold(road_mask_fwd, 127, 255, cv2.THRESH_BINARY)
        road_prob = road_mask_fwd.astype(np.float32) / 255.0

        # --- 锚点帧 ---
        if frame_idx == anchor_idx:
            # 重置 SIFT 链：正向跟踪从锚点开始
            road_detector.init_anchor(gray)

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
            road_video_out.write(_render_road_frame(
                road_traj, rp, None, False, frame_idx))
            frames_written += 1
            frame_idx += 1
            continue

        # ============================================================
        #  正常追踪帧
        # ============================================================

        # --- 预测搜索中心（卡尔曼 + 道路预测融合）---
        kf_cx, kf_cy, std_x, std_y = kalman.predict()

        if road_motion.initialized and road_detector.model.ready:
            rpx, rpz = road_motion.predict(dt=1)
            ri = road_detector.model.path_to_image(rpx, rpz)
            if ri is not None and 0 <= ri[0] < width and 0 <= ri[1] < height:
                pred_cx = 0.5 * kf_cx + 0.5 * ri[0]
                pred_cy = 0.5 * kf_cy + 0.5 * ri[1]
            else:
                pred_cx, pred_cy = kf_cx, kf_cy
        else:
            pred_cx, pred_cy = kf_cx, kf_cy

        # --- 丢失状态 ---
        if not is_tracking:
            cv2.putText(frame, "LOST", (30, 50), FONT, 0.9, C_RED, 2)
            out.write(frame)
            road_video_out.write(_render_road_frame(
                road_traj, None, None, False, frame_idx))
            frames_written += 1
            if frame_idx % PROG_INTERVAL == 0:
                rd_status = "RD-OK" if not road_lost else "RD-LOSS"
                print(f"  帧 {frame_idx}: LOST | {rd_status} "
                      f"| 中心线={len(road_detector.model.pts)}pts")
            frame_idx += 1
            continue

        # --- 搜索区域 ---
        curr_w = int(base_w * curr_scale)
        curr_h = int(base_h * curr_scale)
        pad = max(FWD_PAD, int(max(curr_w, curr_h) * 1.2))
        x1 = max(0, int(pred_cx - pad))
        y1 = max(0, int(pred_cy - pad))
        x2 = min(width, int(pred_cx + pad))
        y2 = min(height, int(pred_cy + pad))
        search = gray[y1:y2, x1:x2]

        if search.size == 0:
            consecutive_lost += 1
            if consecutive_lost > FWD_LOST_MAX:
                is_tracking = False
            out.write(frame)
            road_video_out.write(_render_road_frame(
                road_traj, None, None, False, frame_idx))
            frames_written += 1
            frame_idx += 1
            continue

        # --- 统一搜索 ---
        best_score, best_match = search_target(
            gray, fg_prob, road_prob, pred_cx, pred_cy,
            curr_tmpl, base_w, base_h, curr_scale,
            width, height, pad, FWD_SCALES, FWD_MIN_SIZE)

        if best_match is None:
            consecutive_lost += 1
            if consecutive_lost > FWD_LOST_MAX:
                is_tracking = False
            out.write(frame)
            road_video_out.write(_render_road_frame(
                road_traj, None, None, False, frame_idx))
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
                pred_dt = 1
                pred_rp = road_motion.predict(dt=pred_dt)

                dx = abs(ncc_rp[0] - pred_rp[0])
                dz = abs(ncc_rp[1] - pred_rp[1])

                gap_factor = 1.0 + consecutive_lost * 0.3
                x_thresh = ROAD_X_THRESH * gap_factor
                z_thresh = ROAD_Z_THRESH * gap_factor

                road_ok = dx < x_thresh and dz < z_thresh

                if best_score < 0.60:
                    road_jump, jx, jz, jx_th, jz_th = road_motion.check_jump(
                        ncc_rp[0], ncc_rp[1])
                if road_jump:
                    road_ok = False

                if not road_ok:
                    ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                    if ri is not None:
                        ncc_cx, ncc_cy = ri

        # ============================================================
        #  跟踪决策（简化版：无遮挡状态机）
        # ============================================================
        if best_score > FWD_NCC_GOOD and road_ok:
            consecutive_lost = 0
            final_cx, final_cy = x + w // 2, y + h // 2
            kalman.update(final_cx, final_cy)
            cx, cy = final_cx, final_cy
            _prev_cx, _prev_cy = cx, cy
            target_w, target_h = w, h
            status_color = C_GREEN

            if ncc_rp is not None:
                road_motion.update(ncc_rp[0], ncc_rp[1])
                road_traj.append(ncc_rp)

        elif road_jump:
            consecutive_lost += 1
            if pred_rp is not None:
                ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                if ri is not None:
                    final_cx = 0.3 * (x + w // 2) + 0.7 * ri[0]
                    final_cy = 0.3 * (y + h // 2) + 0.7 * ri[1]
                else:
                    final_cx, final_cy = kf_cx, kf_cy
            else:
                final_cx, final_cy = kf_cx, kf_cy
            cx, cy = final_cx, final_cy
            _prev_cx, _prev_cy = cx, cy
            status_color = C_ORANGE

        else:
            consecutive_lost += 1
            if pred_rp is not None:
                ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                if ri is not None:
                    w_road = min(1.0, 0.5 + consecutive_lost * 0.05)
                    final_cx = w_road * ri[0] + (1 - w_road) * kf_cx
                    final_cy = w_road * ri[1] + (1 - w_road) * kf_cy
                else:
                    final_cx, final_cy = kf_cx, kf_cy
            else:
                final_cx, final_cy = kf_cx, kf_cy
            cx, cy = final_cx, final_cy
            _prev_cx, _prev_cy = cx, cy
            status_color = C_ORANGE

        if consecutive_lost > FWD_LOST_MAX:
            is_tracking = False
            print(f"帧 {frame_idx}: 连续丢失 ({consecutive_lost})，停止跟踪")

        # 出界检测
        if (int(cx) < 0 or int(cx) >= width or int(cy) < 0 or int(cy) >= height):
            is_tracking = False
            print(f"帧 {frame_idx}: 目标出界，停止跟踪")

        trajectory.append((int(cx), int(cy)))

        # ============================================================
        #  模板更新
        # ============================================================
        scale_dropped = matched_scale < curr_scale * TPL_SCALE_TRIGGER
        time_to_update = (frame_idx - last_tpl_update) >= TPL_INTERVAL

        if (time_to_update or scale_dropped) \
           and best_score > FWD_NCC_MARGINAL:
            bx_t = int(cx - w // 2)
            by_t = int(cy - h // 2)
            bx_t = max(0, bx_t)
            by_t = max(0, by_t)
            ew = min(w, gray.shape[1] - bx_t)
            eh = min(h, gray.shape[0] - by_t)
            if ew >= TPL_MIN_PATCH and eh >= TPL_MIN_PATCH:
                patch = gray[by_t:by_t + eh, bx_t:bx_t + ew]
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
        cv2.rectangle(frame, (bx, by), (bx + target_w, by + target_h),
                      status_color, 2)

        status_parts = [f"NCC:{best_score:.2f}", f"s={curr_scale:.2f}"]
        if road_lost:
            status_parts.append("RD-LOSS")
        elif rd_cooldown > 0:
            status_parts.append(f"RD-cd={rd_cooldown}")

        if road_jump:
            status_parts.append(f"JUMP!(j={road_motion.jump_count})")
        if road_motion.initialized:
            status_parts.append(f"X={road_motion.X:.1f}")
        cv2.putText(frame, " ".join(status_parts), (bx, by - 8),
                    FONT, 0.4, status_color, 1)

        # 道路中心线
        if road_detector.model.ready and len(road_detector.model.pts) > 1:
            cpts = np.array(road_detector.model.pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [cpts], False, (255, 255, 0), 1)

        # 道路预测点
        if consecutive_lost > 0 and pred_rp is not None:
            ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
            if ri is not None:
                cv2.circle(frame, (int(ri[0]), int(ri[1])), 6, C_YELLOW, -1)
                cv2.line(frame, (int(cx), int(cy)),
                         (int(ri[0]), int(ri[1])), C_YELLOW, 1)

        # 轨迹
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, C_RED, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, C_BLUE, -1)

        out.write(frame)
        road_video_out.write(_render_road_frame(
            road_traj,
            ncc_rp if ncc_rp is not None else (road_motion.X, road_motion.Z),
            pred_rp if consecutive_lost > 0 else None,
            consecutive_lost > 0, frame_idx))
        frames_written += 1
        frame_idx += 1

        if frame_idx % PROG_INTERVAL == 0:
            rstr = ""
            if road_motion.initialized:
                rstr = (f" road=({road_motion.X:.1f},{road_motion.Z:.1f})"
                        f" v=({road_motion.vx:.2f},{road_motion.vz:.2f})")
            ocstr = f" LOST({consecutive_lost})" if consecutive_lost > 0 else ""
            jstr = f" JUMPS={road_motion.jump_count}" if road_motion.jump_count > 0 else ""
            print(f"  帧 {frame_idx}: NCC={best_score:.2f} s={curr_scale:.2f}"
                  f"{ocstr}{jstr}{rstr}")

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
        py = int(canvas_h - 30
                 - (rz - z_min) / (z_max - z_min) * (canvas_h - 60))
        return px, py

    pts = [tp(p[0], p[1]) for p in road_traj]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (0, 0, 200), 1)
    for i, (px, py) in enumerate(pts):
        alpha = i / max(1, len(pts) - 1)
        cv2.circle(canvas, (px, py), 3,
                   (int(255 * (1 - alpha)), 0, int(255 * alpha)), -1)
    if pts:
        cv2.circle(canvas, pts[0], 7, (0, 255, 0), -1)
        cv2.circle(canvas, pts[-1], 7, (0, 0, 255), -1)

    cv2.putText(canvas, "X (lateral)", (canvas_w // 2 - 40, canvas_h - 10),
                FONT, 0.4, (100, 100, 100), 1)
    cv2.putText(canvas, "^ Z (depth)", (5, canvas_h - 10), FONT, 0.4, (100, 100, 100), 1)
    cv2.putText(canvas, "START", pts[0] if pts else (10, 30),
                FONT, 0.4, (0, 200, 0), 1)

    path = os.path.join(output_dir, "road_trajectory.png")
    cv2.imwrite(path, canvas)
    print(f"  轨迹图: {path}")


if __name__ == "__main__":
    main()
