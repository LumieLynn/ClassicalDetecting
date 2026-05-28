"""Task 2 — 光流法实验 v2：双通道 KLT（目标 + 路面）→ 相对运动。

- 目标通道：跟踪目标边界框内的特征点
- 路面通道：跟踪路面蒙版内的特征点
- relative_flow = target_flow - road_flow = 目标的独立运动（相机运动已抵消）
- 遮挡期间：road_flow（实测）+ last_relative_flow（缓存）→ 数据驱动预测
- 运动显著性：用单应矩阵 warp 特征点，残差大的区域 → 增强 fg_prob

输出路径: output/improved/task2_optical_flow/
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
ROAD_SAMPLE_N = 60
ROAD_POLY_DEG = 3

# ======================== 道路坐标约束 ========================
ROAD_SPEED_EMA = 0.5
ROAD_X_THRESH = 25.0
ROAD_Z_THRESH = 50.0

# ======================== 跳变检测 ========================
JUMP_WINDOW = 8
JUMP_LAT_MULT = 2.5
JUMP_ARC_MULT = 2.5
JUMP_LAT_ABS_MIN = 25.0
JUMP_ARC_ABS_MIN = 50.0

# ======================== 中心线抖动抑制 ========================
RD_JITTER_THRESH = 8.0
RD_LOSS_THRESH = 25.0
RD_JITTER_EMA = 0.6
RD_NORMAL_EMA = 0.3
RD_COOLDOWN = 5

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

# ======================== 光流参数 ========================
OF_MIN_FEATURES = 8              # 单通道最少特征点数
OF_REFILL_INTERVAL = 15          # 特征点补充间隔
OF_VALID_RATIO_MIN = 0.3         # 低于此比例退化为纯恒速预测
OF_RELATIVE_FLOW_WEIGHT = 0.4    # 遮挡期间相对运动权重
OF_SALIENCY_WEIGHT = 0.5         # 运动显著性图与 fg_prob 融合权重
OF_LK_WIN_SIZE = (21, 21)
OF_LK_MAX_LEVEL = 3

# ======================== 通用 ========================
TRAJ_MAXLEN = 2000
PROG_INTERVAL = 50

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_BLUE = (255, 0, 0)
C_ORANGE = (255, 165, 0)
C_YELLOW = (0, 255, 255)
C_MAGENTA = (255, 0, 255)
C_CYAN = (255, 255, 0)


# ================================================================
#  道路路径模型
# ================================================================
class RoadPathModel:
    def __init__(self):
        self.pts = []
        self.arc = []
        self.poly = None
        self.ready = False

    def image_to_path(self, x, y):
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
#  道路路径检测器
# ================================================================
class RoadPathDetector:
    def __init__(self, img_w, img_h):
        self.w = img_w
        self.h = img_h
        self.model = RoadPathModel()
        self._prev_pts = None
        self._ema = 0.5
        self._road_mask = None

    def detect(self, gray, flow_dx=0.0, flow_dy=0.0):
        h, w = gray.shape
        y0 = int(h * 0.03)
        y1 = int(h * 0.97)
        roi = gray[y0:y1, :]

        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels < 2:
            return False
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        road_mask = (labels == largest_idx).astype(np.uint8) * 255

        def _extract_centerline(mask):
            pts = []
            for row in range(mask.shape[0]):
                cols = np.where(mask[row, :] > 0)[0]
                if len(cols) > 20:
                    pts.append(((cols[0] + cols[-1]) / 2.0, float(row + y0)))
            return pts

        raw = _extract_centerline(road_mask)

        if len(raw) >= 10:
            bottom_x = raw[-1][0]
            if bottom_x < w * 0.1 or bottom_x > w * 0.9:
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
                            raw = raw_inv

        if len(raw) < 10:
            return False

        n = min(ROAD_SAMPLE_N, len(raw))
        step = (len(raw) - 1) / max(n - 1, 1)
        pts = [raw[int(round(i * step))] for i in range(n)]

        if self._prev_pts is not None and len(self._prev_pts) == n:
            diffs = [abs(px - qx) for (px, _), (qx, _) in zip(pts, self._prev_pts)]
            mean_diff = sum(diffs) / len(diffs)
            if mean_diff > RD_LOSS_THRESH:
                return False
            wt = RD_NORMAL_EMA if mean_diff <= RD_JITTER_THRESH else RD_JITTER_EMA
            smooth = []
            for (px, py), (qx, qy) in zip(pts, self._prev_pts):
                # 光流 warp：旧中心线 + 路面实测位移 → 对齐当前帧
                wqx = qx + flow_dx
                wqy = qy + flow_dy
                smooth.append((wt * wqx + (1 - wt) * px, wt * wqy + (1 - wt) * py))
            pts = smooth
        self._prev_pts = [(p[0], p[1]) for p in pts]

        arc = [0.0]
        for i in range(1, len(pts)):
            d = np.sqrt(
                (pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2
            )
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
        self._road_mask = road_mask
        self._road_mask_y0 = y0
        return True


# ================================================================
#  道路运动模型
# ================================================================
class RoadMotionModel:
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
        if len(self._lat_history) < 3:
            return False, 0.0, 0.0, 0.0, 0.0

        prev_X = self._lat_history[-1]
        prev_Z = self._arc_history[-1]
        dX = abs(X - prev_X)
        dZ = abs(Z - prev_Z)

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
#  双通道光流辅助类
# ================================================================
class OpticFlowHelper:
    """双通道 KLT 光流：目标特征 + 路面特征 → 相对运动估计。

    目标通道 (target_*):  跟踪目标边界框内的特征点 → 目标的外观运动
    路面通道 (road_*):    跟踪路面蒙版内的特征点   → 相机自运动
    relative_flow = target_flow - road_flow → 目标相对于路面的独立运动

    遮挡期间目标特征丢失，但路面特征仍在：
    predicted = prev_pos + road_flow（实测相机运动） + last_relative_flow（目标独立运动）
    """

    def __init__(self):
        self.lk_params = dict(
            winSize=OF_LK_WIN_SIZE,
            maxLevel=OF_LK_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.ft_params_target = dict(
            maxCorners=30, qualityLevel=0.08, minDistance=5, blockSize=5,
        )
        self.ft_params_road = dict(
            maxCorners=50, qualityLevel=0.05, minDistance=10, blockSize=7,
        )

        # 目标通道
        self.target_pts = None      # (N, 1, 2) 当前帧
        self.target_valid = False
        self.target_flow = (0.0, 0.0)

        # 路面通道
        self.road_pts = None        # (N, 1, 2) 当前帧
        self._road_prev = None      # (N, 1, 2) 上一帧（用于显著性）
        self.road_valid = False
        self.road_flow = (0.0, 0.0)

        # 相对运动
        self.relative_flow = (0.0, 0.0)       # target - road（当前帧）
        self.last_relative_flow = (0.0, 0.0)  # 缓存（遮挡期间使用）
        self._rel_flow_history = deque(maxlen=5)  # 平滑

        self.prev_gray = None
        self._refill_counter = 0

    # ---- 单通道跟踪 ----
    def _track_channel(self, gray, prev_pts):
        """跟踪一组特征点，返回 (median_dx, median_dy, valid, kept_pts)。"""
        if prev_pts is None or len(prev_pts) < 3:
            return 0.0, 0.0, False, None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params)

        if next_pts is None or status is None:
            return 0.0, 0.0, False, None

        status = status.flatten()
        vp = prev_pts[status == 1]
        vn = next_pts[status == 1]

        if len(vp) < 3:
            return 0.0, 0.0, False, vn.reshape(-1, 1, 2).astype(np.float32) if len(vn) > 0 else None

        flows = vn - vp
        med_dx = float(np.median(flows[:, 0, 0]))
        med_dy = float(np.median(flows[:, 0, 1]))

        # 过滤漂移点
        dists = np.sqrt((flows[:, 0, 0] - med_dx)**2 + (flows[:, 0, 1] - med_dy)**2)
        med_dist = np.median(dists)
        thresh = max(med_dist * 2.5, 3.0)
        inliers = dists < thresh

        if inliers.sum() >= 3:
            med_dx = float(np.median(flows[inliers, 0, 0]))
            med_dy = float(np.median(flows[inliers, 0, 1]))
            kept = vn[inliers].reshape(-1, 1, 2).astype(np.float32)
        else:
            kept = vn.reshape(-1, 1, 2).astype(np.float32)

        return med_dx, med_dy, True, kept

    # ---- 特征提取 ----
    def extract_target_features(self, gray, bbox):
        """在目标边界框内提取特征点。"""
        h, w = gray.shape
        x, y, bw, bh = bbox
        # 缩小 bbox 边缘避免背景污染
        margin = max(2, min(bw, bh) // 8)
        mx = max(0, x + margin)
        my = max(0, y + margin)
        mw = min(bw - 2 * margin, w - mx)
        mh = min(bh - 2 * margin, h - my)
        if mw < 10 or mh < 10:
            return False

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[my:my + mh, mx:mx + mw] = 255
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.ft_params_target)
        if pts is not None and len(pts) >= OF_MIN_FEATURES:
            self.target_pts = pts
            self.target_valid = True
            return True
        return False

    def extract_road_features(self, gray, road_mask, cx, cy):
        """在路面蒙版内、目标周围提取特征点。"""
        h, w = gray.shape
        window = 100
        x1 = max(0, int(cx - window))
        y1 = max(0, int(cy - window))
        x2 = min(w, int(cx + window))
        y2 = min(h, int(cy + window))

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        if road_mask is not None:
            if road_mask.shape[0] < h:
                rp_full = np.zeros((h, w), dtype=np.uint8)
                mh_ = road_mask.shape[0]
                y_off = int(h * 0.03)
                rp_full[y_off: y_off + mh_, :] = road_mask
                roi_mask = cv2.bitwise_and(roi_mask, rp_full)
            else:
                roi_mask = cv2.bitwise_and(roi_mask, road_mask)

        pts = cv2.goodFeaturesToTrack(gray, mask=roi_mask, **self.ft_params_road)
        if pts is not None and len(pts) >= OF_MIN_FEATURES:
            self.road_pts = pts
            self.road_valid = True
            return True
        return False

    # ---- 主跟踪 ----
    def track(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return

        # 跟踪目标通道
        tdx, tdy, t_ok, t_kept = self._track_channel(gray, self.target_pts)
        if t_ok and t_kept is not None:
            self.target_flow = (tdx, tdy)
            self.target_pts = t_kept
            self.target_valid = True
        else:
            self.target_valid = False
            self.target_flow = (0.0, 0.0)
            self.target_pts = t_kept  # 可能只剩少量点

        # 保存上一帧位置（用于显著性计算）
        self._road_prev = self.road_pts.copy() if self.road_pts is not None else None
        # 跟踪路面通道
        rdx, rdy, r_ok, r_kept = self._track_channel(gray, self.road_pts)
        if r_ok and r_kept is not None:
            self.road_flow = (rdx, rdy)
            self.road_pts = r_kept
            self.road_valid = True
        else:
            self.road_valid = False
            self.road_flow = (0.0, 0.0)
            self.road_pts = r_kept

        # 计算相对运动（相机运动抵消）
        if self.target_valid and self.road_valid:
            rel_dx = self.target_flow[0] - self.road_flow[0]
            rel_dy = self.target_flow[1] - self.road_flow[1]
            self._rel_flow_history.append((rel_dx, rel_dy))
            # 平滑
            if len(self._rel_flow_history) > 0:
                rel_dx = float(np.median([f[0] for f in self._rel_flow_history]))
                rel_dy = float(np.median([f[1] for f in self._rel_flow_history]))
            self.relative_flow = (rel_dx, rel_dy)
            self.last_relative_flow = (rel_dx, rel_dy)

        self.prev_gray = gray.copy()
        self._refill_counter += 1

    # ---- 遮挡期间预测 ----
    def predict_target_position(self, prev_cx, prev_cy):
        """遮挡期间：实测相机运动 + 缓存的目标相对运动 → 预测目标位置。"""
        if self.road_valid and self.last_relative_flow != (0.0, 0.0):
            pred_cx = prev_cx + self.road_flow[0] + self.last_relative_flow[0]
            pred_cy = prev_cy + self.road_flow[1] + self.last_relative_flow[1]
            return pred_cx, pred_cy
        return prev_cx, prev_cy

    @property
    def needs_target_refill(self):
        ok = self.target_pts is not None and len(self.target_pts) >= OF_MIN_FEATURES
        return not ok or self._refill_counter >= OF_REFILL_INTERVAL

    @property
    def needs_road_refill(self):
        ok = self.road_pts is not None and len(self.road_pts) >= OF_MIN_FEATURES
        return not ok or self._refill_counter >= OF_REFILL_INTERVAL

    def reset(self):
        self.target_pts = None
        self.target_valid = False
        self.target_flow = (0.0, 0.0)
        self.road_pts = None
        self.road_valid = False
        self.road_flow = (0.0, 0.0)
        self.relative_flow = (0.0, 0.0)
        self.last_relative_flow = (0.0, 0.0)
        self._rel_flow_history.clear()
        self.prev_gray = None
        self._refill_counter = 0


# ================================================================
#  运动显著性图（基于流场残差）
# ================================================================
def compute_flow_saliency(prev_pts, curr_pts, H, img_shape):
    """用单应矩阵 warp 特征点，残差大的区域 → 运动显著性图。

    prev_pts: 上一帧的跟踪点 (N, 2)
    curr_pts: 当前帧的跟踪点 (N, 2)
    H: 上一帧到当前帧的单应矩阵（来自 SceneMotionEstimator）
    img_shape: (h, w)

    返回: saliency_map (h, w) float32 [0,1]
    """
    if H is None or prev_pts is None or len(prev_pts) < 5:
        return np.zeros(img_shape, dtype=np.float32)

    # 确保是 (N, 2) 格式
    prev_pts = np.squeeze(prev_pts)
    curr_pts = np.squeeze(curr_pts)
    if prev_pts.ndim != 2 or prev_pts.shape[1] != 2:
        return np.zeros(img_shape, dtype=np.float32)

    # 用 H 将 prev_pts warp 到当前帧 → expected
    prev_hom = np.hstack([prev_pts, np.ones((len(prev_pts), 1))])
    expected = prev_hom @ H.T
    expected = expected[:, :2] / (expected[:, 2:3] + 1e-8)

    # 残差 = 实际位置 - 期望位置
    residuals = np.sqrt(np.sum((curr_pts - expected) ** 2, axis=1))

    # 在图上绘制残差（稀疏 → 模糊 → 显著性图）
    h, w = img_shape
    saliency = np.zeros((h, w), dtype=np.float32)
    for (ex, ey), r in zip(expected, residuals):
        px, py = int(np.clip(ex, 0, w - 1)), int(np.clip(ey, 0, h - 1))
        weight = min(r / 10.0, 1.0)  # 残差 > 10px 视为高显著
        cv2.circle(saliency, (px, py), max(3, int(r / 3)),
                   weight, -1, cv2.LINE_AA)

    saliency = cv2.GaussianBlur(saliency, (31, 31), 0)
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency


def image_flow_to_road_delta(median_dx, median_dy, cx, cy, road_model):
    """图像空间光流 -> 道路坐标变化量 (dlat, darc)。"""
    if not road_model.ready:
        return 0.0, 0.0

    eps = 2.0
    rp0 = road_model.image_to_path(cx, cy)
    rpx = road_model.image_to_path(cx + eps, cy)
    rpy = road_model.image_to_path(cx, cy + eps)

    if rp0 is None or rpx is None or rpy is None:
        return 0.0, 0.0

    J11 = (rpx[0] - rp0[0]) / eps
    J12 = (rpy[0] - rp0[0]) / eps
    J21 = (rpx[1] - rp0[1]) / eps
    J22 = (rpy[1] - rp0[1]) / eps

    dlat = J11 * median_dx + J12 * median_dy
    darc = J21 * median_dx + J22 * median_dy

    return dlat, darc


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
#  统一目标搜索
# ================================================================
def search_target(
    gray, fg_prob, road_mask, pred_cx, pred_cy,
    curr_tmpl, base_w, base_h, curr_scale,
    img_w, img_h, pad, scales, min_size,
):
    if road_mask is not None and road_mask.shape[0] < img_h:
        rp = np.zeros((img_h, img_w), dtype=np.float32)
        mh = road_mask.shape[0]
        y_off = int(img_h * 0.03)
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
        if sw < min_size or sh < min_size or sw > search.shape[1] or sh > search.shape[0]:
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
    ellipse = cv2.fitEllipse(cnt)
    (ecx, ecy), (major, minor), angle = ellipse
    a, b = major / 2, minor / 2
    iw = a * np.sqrt(2)
    ih = b * np.sqrt(2)
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

    out_dir = "output/improved/task2_optical_flow"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "auto_template.png"), gray_patch)
    print(f"模板: {gray_patch.shape[1]}x{gray_patch.shape[0]}")
    return gray_patch


# ================================================================
#  主程序
# ================================================================
def main():
    np.random.seed(0)
    cv2.setRNGSeed(0)

    video_path = "data/task2/大疆无人机航拍视频.mp4"
    ref_image_path = "data/task2/大疆无人机航拍视频目标.png"
    output_dir = "output/improved/task2_optical_flow"
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
    prev_gray_rev = scan_grays[anchor_idx]

    for i in range(anchor_idx - 1, -1, -1):
        gray = scan_grays[i]
        bx, by, bw, bh = history.get(i + 1, anchor_box)
        cx, cy = bx + bw // 2, by + bh // 2

        frame_diff = cv2.absdiff(gray, prev_gray_rev)
        _, fg_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        fg_prob_rev = cv2.GaussianBlur(fg_mask.astype(np.float32) / 255.0, (21, 21), 0)
        prev_gray_rev = gray
        road_mask_rev = road_detector._road_mask
        best_score, best_match = search_target(
            gray, fg_prob_rev, road_mask_rev, cx, cy,
            base_template, base_w, base_h, curr_scale,
            width, height, REV_PAD, REV_SCALES, REV_MIN_SIZE,
        )
        if best_score > REV_MIN_SCORE:
            _, _, _, _, curr_scale = best_match
            history[i] = (best_match[0], best_match[1], best_match[2], best_match[3])
        else:
            history[i] = history.get(i + 1, anchor_box)

    print("反向追踪完成。")

    # ==================== 阶段三：正向追踪（双通道光流增强）====================
    print("阶段三：正向追踪（双通道 KLT 光流 + 运动显著性）...")
    out = cv2.VideoWriter(
        os.path.join(output_dir, "tracked_result.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )

    ROAD_VIDEO_W = 500
    ROAD_VIDEO_H = 500
    road_video_out = cv2.VideoWriter(
        os.path.join(output_dir, "road_trajectory.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (ROAD_VIDEO_W, ROAD_VIDEO_H),
    )

    scene_motion = SceneMotionEstimator()
    kalman = KalmanTracker()
    oflow = OpticFlowHelper()
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
    in_occlusion = False
    is_tracking = True
    occ_good = [0]
    ncc_cooldown = 0
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
            py = int(
                ROAD_VIDEO_H - 20 - (rz - z_min) / (z_max - z_min) * (ROAD_VIDEO_H - 40)
            )
            return px, py

        if len(rtraj) > 1:
            pts = [to_px(p[0], p[1]) for p in rtraj]
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, (0, 0, 200), 1)
            for i, (px, py) in enumerate(pts):
                alpha = i / max(1, len(pts) - 1)
                cv2.circle(
                    canvas, (px, py), 2,
                    (int(255 * (1 - alpha)), 0, int(255 * alpha)), -1,
                )

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
        cv2.putText(
            canvas, "X (lateral) ->", (ROAD_VIDEO_W - 120, ROAD_VIDEO_H - 8),
            FONT, 0.35, (100, 100, 100), 1,
        )
        cv2.putText(
            canvas, "^ Z (depth)", (5, ROAD_VIDEO_H - 8), FONT, 0.35, (100, 100, 100), 1,
        )
        if occ:
            cv2.putText(
                canvas, "OCC", (ROAD_VIDEO_W - 50, 20), FONT, 0.5, (0, 0, 255), 2,
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

        # --- 道路检测更新 ---
        road_lost = False
        if rd_cooldown > 0:
            rd_cooldown -= 1
        elif frame_idx % ROAD_DETECT_INTERVAL == 0:
            rdx, rdy = oflow.road_flow if oflow.road_valid else (0.0, 0.0)
            if road_detector.detect(gray, flow_dx=rdx, flow_dy=rdy):
                pass
            else:
                road_lost = road_detector.model.ready
                rd_cooldown = RD_COOLDOWN

        # --- 前景概率图 + 单应矩阵 ---
        fg_prob, H, _ = scene_motion.align_and_diff(gray)

        # --- 双通道光流跟踪 ---
        oflow.track(gray)

        # --- 运动显著性图（当前版本跳过，点集长度可能因 KLT 丢失不一致）---
        # 双通道相对运动仍正常工作

        # --- 锚点帧 ---
        if frame_idx == anchor_idx:
            bx, by, bw, bh = anchor_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_RED, 2)
            cv2.putText(frame, "ANCHOR", (bx, by - 10), FONT, 0.7, C_RED, 2)
            trajectory.append((bx + bw // 2, by + bh // 2))

            # 初始化双通道光流特征
            oflow.extract_target_features(gray, anchor_box)
            if road_detector._road_mask is not None:
                oflow.extract_road_features(gray, road_detector._road_mask, cx, cy)
            # 手动设置 prev_gray
            if oflow.prev_gray is None:
                oflow.prev_gray = gray.copy()

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

        # --- 预测搜索中心（卡尔曼 + 道路预测 + 光流相对运动融合）---
        kf_cx, kf_cy, std_x, std_y = kalman.predict()

        if road_motion.initialized and road_detector.model.ready and not in_occlusion:
            rpx, rpz = road_motion.predict(dt=1)
            ri = road_detector.model.path_to_image(rpx, rpz)
            if ri is not None and 0 <= ri[0] < width and 0 <= ri[1] < height:
                pred_cx = 0.5 * kf_cx + 0.5 * ri[0]
                pred_cy = 0.5 * kf_cy + 0.5 * ri[1]
            else:
                pred_cx, pred_cy = kf_cx, kf_cy
        elif in_occlusion and oflow.road_valid:
            # 遮挡期间：光流预测融合卡尔曼
            flow_cx, flow_cy = oflow.predict_target_position(_prev_cx, _prev_cy)
            pred_cx = 0.3 * kf_cx + 0.7 * flow_cx
            pred_cy = 0.3 * kf_cy + 0.7 * flow_cy
        else:
            pred_cx, pred_cy = kf_cx, kf_cy

        # --- 丢失状态 ---
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

        # --- 搜索区域 ---
        curr_w = int(base_w * curr_scale)
        curr_h = int(base_h * curr_scale)
        if in_occlusion:
            pad = max(FWD_PAD // 2, int(max(curr_w, curr_h) * 0.6))
        else:
            pad = max(FWD_PAD, int(max(curr_w, curr_h) * 1.2))

        # --- 统一搜索（使用增强后的 fg_prob + saliency）---
        road_mask_fwd = road_detector._road_mask
        best_ncc, best_match = search_target(
            gray, fg_prob, road_mask_fwd, pred_cx, pred_cy,
            curr_tmpl, base_w, base_h, curr_scale,
            width, height, pad, FWD_SCALES, FWD_MIN_SIZE,
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

                # 遮挡期间：用相对光流修正道路坐标预测
                if in_occlusion and oflow.road_valid:
                    flow_dlat, flow_darc = image_flow_to_road_delta(
                        oflow.relative_flow[0], oflow.relative_flow[1],
                        pred_cx, pred_cy, road_detector.model)
                    flow_dlat = max(-30.0, min(30.0, flow_dlat))
                    flow_darc = max(-60.0, min(60.0, flow_darc))
                    pred_rp = (
                        pred_rp[0] + OF_RELATIVE_FLOW_WEIGHT * flow_dlat,
                        pred_rp[1] + OF_RELATIVE_FLOW_WEIGHT * flow_darc,
                    )

                dx = abs(ncc_rp[0] - pred_rp[0])
                dz = abs(ncc_rp[1] - pred_rp[1])

                gap_factor = 1.0 + consecutive_lost * 0.3
                x_thresh = ROAD_X_THRESH * gap_factor
                z_thresh = ROAD_Z_THRESH * gap_factor

                road_ok = dx < x_thresh and dz < z_thresh

                road_jump, jx, jz, jx_th, jz_th = road_motion.check_jump(
                    ncc_rp[0], ncc_rp[1])
                if road_jump:
                    road_ok = False

                if not road_ok:
                    ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                    if ri is not None:
                        ncc_cx, ncc_cy = ri

        # ============================================================
        #  跟踪决策
        # ============================================================
        track_healthy = best_ncc > FWD_NCC_GOOD or (
            best_ncc > FWD_NCC_MARGINAL and road_ok
        )
        if track_healthy and (road_ok or not road_motion.initialized):
            if in_occlusion:
                occ_good[0] += 1
                if occ_good[0] >= 5:
                    in_occlusion = False
                    consecutive_lost = 0
                    occ_good[0] = 0
                    ncc_cooldown = 5
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

            # 目标可见：补充双通道特征
            if oflow.needs_target_refill:
                oflow.extract_target_features(gray, (x, y, w, h))
            if oflow.needs_road_refill and road_detector._road_mask is not None:
                oflow.extract_road_features(gray, road_detector._road_mask, cx, cy)

            if ncc_rp is not None:
                if ncc_cooldown > 0:
                    ncc_cooldown -= 1
                    if ncc_cooldown == 0:
                        bx_t = int(cx - w // 2)
                        by_t = int(cy - h // 2)
                        bx_t = max(0, bx_t)
                        by_t = max(0, by_t)
                        ew = min(w, gray.shape[1] - bx_t)
                        eh = min(h, gray.shape[0] - by_t)
                        if ew >= TPL_MIN_PATCH and eh >= TPL_MIN_PATCH:
                            patch = gray[by_t : by_t + eh, bx_t : bx_t + ew]
                            curr_tmpl = cv2.resize(patch, (base_w, base_h)).astype(
                                np.float32)
                            last_tpl_update = frame_idx
                else:
                    road_motion.update(ncc_rp[0], ncc_rp[1])
                road_traj.append(ncc_rp)

        elif best_ncc > FWD_NCC_GOOD and road_motion.initialized and not road_ok:
            if road_jump:
                if not in_occlusion:
                    in_occlusion = True
                    consecutive_lost = 1
                    occ_good[0] = 0
                else:
                    consecutive_lost += 1
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
                if in_occlusion:
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
                    in_occlusion = False
                    consecutive_lost = 0
                    final_cx, final_cy = x + w // 2, y + h // 2
                    kalman.update(final_cx, final_cy)
                    cx, cy = final_cx, final_cy
                    _prev_cx, _prev_cy = cx, cy
                    target_w, target_h = w, h
                    status_color = (0, 255, 255)

        else:
            # NCC 低 -> 遮挡
            if not in_occlusion:
                in_occlusion = True
                consecutive_lost = 1
                occ_good[0] = 0
            else:
                consecutive_lost += 1

            # 融合：道路预测 + 光流相对运动预测 + 卡尔曼
            if pred_rp is not None:
                ri = road_detector.model.path_to_image(pred_rp[0], pred_rp[1])
                if ri is not None and oflow.road_valid:
                    # 光流预测也参与
                    flow_cx, flow_cy = oflow.predict_target_position(_prev_cx, _prev_cy)
                    final_cx = 0.3 * ri[0] + 0.3 * flow_cx + 0.4 * kf_cx
                    final_cy = 0.3 * ri[1] + 0.3 * flow_cy + 0.4 * kf_cy
                elif ri is not None:
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
                print(f"帧 {frame_idx}: 连续 NCC 低 ({consecutive_lost})，丢失")

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
            status_parts.append(f"JUMP!({road_motion.jump_count})")
        if road_motion.initialized:
            status_parts.append(f"X={road_motion.X:.1f}")
        if oflow.road_valid and oflow.target_valid:
            status_parts.append(
                f"rel=({oflow.relative_flow[0]:.1f},{oflow.relative_flow[1]:.1f})")
        cv2.putText(
            frame, " ".join(status_parts), (bx, by - 8), FONT, 0.4, status_color, 1
        )

        # 道路中心线
        if road_detector.model.ready and len(road_detector.model.pts) > 1:
            cpts = np.array(road_detector.model.pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [cpts], False, (255, 255, 0), 1)

        # 光流可视化
        # 目标特征点（红色）
        if oflow.target_pts is not None and len(oflow.target_pts) > 0:
            for pt in oflow.target_pts:
                px, py = int(pt[0, 0]), int(pt[0, 1])
                cv2.circle(frame, (px, py), 3, C_RED, -1)
        # 路面特征点（绿色）
        if oflow.road_pts is not None and len(oflow.road_pts) > 0:
            for pt in oflow.road_pts:
                px, py = int(pt[0, 0]), int(pt[0, 1])
                cv2.circle(frame, (px, py), 2, C_GREEN, -1)
        # 相对运动箭头（品红）：从目标中心出发
        if oflow.road_valid and oflow.target_valid:
            rx, ry = oflow.relative_flow
            arrow_end = (int(pred_cx + rx * 2), int(pred_cy + ry * 2))
            cv2.arrowedLine(frame, (int(pred_cx), int(pred_cy)), arrow_end,
                           C_MAGENTA, 2, tipLength=0.3)

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
                in_occlusion, frame_idx,
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
            nt = len(oflow.target_pts) if oflow.target_pts is not None else 0
            nr = len(oflow.road_pts) if oflow.road_pts is not None else 0
            rlx, rly = oflow.relative_flow
            ofstr = f" OF(t={nt},r={nr},rel=({rlx:.1f},{rly:.1f}))"
            print(
                f"  帧 {frame_idx}: NCC={best_ncc:.2f} s={curr_scale:.2f}"
                f"{ocstr}{jstr}{ofstr}{rstr}"
            )

    if len(road_traj) > 2:
        _save_road_trajectory_plot(road_traj, output_dir)

    cap.release()
    out.release()
    road_video_out.release()
    print("完成。")
    print(f"  跟踪视频: {os.path.join(output_dir, 'tracked_result.mp4')}")
    print(f"  道路轨迹: {os.path.join(output_dir, 'road_trajectory.mp4')}")


def _save_road_trajectory_plot(road_traj, output_dir):
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
        canvas, "X (lateral)", (canvas_w // 2 - 40, canvas_h - 10),
        FONT, 0.4, (100, 100, 100), 1,
    )
    cv2.putText(
        canvas, "^ Z (depth)", (5, canvas_h - 10), FONT, 0.4, (100, 100, 100), 1,
    )
    cv2.putText(
        canvas, "START", pts[0] if pts else (10, 30), FONT, 0.4, (0, 200, 0), 1,
    )

    path = os.path.join(output_dir, "road_trajectory.png")
    cv2.imwrite(path, canvas)
    print(f"  轨迹图: {path}")


if __name__ == "__main__":
    main()
