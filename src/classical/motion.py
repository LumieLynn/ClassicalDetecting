import cv2
import numpy as np


class KalmanTracker:
    """恒速卡尔曼滤波器，用于目标位置预测。"""

    def __init__(self, dt=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        # 过程噪声 → 预测不确定性
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5.0
        # 测量噪声 → 观测可信度
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500.0

        self.initialized = False
        self.dt = dt

    def init(self, cx, cy):
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        """返回预测位置 (cx, cy) 和不确定性半径 (std_x, std_y)。"""
        if not self.initialized:
            return 0, 0, 50, 50
        predicted = self.kf.predict()
        cx, cy = predicted[0, 0], predicted[1, 0]
        P = self.kf.errorCovPre
        std_x = np.sqrt(P[0, 0]) * 3.0
        std_y = np.sqrt(P[1, 1]) * 3.0
        return cx, cy, max(std_x, 15.0), max(std_y, 15.0)

    def update(self, cx, cy):
        """用观测值更新滤波器。"""
        if not self.initialized:
            self.init(cx, cy)
            return
        self.kf.correct(np.array([[cx], [cy]], np.float32))

    @property
    def velocity(self):
        if not self.initialized:
            return 0.0, 0.0
        state = self.kf.statePost
        return float(state[2, 0]), float(state[3, 0])


class OpticalFlowTracker:
    """稀疏 LK 光流点跟踪 + 前向后向一致性检验。"""

    def __init__(self, max_corners=50, quality=0.01, min_dist=5):
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.prev_points = None
        self.prev_gray = None

    def init_roi(self, gray, bbox):
        """在 bbox 区域内提取 Shi-Tomasi 角点。"""
        x, y, w, h = bbox
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        pts = cv2.goodFeaturesToTrack(
            gray, self.max_corners, self.quality, self.min_dist, mask=mask
        )
        if pts is not None:
            self.prev_points = pts.reshape(-1, 2)
        else:
            self.prev_points = None
        self.prev_gray = gray.copy()

    def track(self, curr_gray):
        """跟踪角点，返回中值位移 (dx, dy) 和成功跟踪比例。"""
        if self.prev_points is None or self.prev_gray is None:
            return 0.0, 0.0, 0.0

        # 前向跟踪
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_points, None, **self.lk_params
        )
        if new_pts is None or status is None:
            return 0.0, 0.0, 0.0

        # 后向验证
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, new_pts, None, **self.lk_params
        )
        diff = np.linalg.norm(self.prev_points - back_pts, axis=1)
        consistent = (status.ravel() == 1) & (back_status.ravel() == 1) & (diff < 1.0)

        valid_pts = new_pts[consistent]
        if len(valid_pts) < 3:
            self.prev_gray = curr_gray.copy()
            return 0.0, 0.0, 0.0

        displacements = valid_pts - self.prev_points[consistent]
        median_dx = np.median(displacements[:, 0])
        median_dy = np.median(displacements[:, 1])
        ratio = len(valid_pts) / len(self.prev_points)

        self.prev_points = valid_pts
        self.prev_gray = curr_gray.copy()
        return median_dx, median_dy, ratio
