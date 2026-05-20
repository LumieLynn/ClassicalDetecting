import cv2
import numpy as np


class HistogramMatcher:
    """HSV 颜色直方图反向投影 + CamShift 目标定位。"""

    def __init__(self, h_bins=30, s_bins=32, v_channels=1):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.hist = None
        # V 通道只用前 v_channels 个 bin，减少光照敏感度
        self.v_bins = v_channels
        self.channels = [0, 1]
        self.hist_size = [h_bins, s_bins]
        self.ranges = [0, 180, 0, 256]

    def build_model(self, bgr_patch):
        if bgr_patch.dtype != np.uint8:
            bgr_patch = bgr_patch.astype(np.uint8)
        if len(bgr_patch.shape) == 2:
            bgr_patch = np.stack([bgr_patch] * 3, axis=-1)
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsv], self.channels, None, self.hist_size, self.ranges)
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        return self.hist

    def back_project(self, search_bgr):
        if self.hist is None:
            return None
        if search_bgr.dtype != np.uint8:
            search_bgr = search_bgr.astype(np.uint8)
        if len(search_bgr.shape) == 2:
            search_bgr = np.stack([search_bgr] * 3, axis=-1)
        hsv = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([hsv], self.channels, self.hist, self.ranges, 1)
        return prob

    def match(self, search_bgr, initial_bbox):
        """在 search_bgr 中定位目标，initial_bbox 为搜索起点。

        Returns: (score, (x, y, w, h))
        """
        if search_bgr.dtype != np.uint8:
            search_bgr = search_bgr.astype(np.uint8)
        if len(search_bgr.shape) == 2:
            search_bgr = np.stack([search_bgr] * 3, axis=-1)
        prob = self.back_project(search_bgr)
        if prob is None or prob.size == 0:
            return 0.0, initial_bbox

        x, y, w, h = initial_bbox
        track_box = (x, y, w, h)

        # CamShift 迭代
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        ret, track_box = cv2.CamShift(prob, track_box, term_crit)
        # 取 CamShift 结果的包围盒
        try:
            if ret is not None and len(ret) >= 2:
                rw_test, rh_test = ret[1]
                if rw_test > 0 and rh_test > 0:
                    rx, ry, rw, rh = cv2.boundingRect(ret)
                else:
                    rx, ry, rw, rh = x, y, w, h
            else:
                rx, ry, rw, rh = x, y, w, h
        except Exception:
            rx, ry, rw, rh = x, y, w, h

        if rw < 4 or rh < 4:
            return 0.0, initial_bbox

        # 置信度：收敛区域内概率图的均值
        patch_prob = prob[ry:ry+rh, rx:rx+rw]
        score = float(np.mean(patch_prob)) / 255.0
        score = np.clip(score, 0.0, 1.0)

        return score, (rx, ry, rw, rh)
