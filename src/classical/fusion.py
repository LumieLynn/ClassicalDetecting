import numpy as np
from .ncc import fast_ncc_match
from .features import FeatureMatcher
from .motion import KalmanTracker
from .edge_match import EdgeMatcher
from .histogram import HistogramMatcher


class MultiCueTracker:
    """多线索融合跟踪器。

    融合策略：级联 + 自适应加权投票。
    每条线索独立匹配，结果按置信度加权融合。
    权重根据历史一致性动态调整。
    """

    def __init__(self, template, enabled_cues=None,
                 score_threshold=0.4, lost_frames_max=30):
        self.base_template = template.astype(np.float32)
        self.base_w = template.shape[1]
        self.base_h = template.shape[0]

        self.enabled = enabled_cues or ["ncc"]
        self.score_threshold = score_threshold
        self.lost_frames_max = lost_frames_max

        # 初始化各匹配器
        self._ncc_scale_factors = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
        if "sift" in self.enabled or "orb" in self.enabled:
            method = "SIFT" if "sift" in self.enabled else "ORB"
            self.feature_matcher = FeatureMatcher(method=method)
        if "edge" in self.enabled:
            self.edge_matcher = EdgeMatcher()
        if "hist" in self.enabled:
            self.hist_matcher = HistogramMatcher()
            # 从模板建颜色直方图模型
            tmpl_uint8 = np.clip(template, 0, 255).astype(np.uint8)
            tmpl_bgr = np.stack([tmpl_uint8] * 3, axis=-1)
            self.hist_matcher.build_model(tmpl_bgr)

        self.kalman = KalmanTracker()

        # 各线索权重（动态调整）
        self.cue_weights = {}
        default_weight = 1.0 / max(1, len(self.enabled))
        for cue in self.enabled:
            self.cue_weights[cue] = default_weight
        self._weight_ema_alpha = 0.9

        # 状态
        self.curr_scale = 1.0
        self.scale_ema_alpha = 0.5
        self.consecutive_lost = 0
        self.initialized = False

    def init_position(self, cx, cy, scale=1.0):
        """在首帧设置初始位置。"""
        self.kalman.init(cx, cy)
        self.curr_scale = scale
        self.initialized = True

    def match_full_frame(self, gray, scales=None):
        """全图搜索匹配（用于锚点扫描 / 丢失恢复）。

        Returns: (score, (x, y, w, h, scale))
        """
        if scales is None:
            scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

        best_score = -1.0
        best_result = None

        for s in scales:
            sw = int(self.base_w * s)
            sh = int(self.base_h * s)
            if sw < 8 or sh < 8 or sw > gray.shape[1] or sh > gray.shape[0]:
                continue
            scaled = self._resize_template(self.base_template, sw, sh)
            score, (lx, ly) = fast_ncc_match(gray, scaled)
            if score > best_score:
                best_score = score
                best_result = (lx, ly, sw, sh, s)

        if best_result is None:
            return -1.0, (0, 0, 0, 0, 1.0)
        return best_score, best_result

    def track(self, frame_gray):
        """逐帧跟踪。

        Args:
            frame_gray: 当前帧灰度图 [H, W]

        Returns:
            (bbox, confidence, info)
            bbox:       (x, y, w, h) 在 frame_gray 坐标系中
            confidence: float
            info:       dict {cue_name: score, ...} 各线索贡献
        """
        if not self.initialized:
            return (0, 0, 0, 0), 0.0, {}

        H, W = frame_gray.shape

        # 1. 卡尔曼预测搜索中心
        pred_cx, pred_cy, std_x, std_y = self.kalman.predict()
        curr_w = int(self.base_w * self.curr_scale)
        curr_h = int(self.base_h * self.curr_scale)

        # 搜索区域：卡尔曼不确定性 + 目标尺寸 + 最小 margin
        pad_x = int(max(std_x, curr_w * 0.8, curr_w))
        pad_y = int(max(std_y, curr_h * 0.8, curr_h))
        x1 = max(0, int(pred_cx - pad_x))
        y1 = max(0, int(pred_cy - pad_y))
        x2 = min(W, int(pred_cx + pad_x))
        y2 = min(H, int(pred_cy + pad_y))
        search = frame_gray[y1:y2, x1:x2]

        if search.size == 0:
            self.consecutive_lost += 1
            bbox = (int(pred_cx - curr_w/2), int(pred_cy - curr_h/2), curr_w, curr_h)
            return bbox, 0.0, {}

        # 2. 运行各线索
        proposals = {}   # cue_name -> (cx, cy, w, h, confidence)
        scores = {}

        # --- NCC ---
        if "ncc" in self.enabled:
            best_ncc = -1.0
            best_ncc_match = None
            for sf in self._ncc_scale_factors:
                s = self.curr_scale * sf
                sw = int(self.base_w * s)
                sh = int(self.base_h * s)
                if sw < 6 or sh < 6 or sw > search.shape[1] or sh > search.shape[0]:
                    continue
                scaled = self._resize_template(self.base_template, sw, sh)
                score, (lx, ly) = fast_ncc_match(search, scaled)
                if score > best_ncc:
                    best_ncc = score
                    best_ncc_match = (lx, ly, sw, sh)
            if best_ncc_match and best_ncc > self.score_threshold * 0.6:
                lx, ly, sw, sh = best_ncc_match
                bx, by = x1 + lx, y1 + ly
                proposals["ncc"] = (bx + sw//2, by + sh//2, sw, sh)
                scores["ncc"] = best_ncc

        # --- 特征点 ---
        feature_cue = "sift" if "sift" in self.enabled else ("orb" if "orb" in self.enabled else None)
        if feature_cue:
            score, (fx, fy, fw, fh) = self.feature_matcher.match(search, self.base_template)
            if score > 0.3:
                bx, by = x1 + fx, y1 + fy
                proposals[feature_cue] = (bx + fw//2, by + fh//2, fw, fh)
                scores[feature_cue] = score

        # --- 边缘 ---
        if "edge" in self.enabled:
            edge_template = self._resize_template(self.base_template, curr_w, curr_h)
            score, (lx, ly) = self.edge_matcher.match(search, edge_template)
            if score > 0.1:
                bx, by = x1 + lx, y1 + ly
                proposals["edge"] = (bx + curr_w//2, by + curr_h//2, curr_w, curr_h)
                scores["edge"] = score

        # --- 颜色直方图 ---
        if "hist" in self.enabled:
            search_bgr = np.stack([search] * 3, axis=-1).astype(np.uint8)
            init_bbox = (pad_x - curr_w//2, pad_y - curr_h//2, curr_w, curr_h)
            init_bbox = (max(0, init_bbox[0]), max(0, init_bbox[1]), curr_w, curr_h)
            score, (hx, hy, hw, hh) = self.hist_matcher.match(search_bgr, init_bbox)
            if score > 0.1:
                bx, by = x1 + hx, y1 + hy
                proposals["hist"] = (bx + hw//2, by + hh//2, hw, hh)
                scores["hist"] = score

        # 3. 融合
        if not proposals:
            self.consecutive_lost += 1
            cx, cy = pred_cx, pred_cy
            bbox = (int(cx - curr_w/2), int(cy - curr_h/2), curr_w, curr_h)
            info = {"lost": self.consecutive_lost}
            if self.consecutive_lost >= self.lost_frames_max:
                self.initialized = False
                info["recovery_needed"] = True
            return bbox, 0.0, info

        self.consecutive_lost = 0

        # 加权融合位置
        total_w = 0.0
        fused_cx, fused_cy = 0.0, 0.0
        for cue, (cx, cy, w, h) in proposals.items():
            weight = scores[cue] * self.cue_weights[cue]
            fused_cx += cx * weight
            fused_cy += cy * weight
            total_w += weight
        if total_w > 0:
            fused_cx /= total_w
            fused_cy /= total_w
        else:
            fused_cx, fused_cy = pred_cx, pred_cy

        # 融合尺度：取 NCC（最可靠）或特征点的尺度
        if "ncc" in proposals:
            _, _, fused_w, fused_h = proposals["ncc"]
            fused_scale = fused_w / self.base_w
        elif feature_cue and feature_cue in proposals:
            _, _, fused_w, fused_h = proposals[feature_cue]
            fused_scale = fused_w / self.base_w
        else:
            fused_w, fused_h = curr_w, curr_h
            fused_scale = self.curr_scale

        # 尺度 EMA 平滑
        self.curr_scale = (self.scale_ema_alpha * self.curr_scale
                           + (1 - self.scale_ema_alpha) * fused_scale)

        # 卡尔曼更新
        self.kalman.update(fused_cx, fused_cy)

        # 4. 更新线索权重：偏离融合结果越远 → 降权
        for cue in proposals:
            cx, cy, _, _ = proposals[cue]
            dist = np.sqrt((cx - fused_cx)**2 + (cy - fused_cy)**2)
            agreement = np.exp(-dist / max(fused_w, fused_h))
            self.cue_weights[cue] = (
                self._weight_ema_alpha * self.cue_weights[cue]
                + (1 - self._weight_ema_alpha) * agreement
            )

        # 归一化权重
        w_sum = sum(self.cue_weights.values())
        if w_sum > 0:
            for k in self.cue_weights:
                self.cue_weights[k] /= w_sum

        # 最终置信度：取最高的那条
        final_conf = max(scores.values()) if scores else 0.0

        x = int(fused_cx - fused_w / 2)
        y = int(fused_cy - fused_h / 2)
        bbox = (x, y, fused_w, fused_h)

        return bbox, final_conf, scores

    @staticmethod
    def _resize_template(tmpl, w, h):
        import cv2
        return cv2.resize(tmpl, (w, h))
