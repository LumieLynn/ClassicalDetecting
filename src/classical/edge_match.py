import cv2
import numpy as np


class EdgeMatcher:
    """Canny 边缘 + 倒角距离匹配，用积分图加速。"""

    def __init__(self, canny_low=50, canny_high=150):
        self.canny_low = canny_low
        self.canny_high = canny_high

    def _edge_map(self, img):
        return cv2.Canny(img, self.canny_low, self.canny_high)

    def match(self, search_img, template):
        """在 search_img 中用边缘倒角距离匹配 template。

        Returns: (score, (x, y))
          score ∈ [-1, 1]，越高越好
        """
        if search_img.dtype != np.uint8:
            search_img = search_img.astype(np.uint8)
        if template.dtype != np.uint8:
            template = template.astype(np.uint8)

        H, W = search_img.shape
        h, w = template.shape
        if H < h or W < w:
            return -1.0, (0, 0)

        edges_s = self._edge_map(search_img)
        edges_t = self._edge_map(template)

        # 对搜索图边缘做距离变换
        dist = cv2.distanceTransform(
            (edges_s == 0).astype(np.uint8), cv2.DIST_L2, 5
        )

        # 积分图加速窗口求和
        integral = cv2.integral(dist)
        rows, cols = H - h + 1, W - w + 1
        i_idx = np.arange(rows)[:, None]
        j_idx = np.arange(cols)[None, :]

        sum_dist = (
            integral[i_idx + h, j_idx + w]
            - integral[i_idx + h, j_idx]
            - integral[i_idx, j_idx + w]
            + integral[i_idx, j_idx]
        )

        # 只在模板有边缘的位置计分
        t_edge_count = np.count_nonzero(edges_t)
        if t_edge_count < 10:
            return -1.0, (0, 0)

        # 归一化：平均倒角距离
        avg_chamfer = sum_dist / t_edge_count

        # 距离越小越好，映射到 [-1, 1]
        max_dist = max(w, h) * 0.5
        score = 1.0 - 2.0 * np.clip(avg_chamfer / max_dist, 0.0, 1.0)

        max_idx = np.unravel_index(np.argmax(score), score.shape)
        return float(score[max_idx]), (int(max_idx[1]), int(max_idx[0]))
