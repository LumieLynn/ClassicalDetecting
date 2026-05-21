"""单应矩阵运动补偿 + 前景概率图。

用 SIFT + RANSAC 单应矩阵理解相机运动（旋转、缩放、透视），
在 warp 对齐空间中分离"场景结构（静止）"和"运动目标"。
"""

import cv2
import numpy as np

_HOMOGRAPHY_MIN_INLIERS = 10
_HOMOGRAPHY_RANSAC_THRESH = 3.0
_FG_DIFF_THRESH = 15
_FG_BLUR_SIZE = 25


class SceneMotionEstimator:
    """用单应矩阵对齐连续帧，生成前景概率图。"""

    def __init__(self):
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def align_and_diff(self, curr_gray):
        """对齐 prev_gray 到 curr_gray，返回 (foreground_prob_map, H, num_inliers)。

        foreground_prob_map: [0, 1]，运动目标区域高、静止场景低
        H: 3×3 单应矩阵（prev -> curr），失败时为 None
        num_inliers: RANSAC 内点数
        """
        if self.prev_gray is None:
            self._update_prev(curr_gray)
            return np.zeros_like(curr_gray, dtype=np.float32), None, 0

        H, num_inliers = self._estimate_homography(curr_gray)
        if H is not None:
            aligned = cv2.warpPerspective(
                self.prev_gray, H,
                (curr_gray.shape[1], curr_gray.shape[0]),
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            # 退化为相位相关平移
            try:
                shift, _ = cv2.phaseCorrelate(
                    np.float32(self.prev_gray), np.float32(curr_gray))
                dx, dy = shift
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(
                    self.prev_gray, M,
                    (curr_gray.shape[1], curr_gray.shape[0]),
                    borderMode=cv2.BORDER_REPLICATE,
                )
            except Exception:
                aligned = self.prev_gray

        # 帧差 -> 前景概率图
        diff = cv2.absdiff(curr_gray, aligned)
        _, mask = cv2.threshold(diff, _FG_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        prob = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                                (_FG_BLUR_SIZE, _FG_BLUR_SIZE), 0)

        self._update_prev(curr_gray)
        return prob, H, num_inliers

    def _estimate_homography(self, curr_gray):
        kp2, des2 = self.sift.detectAndCompute(curr_gray, None)
        if des2 is None or self.prev_des is None or len(kp2) < 4:
            return None, 0

        matches = self.matcher.knnMatch(self.prev_des, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < _HOMOGRAPHY_MIN_INLIERS:
            return None, len(good)

        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good]
                             ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]
                             ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                     _HOMOGRAPHY_RANSAC_THRESH)
        if H is None:
            return None, 0
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < _HOMOGRAPHY_MIN_INLIERS:
            return None, inliers
        return H, inliers

    def _update_prev(self, gray):
        self.prev_gray = gray.copy()
        self.prev_kp, self.prev_des = self.sift.detectAndCompute(gray, None)
