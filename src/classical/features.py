import cv2
import numpy as np


class FeatureMatcher:
    """SIFT / ORB 特征点匹配器，输出匹配置信度和目标边界框。"""

    def __init__(self, method="SIFT", ratio_thresh=0.75, min_inliers=4):
        self.ratio_thresh = ratio_thresh
        self.min_inliers = min_inliers

        if method.upper() == "SIFT":
            self.detector = cv2.SIFT_create()
            # FLANN for SIFT (float descriptors)
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.detector = cv2.ORB_create(nfeatures=500)
            # Brute-Force Hamming for ORB (binary descriptors)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.method = method.upper()

    def match(self, search_img, template):
        """在 search_img 中匹配 template。

        Returns: (score, (x, y, w, h))
          score  ∈ [0, 1]，越低表示越不可靠
          bbox   匹配到的边界框（在 search_img 坐标系中）
          失败时返回 (0.0, (0, 0, 0, 0))
        """
        h, w = template.shape
        if search_img.shape[0] < h or search_img.shape[1] < w:
            return 0.0, (0, 0, 0, 0)

        # 确保 uint8（SIFT/ORB 要求）
        if template.dtype != np.uint8:
            template = template.astype(np.uint8)
        if search_img.dtype != np.uint8:
            search_img = search_img.astype(np.uint8)

        kp1, des1 = self.detector.detectAndCompute(template, None)
        kp2, des2 = self.detector.detectAndCompute(search_img, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return 0.0, (0, 0, 0, 0)

        if self.method == "SIFT":
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]
        else:
            raw = self.matcher.knnMatch(des1, des2, k=2)
            good = []
            for pair in raw:
                if len(pair) >= 2:
                    m, n = pair[0], pair[1]
                    if m.distance < self.ratio_thresh * n.distance:
                        good.append(m)

        if len(good) < self.min_inliers:
            return 0.0, (0, 0, 0, 0)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return 0.0, (0, 0, 0, 0)

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_inliers:
            return 0.0, (0, 0, 0, 0)

        # 将模板四角映射到搜索图像，取包围盒
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)
        x1 = max(0, int(projected[:, 0, 0].min()))
        y1 = max(0, int(projected[:, 0, 1].min()))
        x2 = min(search_img.shape[1], int(projected[:, 0, 0].max()))
        y2 = min(search_img.shape[0], int(projected[:, 0, 1].max()))
        bw, bh = x2 - x1, y2 - y1

        if bw < 4 or bh < 4:
            return 0.0, (0, 0, 0, 0)

        # 置信度：inlier 比例 × 匹配数归一化
        inlier_ratio = inliers / len(good)
        match_quality = min(1.0, len(good) / 30.0)
        score = 0.5 * inlier_ratio + 0.5 * match_quality

        return score, (x1, y1, bw, bh)
