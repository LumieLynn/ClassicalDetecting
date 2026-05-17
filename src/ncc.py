import cv2
import numpy as np

# NCC 数值稳定性常量
_MIN_STD = 1e-5       # 低方差区域抑制阈值
_DENOM_EPS = 1e-8     # 防止除零


def fast_ncc_match(search_img, template):
    """手写 NCC（积分图 + filter2D），等价于 cv2.TM_CCOEFF_NORMED"""
    H, W = search_img.shape
    h, w = template.shape
    if H < h or W < w:
        return -1.0, (0, 0)

    img = search_img.astype(np.float64)
    tpl = template.astype(np.float64)
    N = h * w

    # 积分图：O(1) 窗口求和
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
    std_I = np.sqrt(var_I * N)          # sqrt( Σ(I - μ_I)² )

    # 模板统计量
    t_mean = tpl.mean()
    t_diff = tpl - t_mean
    t_std = np.sqrt(np.sum(t_diff * t_diff))
    if t_std < _MIN_STD:
        return -1.0, (0, 0)
    t_sum = tpl.sum()

    # 互相关（filter2D，锚点左上角）
    corr = cv2.filter2D(img, cv2.CV_64F, tpl, anchor=(0, 0))
    corr = corr[:rows, :cols]

    # 合成 NCC
    numerator = corr - mean_I * t_sum
    denominator = std_I * t_std
    ncc_map = np.full((rows, cols), -1.0, dtype=np.float64)
    valid = std_I > _MIN_STD
    ncc_map[valid] = numerator[valid] / (denominator[valid] + _DENOM_EPS)

    max_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    score = float(ncc_map[max_idx])
    y, x = max_idx
    return score, (x, y)
