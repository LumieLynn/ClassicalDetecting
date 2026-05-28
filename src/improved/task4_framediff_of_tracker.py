"""Task 4 — 帧差法 + 最大连通域面积 + 光流辅助 无人机追踪。

核心假设：
  - 树叶在动但离散 → 帧差连通域面积小
  - 无人机集中运动 → 帧差连通域面积大
  - 取最大连通域 → 即无人机

方法：
  1. 连续帧灰度差 → 二值化 + 形态学 → 最大连通域 → 无人机位置
  2. 检测成功时：在目标周围布 LK 光流网格点
  3. 检测丢失时：用光流点中值位移更新位置
     - 无人机急停 → 光流位移 ≈ 0 → 位置稳住，不冲过头
     - 树叶遮挡 → 光流点逐渐丢失 → 保持最后位置，等帧差恢复

输出路径: output/improved/task4/
"""

import os
import sys
from collections import deque

import cv2
import numpy as np

# ======================== 帧差参数 ========================
DIFF_THRESH = 18                # 灰度差阈值
DIFF_FRAME_GAP = 2              # 帧间隔（>1 放大位移信号）
DIFF_ACCUM = 3                  # 累积帧数（多帧 diff 叠加）

# ======================== 形态学 ========================
MORPH_OPEN_SIZE = 3             # 开运算核大小（去噪点）
MORPH_CLOSE_SIZE = 7            # 闭运算核大小（合并碎片）

# ======================== 连通域筛选 ========================
AREA_MIN = 80                   # 最小面积（太小 = 噪声）
AREA_MAX = 800                  # 最大面积（太大 = 树叶群）
COMPACTNESS_MIN = 0.25          # 最小紧致度 4π·area/perimeter²（排除长条树枝）

# ======================== 光流辅助 ========================
OF_GRID_SPACING = 6             # 网格间距
OF_GRID_HALF = 28               # 网格半宽（目标周围 56x56 区域）
OF_WIN = (15, 15)
OF_MAX_LEVEL = 3
OF_MIN_VALID = 4                # 最少有效光流点数
OF_REFILL_INTERVAL = 8          # 每 N 次检测重新布点

# ======================== 跟踪 ========================
LOST_MAX = 90                   # 光流也连续失败的上限

# ======================== 通用 ========================
TRAJ_MAXLEN = 4000
PROG_INTERVAL = 50

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_BLUE = (255, 0, 0)
C_YELLOW = (0, 255, 255)
C_CYAN = (255, 255, 0)


def place_of_grid(cx, cy, half, spacing, h, w):
    """在目标周围放置均匀网格点。返回 (N, 1, 2) float32。"""
    pts = []
    for gy in range(int(cy) - half, int(cy) + half + 1, spacing):
        for gx in range(int(cx) - half, int(cx) + half + 1, spacing):
            if 0 <= gx < w and 0 <= gy < h:
                pts.append([float(gx), float(gy)])
    if len(pts) < OF_MIN_VALID:
        return None
    return np.float32(pts).reshape(-1, 1, 2)


def detect_motion_blob(diff_gray):
    """从灰度差图中检测最大运动连通域。

    Returns: (cx, cy, area, compactness) 或 None
    """
    _, binary = cv2.threshold(diff_gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    if n_labels <= 1:
        return None

    best = None
    best_area = AREA_MIN - 1
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < AREA_MIN or area > AREA_MAX:
            continue
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            compactness = (4 * np.pi * area / (perimeter * perimeter)
                           if perimeter > 0 else 0)
        else:
            compactness = 0
        if compactness < COMPACTNESS_MIN:
            continue
        if area > best_area:
            best_area = area
            cx = float(centroids[i, 0])
            cy = float(centroids[i, 1])
            best = (cx, cy, area, compactness)

    return best


def main():
    np.random.seed(0)
    cv2.setRNGSeed(0)

    video_path = "data/task4/地面光学站跟踪无人机.avi"
    output_dir = "output/improved/task4"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {width}x{height}, {fps:.2f}fps, "
          f"共 {total_frames} 帧 ({total_frames/fps:.1f}s)")

    out = cv2.VideoWriter(
        os.path.join(output_dir, "tracked_result.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height),
    )

    trajectory = deque(maxlen=TRAJ_MAXLEN)
    gray_buffer = deque(maxlen=max(DIFF_FRAME_GAP, DIFF_ACCUM) + 2)

    # 光流
    lk_params = dict(winSize=OF_WIN, maxLevel=OF_MAX_LEVEL,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               30, 0.01))
    of_pts = None
    prev_gray_of = None
    of_refill_count = 0

    is_tracking = False
    lost_count = 0
    target_center = (0, 0)

    frame_idx = 0
    frames_written = 0

    print("开始帧差 + 光流辅助跟踪...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_buffer.append(gray)

        # ---- 累积帧差 ----
        if len(gray_buffer) >= DIFF_FRAME_GAP + 1:
            prev_gray = gray_buffer[-1 - DIFF_FRAME_GAP]
            diff = cv2.absdiff(gray, prev_gray)

            if len(gray_buffer) >= DIFF_FRAME_GAP + DIFF_ACCUM:
                accum = diff.astype(np.float32)
                for k in range(1, min(DIFF_ACCUM,
                                      len(gray_buffer) - DIFF_FRAME_GAP - 1)):
                    gp = gray_buffer[-1 - DIFF_FRAME_GAP - k]
                    d = cv2.absdiff(gray, gp)
                    accum += d.astype(np.float32)
                diff = np.clip(accum / DIFF_ACCUM, 0, 255).astype(np.uint8)
        else:
            diff = None
            if len(gray_buffer) >= 2:
                diff = cv2.absdiff(gray, gray_buffer[-2])

        # ---- 帧差检测 ----
        blob = None
        if diff is not None:
            blob = detect_motion_blob(diff)

        # ---- 光流跟踪 ----
        of_dx, of_dy = 0.0, 0.0
        of_valid = 0
        if prev_gray_of is not None and of_pts is not None and len(of_pts) >= OF_MIN_VALID:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray_of, gray, of_pts, None, **lk_params)
            if next_pts is not None and status is not None:
                status = status.flatten()
                vp = of_pts[status == 1]
                vn = next_pts[status == 1]
                if len(vp) >= OF_MIN_VALID:
                    disp = vn - vp
                    of_dx = float(np.median(disp[:, 0, 0]))
                    of_dy = float(np.median(disp[:, 0, 1]))
                    of_valid = len(vp)
                    of_pts = vn.reshape(-1, 1, 2).astype(np.float32)
                else:
                    of_pts = None

        # ---- 状态机 ----
        if blob is not None:
            cx, cy, area, compactness = blob

            if not is_tracking:
                print(f"帧 {frame_idx}: 检测到无人机! "
                      f"pos=({cx:.0f},{cy:.0f}) area={area}")

            is_tracking = True
            lost_count = 0
            target_center = (int(cx), int(cy))
            trajectory.append(target_center)

            # 刷新光流网格点
            of_refill_count += 1
            if of_refill_count >= OF_REFILL_INTERVAL or of_pts is None:
                new_pts = place_of_grid(cx, cy, OF_GRID_HALF, OF_GRID_SPACING,
                                        height, width)
                if new_pts is not None:
                    of_pts = new_pts
                of_refill_count = 0
            prev_gray_of = gray.copy()

        elif is_tracking and of_pts is not None and of_valid >= OF_MIN_VALID:
            # 帧差丢失，光流顶上
            lost_count = 0
            new_cx = target_center[0] + of_dx
            new_cy = target_center[1] + of_dy
            new_cx = np.clip(new_cx, 0, width - 1)
            new_cy = np.clip(new_cy, 0, height - 1)
            target_center = (int(new_cx), int(new_cy))
            trajectory.append(target_center)
            prev_gray_of = gray.copy()

        elif is_tracking:
            # 帧差 + 光流都失败 → 保持位置
            lost_count += 1
            if lost_count > LOST_MAX:
                print(f"帧 {frame_idx}: 丢失 ({lost_count}帧)")
                is_tracking = False
                lost_count = 0
                of_pts = None
            else:
                trajectory.append(target_center)

        # ---- 可视化 ----
        if is_tracking:
            cv2.circle(frame, target_center, 10, C_GREEN, 2)
            cv2.circle(frame, target_center, 5, C_RED, -1)

            if blob is not None:
                label = f"DIFF area={blob[2]:.0f}"
            elif of_valid >= OF_MIN_VALID:
                label = f"OF n={of_valid} d=({of_dx:.1f},{of_dy:.1f})"
            else:
                label = "HOLD"
            cv2.putText(frame, label,
                        (target_center[0] + 15, target_center[1] - 10),
                        FONT, 0.5, C_GREEN, 2)

            if of_pts is not None:
                for pt in of_pts:
                    px, py = int(pt[0, 0]), int(pt[0, 1])
                    cv2.circle(frame, (px, py), 1, C_CYAN, -1)

        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, C_RED, 2)

        if blob is not None:
            mode_str = f"DIFF({blob[2]:.0f})"
        elif of_valid >= OF_MIN_VALID:
            mode_str = f"OF({of_valid})"
        else:
            mode_str = "HOLD"
        cv2.putText(frame,
                    f"F:{frame_idx}/{total_frames} | {mode_str} | "
                    f"lost={lost_count}",
                    (10, 30), FONT, 0.6,
                    C_GREEN if is_tracking else C_YELLOW, 2)

        out.write(frame)
        frames_written += 1
        frame_idx += 1

        if frame_idx % PROG_INTERVAL == 0:
            print(f"  帧 {frame_idx}/{total_frames}: "
                  f"pos=({target_center[0]},{target_center[1]}) "
                  f"mode={mode_str}")

    cap.release()
    out.release()
    print(f"完成。输出: {os.path.join(output_dir, 'tracked_result.mp4')}")


if __name__ == "__main__":
    main()
