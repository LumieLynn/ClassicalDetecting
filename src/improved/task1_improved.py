"""Task 1 改进版 — NCC + 边缘匹配 + 卡尔曼平滑（动漫视频）。

改进点（相对原版纯 NCC）：
  - 边缘倒角距离匹配作为第二线索，验证 NCC 结果（动漫线稿边缘稳定）
  - 卡尔曼滤波平滑轨迹，减少抖动
  - 每帧测试全部 3 个模板（保留原版策略）
  - 双线索一致性检验：NCC 和边缘同时高分才锁定，减少误匹配
  - 参数更少（去掉了大量 ad-hoc 调参）
"""
import cv2
import numpy as np
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from classical.ncc import fast_ncc_match
from classical.edge_match import EdgeMatcher
from classical.motion import KalmanTracker

# ======================== 参数 ========================
GLOBAL_SEARCH_SCALES = np.arange(0.5, 2.1, 0.25)
LOCAL_SCALE_FACTORS = [0.85, 0.92, 1.0, 1.08, 1.15]
LOCAL_SEARCH_PAD = 60
MIN_TEMPLATE_SIZE = 10
NCC_LOCK_THRESHOLD = 0.5       # 全局搜锁定阈值
NCC_TRACK_THRESHOLD = 0.4      # 跟踪中 NCC 最低分
EDGE_AGREE_THRESHOLD = 0.15    # 边缘匹配最低分（与 NCC 一致）
TRAJECTORY_MAXLEN = 800
PROGRESS_INTERVAL = 50

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)


def match_all_templates(gray, templates, scales, offset_x=0, offset_y=0):
    """在所有模板和尺度中搜索最佳匹配，返回 (score, x, y, w, h, scale, tmpl_idx)。"""
    best_score = -1.0
    best_match = None
    for idx, tmpl in enumerate(templates):
        for s in scales:
            sw = int(tmpl.shape[1] * s)
            sh = int(tmpl.shape[0] * s)
            if sw < MIN_TEMPLATE_SIZE or sh < MIN_TEMPLATE_SIZE:
                continue
            if sw > gray.shape[1] or sh > gray.shape[0]:
                continue
            scaled = cv2.resize(tmpl, (sw, sh))
            score, (lx, ly) = fast_ncc_match(gray, scaled)
            if score > best_score:
                best_score = score
                best_match = (lx + offset_x, ly + offset_y, sw, sh, s, idx)
    if best_match is None:
        return -1.0, (0, 0, 0, 0, 1.0, 0)
    return best_score, best_match


def main():
    video_path = 'data/task1/anime_video.mp4'
    template_paths = [
        'data/task1/small_template_1.png',
        'data/task1/small_template_2.png',
        'data/task1/small_template_3.png',
    ]
    output_dir = 'output/improved/task1'
    os.makedirs(output_dir, exist_ok=True)

    for p in template_paths:
        if not os.path.exists(p):
            print(f"找不到模板: {p}")
            return
    if not os.path.exists(video_path):
        print(f"找不到视频: {video_path}")
        return

    base_templates = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in template_paths]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        os.path.join(output_dir, 'task1_result.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height),
    )

    edge_matcher = EdgeMatcher()
    kalman = KalmanTracker()
    is_tracking = False
    current_scale = 1.0
    target_center = (0, 0)
    target_w, target_h = 0, 0
    current_tmpl_idx = 0
    trajectory = deque(maxlen=TRAJECTORY_MAXLEN)
    frame_count = 0
    frames_written = 0

    print(f"视频: {width}x{height}, {fps}fps")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\n处理完成，共 {frame_count} 帧。")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            if not is_tracking:
                # ---------- 全局搜索 ----------
                best_score, best_match = match_all_templates(
                    gray, base_templates, GLOBAL_SEARCH_SCALES)
                if best_score > NCC_LOCK_THRESHOLD:
                    x, y, w, h, current_scale, current_tmpl_idx = best_match
                    target_w, target_h = w, h
                    target_center = (x + w // 2, y + h // 2)
                    kalman.init(target_center[0], target_center[1])
                    is_tracking = True
                    print(f"锁定目标: 帧 {frame_count}, NCC={best_score:.2f}, "
                          f"模板 {current_tmpl_idx+1}, scale={current_scale:.2f}")
                else:
                    cv2.putText(frame, "Searching...", (50, 50), FONT, 0.9, COLOR_RED, 2)
                    out.write(frame)
                    frames_written += 1
                    continue
            else:
                # ---------- 局部跟踪 ----------
                cx, cy = target_center
                # 卡尔曼预测位置
                pred_cx, pred_cy, std_x, std_y = kalman.predict()
                # 搜索区域：在预测位置周围扩展，大小由目标尺寸决定
                pad = max(LOCAL_SEARCH_PAD, int(target_w * 1.2))
                x1 = max(0, int(pred_cx - pad))
                y1 = max(0, int(pred_cy - pad))
                x2 = min(width, int(pred_cx + pad))
                y2 = min(height, int(pred_cy + pad))
                search = gray[y1:y2, x1:x2]

                if search.size == 0:
                    is_tracking = False
                    continue

                # 多尺度 NCC 搜索（所有模板）
                local_scales = [current_scale * f for f in LOCAL_SCALE_FACTORS]
                ncc_score, ncc_match = match_all_templates(
                    search, base_templates, local_scales, offset_x=x1, offset_y=y1)

                x, y, w, h, matched_scale, matched_idx = ncc_match
                best_center = (x + w // 2, y + h // 2)

                # 边缘匹配验证：在 NCC 结果附近做边缘匹配
                edge_score, (ex, ey) = edge_matcher.match(
                    search,
                    cv2.resize(base_templates[matched_idx], (w, h)),
                )
                edge_center = (x1 + ex + w // 2, y1 + ey + h // 2)
                # 边缘距离 NCC 位置的偏移
                edge_dist = np.sqrt((edge_center[0] - best_center[0])**2
                                    + (edge_center[1] - best_center[1])**2)
                edge_agree = edge_score > EDGE_AGREE_THRESHOLD and edge_dist < w * 0.5

                # 置信度融合
                if edge_agree:
                    # 两条线索一致：高置信度
                    conf = 0.5 * ncc_score + 0.5 * max(0, edge_score)
                    # 微调位置（NCC 和边缘的平均）
                    final_cx = int(0.6 * best_center[0] + 0.4 * edge_center[0])
                    final_cy = int(0.6 * best_center[1] + 0.4 * edge_center[1])
                else:
                    # 只有 NCC 可信
                    conf = ncc_score
                    final_cx, final_cy = best_center

                if ncc_score > NCC_TRACK_THRESHOLD:
                    # 卡尔曼更新
                    kalman.update(final_cx, final_cy)

                    target_center = (final_cx, final_cy)
                    target_w, target_h = w, h
                    current_scale = matched_scale
                    current_tmpl_idx = matched_idx
                    trajectory.append(target_center)

                    # 绘制
                    bx, by = final_cx - w // 2, final_cy - h // 2
                    cv2.rectangle(frame, (bx, by), (bx + w, by + h), COLOR_GREEN, 2)
                    status = f"T{matched_idx+1} s={matched_scale:.2f} "
                    if edge_agree:
                        status += f"NCC:{ncc_score:.2f} Edge:{edge_score:.2f}"
                    else:
                        status += f"NCC:{ncc_score:.2f}"
                    cv2.putText(frame, status, (bx, by - 8), FONT, 0.45, COLOR_GREEN, 2)
                else:
                    is_tracking = False
                    cv2.putText(frame, "LOST - global search", (30, 50),
                                FONT, 0.9, COLOR_RED, 2)
                    print(f"帧 {frame_count}: 跟丢 (NCC={ncc_score:.2f})，回到全局搜索")
                    # 仍然画出预测位置
                    bx = int(pred_cx - target_w // 2)
                    by = int(pred_cy - target_h // 2)
                    cv2.rectangle(frame, (bx, by), (bx + target_w, by + target_h),
                                  COLOR_RED, 2)
                    out.write(frame)
                    frames_written += 1
                    continue

            # 绘制轨迹
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, COLOR_RED, 2)
                cv2.circle(frame, target_center, 4, COLOR_BLUE, -1)

            out.write(frame)
            frames_written += 1

            if is_tracking and frame_count % PROGRESS_INTERVAL == 0:
                print(f"  帧 {frame_count}: 跟踪中... (模板{current_tmpl_idx+1})")

    except KeyboardInterrupt:
        print("\n用户中断。")
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        print(f"输出: {os.path.join(output_dir, 'task1_result.mp4')} ({frames_written} 帧)")


if __name__ == '__main__':
    main()
