import cv2
import numpy as np
import os
import time
from collections import deque

def fast_ncc_match(search_img, template):
    """手写 NCC：修复低方差区域导致的分数爆炸"""
    H, W = search_img.shape
    h, w = template.shape
    if H < h or W < w:
        return -1.0, (0, 0)

    img = search_img.astype(np.float64)
    tmpl = template.astype(np.float64)
    N = h * w

    # 1. 局部均值
    mean_kernel = np.ones((h, w), dtype=np.float64) / N
    mean_I = cv2.filter2D(img, cv2.CV_64F, mean_kernel, anchor=(0,0))
    mean_I = mean_I[: H - h + 1, : W - w + 1]

    # 2. 局部标准差（σ_I * sqrt(N)）
    sq = img ** 2
    mean_sq = cv2.filter2D(sq, cv2.CV_64F, mean_kernel, anchor=(0,0))
    mean_sq = mean_sq[: H - h + 1, : W - w + 1]
    var_I = mean_sq - mean_I * mean_I
    var_I = np.maximum(var_I, 0.0)
    std_I = np.sqrt(var_I * N)          # 即 sqrt(Σ (I - μ_I)^2)

    # 3. 模板统计量
    t_mean = tmpl.mean()
    t_diff = tmpl - t_mean
    t_sq_sum = np.sum(t_diff * t_diff)
    t_std = np.sqrt(t_sq_sum)           # 即 sqrt(Σ (T - μ_T)^2)
    if t_std < 1e-5:                    # 模板几乎是常数，放弃
        return -1.0, (0, 0)

    t_sum = tmpl.sum()

    # 4. 互相关（不做翻转，直接用模板当核）
    corr = cv2.filter2D(img, cv2.CV_64F, tmpl, anchor=(0,0))
    corr = corr[: H - h + 1, : W - w + 1]

    # 5. 合成 NCC，抑制低方差区域
    numerator = corr - mean_I * t_sum
    denominator = std_I * t_std

    min_std = 1e-5                       # 可调，单位是像素值
    valid = std_I > min_std
    ncc_map = np.full_like(numerator, -1.0, dtype=np.float64)
    ncc_map[valid] = numerator[valid] / (denominator[valid] + 1e-8)

    max_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    score = float(ncc_map[max_idx])
    y, x = max_idx
    return score, (x, y)

def main():
    video_path = 'data/task1/anime_video.mp4'
    template_paths = [
        'data/task1/small_template_1.png',
        'data/task1/small_template_2.png',
        'data/task1/small_template_3.png'
    ]

    # 基础文件检查
    if not os.path.exists(video_path):
        print(f"找不到视频文件: {video_path}")
        return

    base_templates = []
    for p in template_paths:
        if not os.path.exists(p):
            print(f"找不到模板图片: {p}")
            return
        tmpl = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        base_templates.append(tmpl)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件，请检查路径或编码格式。")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height}, {fps} fps")

    # 输出视频编码（macOS 兼容）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/task1/task1_result.mp4', fourcc, fps, (width, height))

    # 跟踪状态
    is_tracking = False
    current_scale = 1.0
    target_center = (0, 0)
    target_w, target_h = 0, 0
    trajectory = deque(maxlen=800)
    frame_count = 0
    frames_written = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\n视频读取结束，共处理 {frame_count} 帧。")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            best_score = -1.0
            best_match = None

            if not is_tracking:
                # ---------- 全局搜索 ----------
                if frame_count == 1:
                    print("第 1 帧：启动全局搜索（多尺度 + 多模板）...")
                    t_start = time.time()
                search_scales = np.arange(0.5, 2.1, 0.25)
                search_img = gray
                offset_x, offset_y = 0, 0
            else:
                # ---------- 局部跟踪：只在上一帧位置附近搜索 ----------
                search_scales = [current_scale * 0.9, current_scale, current_scale * 1.1]
                padding = int(50 * current_scale)
                cx, cy = target_center
                x1 = max(0, cx - target_w // 2 - padding)
                y1 = max(0, cy - target_h // 2 - padding)
                x2 = min(width, cx + target_w // 2 + padding)
                y2 = min(height, cy + target_h // 2 + padding)
                search_img = gray[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

            # 遍历所有尺度和模板
            for scale in search_scales:
                for idx, base_tmpl in enumerate(base_templates):
                    new_w = int(base_tmpl.shape[1] * scale)
                    new_h = int(base_tmpl.shape[0] * scale)
                    if new_w < 10 or new_h < 10:
                        continue
                    if new_w > search_img.shape[1] or new_h > search_img.shape[0]:
                        continue

                    scaled = cv2.resize(base_tmpl, (new_w, new_h))
                    score, (local_x, local_y) = fast_ncc_match(search_img, scaled)

                    # 只在第一帧打印进度，免得刷屏
                    if frame_count == 1 and not is_tracking:
                        print(f"  scale={scale:.2f}, template {idx+1}, current best={max(best_score, score):.3f}")

                    if score > best_score:
                        best_score = score
                        global_x = local_x + offset_x
                        global_y = local_y + offset_y
                        best_match = (global_x, global_y, new_w, new_h, scale, idx)

            # ---------- 判断是否跟踪到目标 ----------
            if best_score > 0.5:
                if not is_tracking:
                    elapsed = time.time() - t_start
                    print(f"锁定目标！帧序号 {frame_count}, 耗时 {elapsed:.1f} 秒, NCC={best_score:.3f}")
                    print("进入局部跟踪模式。")

                x, y, w, h, current_scale, tmpl_idx = best_match
                is_tracking = True
                target_w, target_h = w, h
                target_center = (x + w // 2, y + h // 2)
                trajectory.append(target_center)

                # 画框和文字
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"T{tmpl_idx+1} | scale={current_scale:.2f} | score={best_score:.2f}"
                cv2.putText(frame, text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                is_tracking = False
                cv2.putText(frame, "Lost - searching whole frame",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

            # 绘制运动轨迹
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False,
                              color=(0, 0, 255), thickness=2)
                cv2.circle(frame, target_center, 4, (255, 0, 0), -1)

            out.write(frame)
            frames_written += 1

            if is_tracking and frame_count % 50 == 0:
                print(f"  已跟踪 {frame_count} 帧...")

    except KeyboardInterrupt:
        print("\n用户中断，正在保存已处理的部分...")
    except Exception as e:
        print(f"\n运行出错: {e}")
    finally:
        cap.release()
        out.release()
        print(f"输出视频共 {frames_written} 帧，保存至 task1_result.mp4")


if __name__ == '__main__':
    main()