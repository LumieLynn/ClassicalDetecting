import cv2
import numpy as np
import os
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque
import time

def opencv_ncc_match(search_img, template):
    if search_img.shape[0] < template.shape[0] or search_img.shape[1] < template.shape[1]:
        return -1.0, (0, 0)
    result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc

def main():
    video_path = 'data/task1/anime_video.mp4'
    template_paths = [
        'data/task1/small_template_1.png', 
        'data/task1/small_template_2.png', 
        'data/task1/small_template_3.png']
    
    # --- 基础检查 ---
    if not os.path.exists(video_path):
        print(f"File Not Found '{video_path}'")
        return

    base_templates = []
    for p in template_paths:
        if not os.path.exists(p):
            print(f"No Template Found '{p}'")
            return
        tmpl = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        base_templates.append(tmpl)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can't decode via OpenCV.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video read. Resolution: {width}x{height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('task1_result.mp4', fourcc, fps, (width, height))

    # 状态变量
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
                print(f"Finished. Processed {frame_count} frames")
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            best_score = -1.0
            best_match = None

            if not is_tracking:
                if frame_count == 1:
                    print(f"Frame {frame_count}: 全图盲搜")
                    start_time = time.time()
                search_scales = np.arange(0.5, 2.1, 0.25) 
                search_img = frame_gray
                offset_x, offset_y = 0, 0
            else:
                search_scales = [current_scale * 0.9, current_scale, current_scale * 1.1]
                padding = int(50 * current_scale)
                cx, cy = target_center
                x1 = max(0, cx - target_w//2 - padding)
                y1 = max(0, cy - target_h//2 - padding)
                x2 = min(width, cx + target_w//2 + padding)
                y2 = min(height, cy + target_h//2 + padding)
                search_img = frame_gray[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

            for scale in search_scales:
                for tmpl_idx, base_tmpl in enumerate(base_templates):
                    new_w = int(base_tmpl.shape[1] * scale)
                    new_h = int(base_tmpl.shape[0] * scale)
                    if new_w < 10 or new_h < 10 or new_w > search_img.shape[1] or new_h > search_img.shape[0]:
                        continue
                    
                    scaled_tmpl = cv2.resize(base_tmpl, (new_w, new_h))
                    score, (local_x, local_y) = opencv_ncc_match(search_img, scaled_tmpl)
                    
                    # 进度汇报
                    if frame_count == 1:
                        print(f"   [Scanning] scale: {scale:.2f}, template: {tmpl_idx+1}, highest_score: {max(best_score, score):.2f}")

                    if score > best_score:
                        best_score = score
                        global_x = local_x + offset_x
                        global_y = local_y + offset_y
                        best_match = (global_x, global_y, new_w, new_h, scale, tmpl_idx)

            if best_score > 0.5:
                if not is_tracking:
                    cost_time = time.time() - start_time
                    print(f"Target locked in {frame_count}, cost {cost_time:.1f}, score: {best_score:.2f}")
                    print(f"searching and targeting...")
                
                x, y, w, h, current_scale, tmpl_idx = best_match
                is_tracking = True
                target_w, target_h = w, h
                target_center = (x + w//2, y + h//2)
                trajectory.append(target_center)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                info = f"Tmpl:{tmpl_idx+1} Scale:{current_scale:.2f} Score:{best_score:.2f}"
                cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                is_tracking = False
                cv2.putText(frame, "Target Lost! Global Searching...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                cv2.circle(frame, target_center, 4, (255, 0, 0), -1)

            out.write(frame)
            frames_written += 1

            # 每处理 50 帧在终端报个平安
            if is_tracking and frame_count % 50 == 0:
                print(f"written {frame_count} frames...")

    except KeyboardInterrupt:
        print("\n exiting...")
    except Exception as e:
        print(f"\n Crashed: {e}")
    finally:
        cap.release()
        out.release()
        print(f"Finished. Written {frames_written} frames to task1_result.mp4.")

if __name__ == '__main__':
    main()