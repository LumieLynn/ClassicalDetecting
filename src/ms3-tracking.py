import cv2
import numpy as np
import os
import time
from collections import deque

from ncc import fast_ncc_match

# ======================== CLAHE 参数 ========================
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ======================== 相位相关参数 ========================
PHASE_MAX_SHIFT = 50             # 位移超过此值视为误检

# ======================== 模板提取参数 ========================
TM_HSV_LOWER = np.array([15, 40, 40])
TM_HSV_UPPER = np.array([45, 255, 255])
TM_BORDER_MASK_RATIO = 0.1
TM_MORPH_KERNEL_SIZE = (3, 3)
TM_MORPH_ITERATIONS = 1
TM_MIN_CONTOUR_AREA = 20
TM_INSCRIBED_SCALE = 0.85       # 内接正方形缩小系数

# ======================== 锚点扫描参数 ========================
ANCHOR_SCAN_START_SEC = 9
ANCHOR_SCAN_END_SEC = 13
ANCHOR_MIN_SCORE = 0.4
ANCHOR_SCAN_SCALES = [0.6, 0.7, 0.8, 0.9, 1.0]
ANCHOR_MIN_TEMPLATE_SIZE = 10

# ======================== 反向追踪参数 ========================
REV_VELOCITY_DAMPING = 0.9       # 预测位置阻尼
REV_CAM_DAMPING = 0.5            # 相机位移补偿阻尼
REV_PAD_RATIO = 0.25             # 搜索 pad 比例
REV_PAD_MIN = 12
REV_PAD_MOTION_RATIO = 0.5       # 运动幅度 → pad 转换系数
REV_SCALE_FACTORS = [0.95, 0.98, 1.0]
REV_MIN_TEMPLATE_SIZE = 6
REV_RESEARCH_MIN_SCORE = 0.4     # 低于此分触发重搜
REV_RESEARCH_PAD_RATIO = 0.4
REV_RESEARCH_PAD_MIN = 15
REV_RESEARCH_SCALE_FACTORS = [0.92, 0.97, 1.0]
REV_RESEARCH_SCORE_MARGIN = 0.03 # 重搜结果必须比原最高分高出至少此值
REV_RESEARCH_MIN_ACCEPT = 0.45   # 重搜结果自身最低接受分
REV_CON_LOST_MAX = 10
REV_MIN_SCORE = 0.4
REV_VELOCITY_EMA_ALPHA = 0.7     # 速度指数平滑系数
REV_SCALE_CLAMP_MIN_RATIO = 0.95 # 尺度下界相对于 prev_scale
REV_SCALE_CLAMP_MAX = 1.0        # 尺度上界（严禁放大）

# ======================== 正向追踪参数 ========================
FWD_VELOCITY_DAMPING = 0.9
FWD_CAM_DAMPING = 0.4            # 位移补偿阻尼
FWD_SCALE_DAMPING = 0.9
FWD_PAD_RATIO = 0.5
FWD_PAD_MIN = 10
FWD_PAD_MOTION_RATIO = 0.5
FWD_SCALE_FACTORS = [0.9735, 0.989, 1.0, 1.037, 1.067]
FWD_MIN_TEMPLATE_SIZE = 6
FWD_VELOCITY_EMA_ALPHA = 0.4    # 惯性与新观测的比例
FWD_SCALE_EMA_ALPHA = 0.4

# ======================== 正向追踪 – 低分重搜参数 ========================
FWD_RESEARCH_MIN_SCORE = 0.35
FWD_RESEARCH_PAD_RATIO = 0.2
FWD_RESEARCH_PAD_MIN = 25
FWD_RESEARCH_SCALE_FACTORS = [0.93, 0.97, 1.0, 1.03, 1.04]
FWD_RESEARCH_SCORE_MARGIN = 0.02
FWD_RESEARCH_MIN_ACCEPT = 0.4

# ======================== 模板更新参数 ========================
TEMPLATE_UPDATE_INTERVAL = 10
TEMPLATE_UPDATE_MIN_SCORE = 0.5
TEMPLATE_BLEND_MIN_SCORE = 0.6
TEMPLATE_BLEND_ALPHA_RETAIN = 0.95   # 保留旧模板的比例
TEMPLATE_UPDATE_MIN_PATCH = 6

# ======================== 丢失恢复参数 ========================
CONSECUTIVE_LOST_MAX = 30

# ======================== 通用参数 ========================
TRAJECTORY_MAXLEN = 2000
PROGRESS_INTERVAL = 50

# ======================== 可视化参数 ========================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_INFO = 0.5
FONT_SCALE_ANCHOR = 0.7
FONT_SCALE_LOST = 1.5
FONT_THICKNESS = 2
FONT_THICKNESS_LOST = 3
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
BOX_THICKNESS = 2
TRAJECTORY_THICKNESS = 2
CIRCLE_RADIUS = 4


# ====================== 全局位移估计（相位相关） ======================
def estimate_global_shift(prev_gray, curr_gray):
    """返回 (dx, dy)，若失败则返回 (0, 0)"""
    try:
        shift, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(curr_gray))
        dx, dy = shift
        if abs(dx) > PHASE_MAX_SHIFT or abs(dy) > PHASE_MAX_SHIFT:
            return 0.0, 0.0
        return dx, dy
    except Exception:
        return 0.0, 0.0


# ====================== 模板自动提取（黄色圆圈，内缩防背景） ======================
def extract_template_from_ref(ref_path):
    print(f"从参考图提取模板: {ref_path}")
    img = cv2.imread(ref_path)
    if img is None:
        raise FileNotFoundError(f"无法读取参考图: {ref_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 黄色
    mask = cv2.inRange(hsv, TM_HSV_LOWER, TM_HSV_UPPER)

    # 屏蔽上下边缘
    mask[:int(h * TM_BORDER_MASK_RATIO), :] = 0
    mask[int(h * (1.0 - TM_BORDER_MASK_RATIO)):, :] = 0

    kernel = np.ones(TM_MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=TM_MORPH_ITERATIONS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imwrite("output/task3/debug_mask.png", mask)
        raise ValueError("未找到黄色区域，已保存 debug_mask.png")

    # 取面积最大的轮廓（黄圈）
    best_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best_cnt) < TM_MIN_CONTOUR_AREA:
        raise ValueError("黄圈面积太小")

    # 最小外接圆，取内接正方形，并适当缩小，确保只含骑车人
    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    side = int(radius * TM_INSCRIBED_SCALE)
    side = min(side, w, h)
    x1 = max(0, int(cx - side // 2))
    y1 = max(0, int(cy - side // 2))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    patch = img[y1:y2, x1:x2]
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("output/task3/auto_template.png", gray_patch)
    print(f"模板提取成功，尺寸 {gray_patch.shape[1]}x{gray_patch.shape[0]}")
    return gray_patch


# ====================== 主程序 ======================
def main():
    video_path = "data/task3/大疆无人机航拍骑车人.mp4"
    ref_image_path = "data/task3/大疆无人机航拍骑车人目标.png"
    output_dir = "output/task3"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 {video_path}")
        return
    if not os.path.exists(ref_image_path):
        print(f"错误: 参考图不存在 {ref_image_path}")
        return

    # 提取模板
    try:
        base_template = extract_template_from_ref(ref_image_path)
    except Exception as e:
        print(f"模板提取失败: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}, {fps}fps, 共约 {total_frames} 帧")

    # 锚点搜索时间窗口
    start_scan_frame = int(fps * ANCHOR_SCAN_START_SEC)
    max_scan = min(int(fps * ANCHOR_SCAN_END_SEC), total_frames)

    # ---------- 阶段一：在限定时间范围内搜索锚点（逐帧，同时计算全局位移） ----------
    print(f"开始扫描锚点帧（{ANCHOR_SCAN_START_SEC}～{ANCHOR_SCAN_END_SEC} 秒，逐帧）...")
    anchor_idx = -1
    anchor_box = None
    best_anchor_score = ANCHOR_MIN_SCORE
    scan_frames = []
    frame_shifts = []          # 每帧相对上一帧的 (dx, dy)，第一帧为 (0,0)
    prev_gray_scan = None

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    base_template_clahe = clahe.apply(base_template)

    frame_idx = 0
    while frame_idx < max_scan:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算全局位移
        if prev_gray_scan is not None:
            dx, dy = estimate_global_shift(prev_gray_scan, gray)
            frame_shifts.append((dx, dy))
        else:
            frame_shifts.append((0.0, 0.0))
        prev_gray_scan = gray

        scan_frames.append(frame)

        # 逐帧搜索锚点
        if frame_idx >= start_scan_frame:
            enhanced = clahe.apply(gray)
            # 多尺度搜索，专门应对开头横拉时目标变小
            best_local_score = -1.0
            best_local_match = None
            for s in ANCHOR_SCAN_SCALES:
                sw = int(base_template.shape[1] * s)
                sh = int(base_template.shape[0] * s)
                if sw < ANCHOR_MIN_TEMPLATE_SIZE or sh < ANCHOR_MIN_TEMPLATE_SIZE \
                   or sw > enhanced.shape[1] or sh > enhanced.shape[0]:
                    continue
                scaled_tmpl = cv2.resize(base_template_clahe, (sw, sh))
                score, (lx, ly) = fast_ncc_match(enhanced, scaled_tmpl)
                if score > best_local_score:
                    best_local_score = score
                    best_local_match = (lx, ly, sw, sh, s)
            if best_local_match is not None:
                lx, ly, sw, sh, s = best_local_match
                score = best_local_score
            else:
                continue   # 无合法尺度，跳过本帧
            if score > best_anchor_score:
                best_anchor_score = score
                anchor_idx = frame_idx
                anchor_box = (lx, ly, base_template.shape[1], base_template.shape[0])
                print(f"  发现更优锚点: 帧 {anchor_idx} ({anchor_idx/fps:.1f}s), 得分 {score:.3f}")
        frame_idx += 1

    if anchor_idx == -1:
        print("错误: 在指定时间范围内未找到目标，请检查视频或参考图。")
        cap.release()
        return

    print(f"锚点定位完成: 帧 {anchor_idx} ({anchor_idx/fps:.1f}s), 得分 {best_anchor_score:.3f}")

    # 重置视频
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(
        os.path.join(output_dir, "tracked_result.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    # ---------- 阶段二：反向追踪（保守版，杜绝路边诱骗） ----------
    print("开始反向追踪...")
    base_template_clahe_back = clahe.apply(base_template)
    curr_tmpl_back = base_template_clahe_back.astype(np.float32)
    curr_scale = 1.0
    prev_scale = 1.0
    history = {}
    history[anchor_idx] = anchor_box
    cx = anchor_box[0] + anchor_box[2] // 2
    cy = anchor_box[1] + anchor_box[3] // 2
    vx, vy = 0.0, 0.0
    consecutive_lost = 0

    for i in range(anchor_idx - 1, -1, -1):
        frame = scan_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)

        # 实时位移：从后一帧 (i+1) 到当前帧 (i)
        if i + 1 < len(scan_frames):
            prev_gray = cv2.cvtColor(scan_frames[i+1], cv2.COLOR_BGR2GRAY)
            dx_back, dy_back = estimate_global_shift(prev_gray, gray)
        else:
            dx_back, dy_back = 0.0, 0.0

        # 预测位置（位移补偿权重降至0.5，减少误检影响）
        pred_cx = cx + vx * REV_VELOCITY_DAMPING + dx_back * REV_CAM_DAMPING
        pred_cy = cy + vy * REV_VELOCITY_DAMPING + dy_back * REV_CAM_DAMPING

        curr_w = int(base_template.shape[1] * curr_scale)
        curr_h = int(base_template.shape[0] * curr_scale)

        motion_mag = np.sqrt(dx_back**2 + dy_back**2)
        extra_pad = int(motion_mag * REV_PAD_MOTION_RATIO)
        pad = max(REV_PAD_MIN, int(curr_w * REV_PAD_RATIO)) + extra_pad

        x1 = max(0, int(pred_cx - curr_w // 2 - pad))
        y1 = max(0, int(pred_cy - curr_h // 2 - pad))
        x2 = min(width, int(pred_cx + curr_w // 2 + pad))
        y2 = min(height, int(pred_cy + curr_h // 2 + pad))
        search_gray = enhanced[y1:y2, x1:x2]
        if search_gray.size == 0:
            cx, cy = pred_cx, pred_cy
            history[i] = history.get(i + 1, anchor_box)
            continue
        search = search_gray

        # 尺度候选：只缩不放
        scale_candidates = [curr_scale * f for f in REV_SCALE_FACTORS]
        best_score = -1.0
        best_match = None
        for s in scale_candidates:
            sw = int(base_template.shape[1] * s)
            sh = int(base_template.shape[0] * s)
            if sw < REV_MIN_TEMPLATE_SIZE or sh < REV_MIN_TEMPLATE_SIZE \
               or sw > search.shape[1] or sh > search.shape[0]:
                continue
            scaled_tmpl = cv2.resize(curr_tmpl_back, (sw, sh))
            score, (lx, ly) = fast_ncc_match(search, scaled_tmpl)
            if score > best_score:
                best_score = score
                best_match = (lx, ly, sw, sh, s)

        # 低分重搜
        if best_score < REV_RESEARCH_MIN_SCORE:
            pad2 = max(REV_RESEARCH_PAD_MIN, int(curr_w * REV_RESEARCH_PAD_RATIO))
            x1b = max(0, int(pred_cx - curr_w // 2 - pad2))
            y1b = max(0, int(pred_cy - curr_h // 2 - pad2))
            x2b = min(width, int(pred_cx + curr_w // 2 + pad2))
            y2b = min(height, int(pred_cy + curr_h // 2 + pad2))
            search_gray2 = enhanced[y1b:y2b, x1b:x2b]
            if search_gray2.size > 0:
                search2 = search_gray2
                scale_candidates2 = [curr_scale * f for f in REV_RESEARCH_SCALE_FACTORS]
                best_score2 = -1.0
                best_match2 = None
                for s in scale_candidates2:
                    sw = int(base_template.shape[1] * s)
                    sh = int(base_template.shape[0] * s)
                    if sw < REV_MIN_TEMPLATE_SIZE or sh < REV_MIN_TEMPLATE_SIZE \
                       or sw > search2.shape[1] or sh > search2.shape[0]:
                        continue
                    scaled_tmpl = cv2.resize(curr_tmpl_back, (sw, sh))
                    score, (lx, ly) = fast_ncc_match(search2, scaled_tmpl)
                    if score > best_score2:
                        best_score2 = score
                        best_match2 = (lx, ly, sw, sh, s, x1b, y1b)

                # 关键保护：重搜结果必须比原最高分高出至少 margin，且自身得分要可靠
                if best_score2 > best_score + REV_RESEARCH_SCORE_MARGIN \
                   and best_score2 > REV_RESEARCH_MIN_ACCEPT:
                    lx, ly, sw, sh, s, offx, offy = best_match2
                    bx, by = offx + lx, offy + ly
                    best_score = best_score2
                    best_match = (lx, ly, sw, sh, s)
                else:
                    # 放弃重搜，直接用预测位置
                    consecutive_lost += 1
                    cx, cy = pred_cx, pred_cy
                    if consecutive_lost > REV_CON_LOST_MAX:
                        history[i] = history.get(i + 1, anchor_box)
                    else:
                        history[i] = (int(cx - curr_w/2), int(cy - curr_h/2), curr_w, curr_h)
                    continue
            else:
                consecutive_lost += 1
                cx, cy = pred_cx, pred_cy
                history[i] = history.get(i + 1, anchor_box)
                continue

        if best_score > REV_MIN_SCORE:
            consecutive_lost = 0
            lx, ly, sw, sh, s = best_match
            bx, by = x1 + lx, y1 + ly
            new_cx = bx + sw // 2
            new_cy = by + sh // 2

            raw_vx = new_cx - cx
            raw_vy = new_cy - cy
            # 降低速度更新惯性，不让错误帧带偏未来
            vx = REV_VELOCITY_EMA_ALPHA * vx + (1.0 - REV_VELOCITY_EMA_ALPHA) * raw_vx
            vy = REV_VELOCITY_EMA_ALPHA * vy + (1.0 - REV_VELOCITY_EMA_ALPHA) * raw_vy

            s = np.clip(s, prev_scale * REV_SCALE_CLAMP_MIN_RATIO, REV_SCALE_CLAMP_MAX)
            cx, cy = new_cx, new_cy
            curr_scale = s
            prev_scale = s
            history[i] = (bx, by, sw, sh)

            # 反向完全禁用模板更新，保持最干净的初始模板
            # 因为反向时段内目标外观变化极小，初始模板最可靠
        else:
            cx, cy = pred_cx, pred_cy
            history[i] = history.get(i + 1, anchor_box)

    print("反向追踪完成。")

    # ---------- 阶段三：正向追踪（实时位移补偿 + 动态 pad） ----------
    print("开始正向追踪...")
    trajectory = deque(maxlen=TRAJECTORY_MAXLEN)
    curr_tmpl = base_template.astype(np.float32)
    curr_scale = 1.0
    base_w, base_h = base_template.shape[1], base_template.shape[0]

    cx = anchor_box[0] + anchor_box[2] // 2
    cy = anchor_box[1] + anchor_box[3] // 2
    vx, vy = 0.0, 0.0
    vs = 0.0
    last_update_frame = anchor_idx
    consecutive_lost = 0
    prev_gray_forward = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 反向帧绘制
        if frame_idx < anchor_idx and frame_idx in history:
            bx, by, bw, bh = history[frame_idx]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_GREEN, BOX_THICKNESS)
            center = (bx + bw // 2, by + bh // 2)
            trajectory.append(center)
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, COLOR_RED, TRAJECTORY_THICKNESS)
                cv2.circle(frame, center, CIRCLE_RADIUS, COLOR_BLUE, -1)
            out.write(frame)
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 锚点帧绘制
        if frame_idx == anchor_idx:
            bx, by, bw, bh = anchor_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_RED, BOX_THICKNESS)
            cv2.putText(frame, "ANCHOR", (bx, by - 10),
                        FONT, FONT_SCALE_ANCHOR, COLOR_RED, FONT_THICKNESS)
            trajectory.append((cx, cy))
            out.write(frame)
            frame_idx += 1
            prev_gray_forward = gray
            continue

        # 实时位移估计
        if prev_gray_forward is not None:
            dx_forward, dy_forward = estimate_global_shift(prev_gray_forward, gray)
        else:
            dx_forward, dy_forward = 0.0, 0.0
        prev_gray_forward = gray

        # 预测位置（自身运动 + 背景补偿）
        pred_cx = cx + vx * FWD_VELOCITY_DAMPING - dx_forward * FWD_CAM_DAMPING
        pred_cy = cy + vy * FWD_VELOCITY_DAMPING - dy_forward * FWD_CAM_DAMPING
        pred_scale = curr_scale + vs * FWD_SCALE_DAMPING

        curr_w = int(base_w * pred_scale)
        curr_h = int(base_h * pred_scale)

        # 动态 pad：基础 + 位移补偿
        motion_mag = np.sqrt(dx_forward**2 + dy_forward**2)
        extra_pad = int(motion_mag * FWD_PAD_MOTION_RATIO)
        pad = max(FWD_PAD_MIN, int(curr_w * FWD_PAD_RATIO)) + extra_pad

        x1 = max(0, int(pred_cx - curr_w // 2 - pad))
        y1 = max(0, int(pred_cy - curr_h // 2 - pad))
        x2 = min(width, int(pred_cx + curr_w // 2 + pad))
        y2 = min(height, int(pred_cy + curr_h // 2 + pad))
        search_gray = gray[y1:y2, x1:x2]
        if search_gray.size == 0:
            consecutive_lost += 1
            cx, cy = pred_cx, pred_cy
            curr_scale = pred_scale
            if consecutive_lost > CONSECUTIVE_LOST_MAX:
                cv2.putText(frame, "TARGET LOST", (width//2 - 100, height//2),
                            FONT, FONT_SCALE_LOST, COLOR_RED, FONT_THICKNESS_LOST)
            trajectory.append((cx, cy))
            out.write(frame)
            frame_idx += 1
            continue
        search = clahe.apply(search_gray)

        # 尺度候选：适度放大，防止缩小过头
        scale_candidates = [pred_scale * f for f in FWD_SCALE_FACTORS]
        best_score = -1.0
        best_match = None
        for s in scale_candidates:
            sw = int(base_w * s)
            sh = int(base_h * s)
            if sw < FWD_MIN_TEMPLATE_SIZE or sh < FWD_MIN_TEMPLATE_SIZE \
               or sw > search.shape[1] or sh > search.shape[0]:
                continue
            scaled_tmpl = cv2.resize(curr_tmpl, (sw, sh))
            score, (lx, ly) = fast_ncc_match(search, scaled_tmpl)
            if score > best_score:
                best_score = score
                best_match = (lx, ly, sw, sh, s)

        # 低分重搜
        if best_score < FWD_RESEARCH_MIN_SCORE:
            pad2 = max(FWD_RESEARCH_PAD_MIN, int(curr_w * FWD_RESEARCH_PAD_RATIO))
            x1b = max(0, int(pred_cx - curr_w // 2 - pad2))
            y1b = max(0, int(pred_cy - curr_h // 2 - pad2))
            x2b = min(width, int(pred_cx + curr_w // 2 + pad2))
            y2b = min(height, int(pred_cy + curr_h // 2 + pad2))
            search_gray2 = gray[y1b:y2b, x1b:x2b]
            if search_gray2.size == 0:
                consecutive_lost += 1
                cx, cy = pred_cx, pred_cy
                curr_scale = pred_scale
                if consecutive_lost > CONSECUTIVE_LOST_MAX:
                    cv2.putText(frame, "TARGET LOST", (width//2 - 100, height//2),
                                FONT, FONT_SCALE_LOST, COLOR_RED, FONT_THICKNESS_LOST)
                trajectory.append((cx, cy))
                out.write(frame)
                frame_idx += 1
                continue
            search2 = clahe.apply(search_gray2)
            scale_candidates2 = [pred_scale * f for f in FWD_RESEARCH_SCALE_FACTORS]
            best_score2 = -1.0
            best_match2 = None
            for s in scale_candidates2:
                sw = int(base_w * s)
                sh = int(base_h * s)
                if sw < FWD_MIN_TEMPLATE_SIZE or sh < FWD_MIN_TEMPLATE_SIZE \
                   or sw > search2.shape[1] or sh > search2.shape[0]:
                    continue
                scaled_tmpl = cv2.resize(curr_tmpl, (sw, sh))
                score, (lx, ly) = fast_ncc_match(search2, scaled_tmpl)
                if score > best_score2:
                    best_score2 = score
                    best_match2 = (lx, ly, sw, sh, s, x1b, y1b)
            if best_score2 > best_score + FWD_RESEARCH_SCORE_MARGIN \
               and best_score2 > FWD_RESEARCH_MIN_ACCEPT:
                lx, ly, sw, sh, s, offx, offy = best_match2
                bx, by = offx + lx, offy + ly
                best_score = best_score2
                new_cx = bx + sw // 2
                new_cy = by + sh // 2
            else:
                consecutive_lost += 1
                cx, cy = pred_cx, pred_cy
                curr_scale = pred_scale
                if consecutive_lost > CONSECUTIVE_LOST_MAX:
                    cv2.putText(frame, "TARGET LOST", (width//2 - 100, height//2),
                                FONT, FONT_SCALE_LOST, COLOR_RED, FONT_THICKNESS_LOST)
                trajectory.append((cx, cy))
                out.write(frame)
                frame_idx += 1
                if frame_idx % PROGRESS_INTERVAL == 0:
                    print(f"  处理进度: 第 {frame_idx} 帧 (lost)")
                continue
        else:
            consecutive_lost = 0
            lx, ly, sw, sh, s = best_match
            bx, by = x1 + lx, y1 + ly
            new_cx = bx + sw // 2
            new_cy = by + sh // 2

        # 速度与尺度更新
        raw_vx = new_cx - cx
        raw_vy = new_cy - cy
        vx = FWD_VELOCITY_EMA_ALPHA * vx + (1.0 - FWD_VELOCITY_EMA_ALPHA) * raw_vx
        vy = FWD_VELOCITY_EMA_ALPHA * vy + (1.0 - FWD_VELOCITY_EMA_ALPHA) * raw_vy
        raw_vs = s - curr_scale
        vs = FWD_SCALE_EMA_ALPHA * vs + (1.0 - FWD_SCALE_EMA_ALPHA) * raw_vs
        cx, cy = new_cx, new_cy
        curr_scale = s

        # 模板更新
        if (frame_idx - last_update_frame) >= TEMPLATE_UPDATE_INTERVAL \
           and best_score > TEMPLATE_UPDATE_MIN_SCORE:
            patch = gray[by:by+sh, bx:bx+sw]
            patch = clahe.apply(patch)
            if patch.shape[0] >= TEMPLATE_UPDATE_MIN_PATCH \
               and patch.shape[1] >= TEMPLATE_UPDATE_MIN_PATCH:
                curr_tmpl = cv2.resize(patch, (base_w, base_h)).astype(np.float32)
                last_update_frame = frame_idx
        elif best_score > TEMPLATE_BLEND_MIN_SCORE:
            patch = gray[by:by+sh, bx:bx+sw].astype(np.float32)
            if patch.shape[:2] == (sh, sw):
                patch_resized = cv2.resize(patch, (base_w, base_h))
                curr_tmpl = (TEMPLATE_BLEND_ALPHA_RETAIN * curr_tmpl
                             + (1.0 - TEMPLATE_BLEND_ALPHA_RETAIN) * patch_resized.astype(np.float32))

        cv2.rectangle(frame, (bx, by), (bx + sw, by + sh), COLOR_GREEN, BOX_THICKNESS)
        cv2.putText(frame, f"{best_score:.2f} scl={curr_scale:.2f}",
                    (bx, by - 8), FONT, FONT_SCALE_INFO, COLOR_GREEN, FONT_THICKNESS)

        trajectory.append((cx, cy))
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, COLOR_RED, TRAJECTORY_THICKNESS)
            cv2.circle(frame, (int(cx), int(cy)), CIRCLE_RADIUS, COLOR_BLUE, -1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % PROGRESS_INTERVAL == 0:
            print(f"  处理进度: 第 {frame_idx} 帧")

    cap.release()
    out.release()
    print(f"完成。输出视频: {os.path.join(output_dir, 'tracked_result.mp4')}")


if __name__ == "__main__":
    main()
