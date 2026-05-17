import cv2
import numpy as np
import os
import time
from collections import deque

from ncc import fast_ncc_match

# ======================== 模板提取参数 ========================
TM_HSV_LOWER = np.array([75, 20, 20])
TM_HSV_UPPER = np.array([150, 255, 255])
TM_BORDER_MASK_RATIO = 0.1         # 屏蔽上下边缘比例
TM_MORPH_KERNEL_SIZE = (3, 3)
TM_MORPH_ITERATIONS = 1
TM_MIN_CONTOUR_AREA = 30
TM_POS_BIAS_X = 0.1               # 位置评分偏左偏好系数
TM_POS_DECAY_W = 0.2              # 位置评分衰减（相对于图像宽度）
TM_TEMPLATE_PAD = 2               # 裁切扩边像素

# ======================== 锚点扫描参数 ========================
ANCHOR_MAX_SCAN_FRAMES = 600
ANCHOR_SCAN_STEP = 2              # 每隔 N 帧搜索一次
ANCHOR_MIN_SCORE = 0.4
ANCHOR_ROI_START_RATIO = 2        # 搜索区域从 height // 2 开始（道路区域）

# ======================== 反向追踪参数 ========================
REVERSE_SEARCH_PAD = 50
REVERSE_SCALE_FACTORS = [0.92, 0.96, 1.0, 1.04]
REVERSE_MIN_SCORE = 0.4
REVERSE_MIN_TEMPLATE_SIZE = 8

# ======================== 正向追踪参数 ========================
FORWARD_VELOCITY_DAMPING = 0.9    # 预测位置阻尼系数
FORWARD_SCALE_DAMPING = 0.9       # 预测尺度阻尼系数
FORWARD_SEARCH_PAD_RATIO = 0.4    # 搜索 pad 相对于目标宽度的比例
FORWARD_SEARCH_PAD_MIN = 15
FORWARD_SCALE_FACTORS = [0.82, 0.90, 0.96, 1.0, 1.04]
FORWARD_MIN_TEMPLATE_SIZE = 6
FORWARD_VELOCITY_EMA_ALPHA = 0.5  # 速度指数平滑系数

# ======================== 正向追踪 – 低分重搜参数 ========================
FORWARD_RESEARCH_MIN_SCORE = 0.35
FORWARD_RESEARCH_PAD_RATIO = 0.8
FORWARD_RESEARCH_PAD_MIN = 30
FORWARD_RESEARCH_SCALE_FACTORS = [0.75, 0.85, 0.95, 1.0, 1.05]

# ======================== 模板更新参数 ========================
TEMPLATE_UPDATE_INTERVAL = 8      # 帧间隔
TEMPLATE_UPDATE_MIN_SCORE = 0.55  # 低于此分不更新
TEMPLATE_BLEND_MIN_SCORE = 0.7    # 高于此分做微小混合
TEMPLATE_BLEND_ALPHA_NEW = 0.05   # 混合时新模板权重
TEMPLATE_UPDATE_MIN_PATCH = 6     # patch 最小尺寸
TEMPLATE_SCALE_EMA_ALPHA = 0.5    # 尺度变化速度平滑系数

# ======================== 丢失恢复参数 ========================
CONSECUTIVE_LOST_MAX = 30         # 连续丢失多少帧后显示 TARGET LOST

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


# ====================== 模板自动提取 ======================
def extract_template_from_ref(ref_path):
    """从参考图中提取蓝色/青色椭圆内的目标，返回灰度模板（不扩边）"""
    print(f"从参考图提取模板: {ref_path}")
    img = cv2.imread(ref_path)
    if img is None:
        raise FileNotFoundError(f"无法读取参考图: {ref_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色/青色掩膜
    mask = cv2.inRange(hsv, TM_HSV_LOWER, TM_HSV_UPPER)

    # 屏蔽上下边缘区域（进度条等干扰）
    mask[:int(h * TM_BORDER_MASK_RATIO), :] = 0
    mask[int(h * (1.0 - TM_BORDER_MASK_RATIO)):, :] = 0

    # 闭运算连接断裂轮廓
    kernel = np.ones(TM_MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=TM_MORPH_ITERATIONS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imwrite("output/task2/debug_mask.png", mask)
        raise ValueError("未找到彩色区域，已保存 debug_mask.png 供检查")

    # 筛选最佳轮廓：面积、圆度、位置（偏左优先）
    best_contour = None
    best_score = -1
    img_center = (w // 2, h // 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TM_MIN_CONTOUR_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist = np.sqrt((cx - img_center[0] + TM_POS_BIAS_X * w) ** 2 + (cy - img_center[1]) ** 2)
        pos_score = np.exp(-dist / (TM_POS_DECAY_W * w))
        score = area * circularity * pos_score
        if score > best_score:
            best_score = score
            best_contour = (cnt, (cx, cy))

    if best_contour is None:
        raise ValueError("没有找到合适的轮廓")

    cnt, (cx, cy) = best_contour
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = TM_TEMPLATE_PAD
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    patch = img[y1:y2, x1:x2]
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output/task2/auto_template.png", gray_patch)
    print(f"模板提取成功，尺寸 {gray_patch.shape[1]}x{gray_patch.shape[0]}")
    return gray_patch


# ====================== 主程序 ======================
def main():
    # 路径配置
    video_path = "data/task2/大疆无人机航拍视频.mp4"
    ref_image_path = "data/task2/大疆无人机航拍视频目标.png"
    output_dir = "output/task2"
    os.makedirs(output_dir, exist_ok=True)

    # 检查文件
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

    # ---------- 阶段一：扫描寻找锚点帧 ----------
    print("开始扫描锚点帧（搜索前 600 帧）...")
    max_scan = min(ANCHOR_MAX_SCAN_FRAMES, total_frames)
    anchor_idx = -1
    anchor_box = None          # (x, y, w, h)
    best_anchor_score = ANCHOR_MIN_SCORE
    scan_frames = []
    frame_idx = 0

    while frame_idx < max_scan:
        ret, frame = cap.read()
        if not ret:
            break
        scan_frames.append(frame)
        # 每隔 ANCHOR_SCAN_STEP 帧搜一次
        if frame_idx % ANCHOR_SCAN_STEP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 只在道路区域（下半部分）搜索，减小计算量
            roi = gray[height // ANCHOR_ROI_START_RATIO:, :]
            offset_y = height // ANCHOR_ROI_START_RATIO
            score, (lx, ly) = fast_ncc_match(roi, base_template)
            if score > best_anchor_score:
                best_anchor_score = score
                anchor_idx = frame_idx
                anchor_box = (lx, ly + offset_y, base_template.shape[1], base_template.shape[0])
                print(f"  发现更优锚点: 帧 {anchor_idx}, 得分 {score:.3f}")
        frame_idx += 1

    if anchor_idx == -1:
        print("错误: 在前 600 帧内未找到目标，请检查视频或参考图。")
        cap.release()
        return

    print(f"锚点定位完成: 帧 {anchor_idx}, 得分 {best_anchor_score:.3f}")

    # 重置视频，准备正式处理
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(
        os.path.join(output_dir, "tracked_result.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    # ---------- 阶段二：利用已缓存的帧进行反向追踪 ----------
    print("开始反向追踪 (时光倒流) ...")
    curr_tmpl = base_template.astype(np.float32)
    curr_scale = 1.0
    history = {}                     # 帧序号 -> (x,y,w,h)
    history[anchor_idx] = anchor_box
    cx = anchor_box[0] + anchor_box[2] // 2
    cy = anchor_box[1] + anchor_box[3] // 2

    # 反向从 anchor_idx-1 到 0
    for i in range(anchor_idx - 1, -1, -1):
        frame = scan_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 搜索区域在上次位置周围扩展
        pad = REVERSE_SEARCH_PAD
        x1 = max(0, cx - anchor_box[2] // 2 - pad)
        y1 = max(0, cy - anchor_box[3] // 2 - pad)
        x2 = min(width, cx + anchor_box[2] // 2 + pad)
        y2 = min(height, cy + anchor_box[3] // 2 + pad)
        search = gray[y1:y2, x1:x2]
        # 多尺度
        scales = [curr_scale * f for f in REVERSE_SCALE_FACTORS]
        best_score = -1.0
        best_match = None
        for s in scales:
            sw = int(base_template.shape[1] * s)
            sh = int(base_template.shape[0] * s)
            if sw < REVERSE_MIN_TEMPLATE_SIZE or sh < REVERSE_MIN_TEMPLATE_SIZE \
               or sw > search.shape[1] or sh > search.shape[0]:
                continue
            scaled = cv2.resize(base_template, (sw, sh))
            score, (lx, ly) = fast_ncc_match(search, scaled)
            if score > best_score:
                best_score = score
                best_match = (lx, ly, sw, sh, s)
        if best_score > REVERSE_MIN_SCORE:
            lx, ly, sw, sh, curr_scale = best_match
            bx, by = x1 + lx, y1 + ly
            cx, cy = bx + sw // 2, by + sh // 2
            history[i] = (bx, by, sw, sh)
        else:
            # 保持上一帧结果
            history[i] = history.get(i + 1, anchor_box)
            cx = history[i][0] + history[i][2] // 2
            cy = history[i][1] + history[i][3] // 2

    print("反向追踪完成。")

    # ---------- 正向追踪（适应镜头拉远） ----------
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

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 已缓存的反向帧直接绘制历史框
        if frame_idx < anchor_idx and frame_idx in history:
            bx, by, bw, bh = history[frame_idx]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_GREEN, BOX_THICKNESS)
            center = (bx + bw // 2, by + bh // 2)
            trajectory.append(center)
            # 补画轨迹
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, COLOR_RED, TRAJECTORY_THICKNESS)
                cv2.circle(frame, center, CIRCLE_RADIUS, COLOR_BLUE, -1)
            out.write(frame)
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx == anchor_idx:
            bx, by, bw, bh = anchor_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_RED, BOX_THICKNESS)
            cv2.putText(frame, "ANCHOR", (bx, by - 10),
                        FONT, FONT_SCALE_ANCHOR, COLOR_RED, FONT_THICKNESS)
            trajectory.append((cx, cy))
            out.write(frame)
            frame_idx += 1
            continue

        # 预测位置（使用速度平滑，但减小惯性权重）
        pred_cx = cx + vx * FORWARD_VELOCITY_DAMPING
        pred_cy = cy + vy * FORWARD_VELOCITY_DAMPING
        pred_scale = curr_scale + vs * FORWARD_SCALE_DAMPING

        curr_w = int(base_w * pred_scale)
        curr_h = int(base_h * pred_scale)
        # 搜索区域随尺度自适应
        pad = max(FORWARD_SEARCH_PAD_MIN, int(curr_w * FORWARD_SEARCH_PAD_RATIO))
        x1 = max(0, int(pred_cx - curr_w // 2 - pad))
        y1 = max(0, int(pred_cy - curr_h // 2 - pad))
        x2 = min(width, int(pred_cx + curr_w // 2 + pad))
        y2 = min(height, int(pred_cy + curr_h // 2 + pad))
        search = gray[y1:y2, x1:x2]

        # 多尺度搜索
        scale_candidates = [pred_scale * f for f in FORWARD_SCALE_FACTORS]
        best_score = -1.0
        best_match = None
        for s in scale_candidates:
            sw = int(base_w * s)
            sh = int(base_h * s)
            if sw < FORWARD_MIN_TEMPLATE_SIZE or sh < FORWARD_MIN_TEMPLATE_SIZE \
               or sw > search.shape[1] or sh > search.shape[0]:
                continue
            scaled_tmpl = cv2.resize(curr_tmpl, (sw, sh))
            score, (lx, ly) = fast_ncc_match(search, scaled_tmpl)
            if score > best_score:
                best_score = score
                best_match = (lx, ly, sw, sh, s)

        # 如果第一轮得分太低，可能是预测位置偏了，扩大搜索区域再试一次
        if best_score < FORWARD_RESEARCH_MIN_SCORE:
            pad2 = max(FORWARD_RESEARCH_PAD_MIN, int(curr_w * FORWARD_RESEARCH_PAD_RATIO))
            x1b = max(0, int(pred_cx - curr_w // 2 - pad2))
            y1b = max(0, int(pred_cy - curr_h // 2 - pad2))
            x2b = min(width, int(pred_cx + curr_w // 2 + pad2))
            y2b = min(height, int(pred_cy + curr_h // 2 + pad2))
            search2 = gray[y1b:y2b, x1b:x2b]
            scale_candidates2 = [pred_scale * f for f in FORWARD_RESEARCH_SCALE_FACTORS]
            best_score2 = -1.0
            best_match2 = None
            for s in scale_candidates2:
                sw = int(base_w * s)
                sh = int(base_h * s)
                if sw < FORWARD_MIN_TEMPLATE_SIZE or sh < FORWARD_MIN_TEMPLATE_SIZE \
                   or sw > search2.shape[1] or sh > search2.shape[0]:
                    continue
                scaled_tmpl = cv2.resize(curr_tmpl, (sw, sh))
                score, (lx, ly) = fast_ncc_match(search2, scaled_tmpl)
                if score > best_score2:
                    best_score2 = score
                    best_match2 = (lx, ly, sw, sh, s, x1b, y1b)  # 带上偏移
            if best_score2 > best_score:
                # 用扩区域的结果
                lx, ly, sw, sh, s, offx, offy = best_match2
                bx, by = offx + lx, offy + ly
                best_score = best_score2
                best_match = (lx, ly, sw, sh, s)
                new_cx = bx + sw // 2
                new_cy = by + sh // 2
            else:
                # 依然没找到，维持预测
                consecutive_lost += 1
                cx, cy = pred_cx, pred_cy
                curr_scale = pred_scale
                if consecutive_lost > CONSECUTIVE_LOST_MAX:
                    cv2.putText(frame, "TARGET LOST", (width // 2 - 100, height // 2),
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

        # 速度更新（缩小惯性权重，更快适应变化）
        raw_vx = new_cx - cx
        raw_vy = new_cy - cy
        vx = FORWARD_VELOCITY_EMA_ALPHA * vx + (1.0 - FORWARD_VELOCITY_EMA_ALPHA) * raw_vx
        vy = FORWARD_VELOCITY_EMA_ALPHA * vy + (1.0 - FORWARD_VELOCITY_EMA_ALPHA) * raw_vy

        # 尺度变化速度更新
        raw_vs = s - curr_scale
        vs = TEMPLATE_SCALE_EMA_ALPHA * vs + (1.0 - TEMPLATE_SCALE_EMA_ALPHA) * raw_vs

        cx, cy = new_cx, new_cy
        curr_scale = s

        # 模板更新：放宽条件和频率，让模板及时缩小
        if (frame_idx - last_update_frame) >= TEMPLATE_UPDATE_INTERVAL \
           and best_score > TEMPLATE_UPDATE_MIN_SCORE:
            patch = gray[by:by+sh, bx:bx+sw]
            if patch.shape[0] >= TEMPLATE_UPDATE_MIN_PATCH \
               and patch.shape[1] >= TEMPLATE_UPDATE_MIN_PATCH:
                curr_tmpl = cv2.resize(patch, (base_w, base_h)).astype(np.float32)
                last_update_frame = frame_idx
        elif best_score > TEMPLATE_BLEND_MIN_SCORE:
            # 常规微小混合更新
            patch = gray[by:by+sh, bx:bx+sw].astype(np.float32)
            if patch.shape[:2] == (sh, sw):
                patch_resized = cv2.resize(patch, (base_w, base_h))
                curr_tmpl = cv2.addWeighted(curr_tmpl, 1.0 - TEMPLATE_BLEND_ALPHA_NEW,
                                            patch_resized, TEMPLATE_BLEND_ALPHA_NEW, 0)

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
