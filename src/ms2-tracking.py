import cv2
import numpy as np
import os
import time
from collections import deque

# ======================== 快速 NCC（手写但高效） ========================
def fast_ncc_match(search_img, template):
    """
    基于积分图 + filter2D 的归一化互相关。
    不调 cv2.matchTemplate，速度接近 C++ 实现。
    """
    H, W = search_img.shape
    h, w = template.shape
    if H < h or W < w:
        return -1.0, (0, 0)

    img = search_img.astype(np.float64)
    tpl = template.astype(np.float64)
    N = h * w

    # 积分图（窗口求和 O(1)）
    integral = cv2.integral(img)
    integral_sq = cv2.integral(img * img)

    rows, cols = H - h + 1, W - w + 1
    i_idx = np.arange(rows)[:, None]
    j_idx = np.arange(cols)[None, :]

    # 每个窗口的和与平方和
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
    std_I = np.sqrt(var_I * N)          # sqrt( Σ(I - μ_I)^2 )

    # 模板统计量
    t_mean = tpl.mean()
    t_diff = tpl - t_mean
    t_std = np.sqrt(np.sum(t_diff * t_diff))
    if t_std < 1e-5:
        return -1.0, (0, 0)
    t_sum = tpl.sum()

    # 互相关（filter2D，锚点左上角）
    corr = cv2.filter2D(img, cv2.CV_64F, tpl, anchor=(0, 0))
    corr = corr[:rows, :cols]

    # 合成 NCC
    numerator = corr - mean_I * t_sum
    denominator = std_I * t_std
    ncc_map = np.full((rows, cols), -1.0, dtype=np.float64)
    valid = std_I > 1e-5
    ncc_map[valid] = numerator[valid] / (denominator[valid] + 1e-8)

    max_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    score = float(ncc_map[max_idx])
    y, x = max_idx
    return score, (x, y)


# ====================== 模板自动提取 ======================
def extract_template_from_ref(ref_path):
    """从参考图中提取蓝色/青色椭圆内的目标，返回灰度模板（不扩边）"""
    print(f"从参考图提取模板: {ref_path}")
    img = cv2.imread(ref_path)
    if img is None:
        raise FileNotFoundError(f"无法读取参考图: {ref_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 蓝色/青色掩膜，放宽饱和度
    lower = np.array([75, 20, 20])
    upper = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 屏蔽上下 10% 区域（进度条等干扰）
    mask[:int(h * 0.1), :] = 0
    mask[int(h * 0.9):, :] = 0

    # 闭运算连接断裂轮廓
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

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
        if area < 30:
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
        dist = np.sqrt((cx - img_center[0] + 0.1 * w) ** 2 + (cy - img_center[1]) ** 2)
        pos_score = np.exp(-dist / (0.2 * w))
        score = area * circularity * pos_score
        if score > best_score:
            best_score = score
            best_contour = (cnt, (cx, cy))

    if best_contour is None:
        raise ValueError("没有找到合适的轮廓")

    cnt, (cx, cy) = best_contour
    # 用外接矩形裁切，只扩 2 像素防锯齿，不引入背景
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = 2
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
    max_scan = min(600, total_frames)
    anchor_idx = -1
    anchor_box = None          # (x, y, w, h)
    best_anchor_score = 0.4    # 最低接受阈值
    scan_frames = []
    frame_idx = 0

    while frame_idx < max_scan:
        ret, frame = cap.read()
        if not ret:
            break
        scan_frames.append(frame)
        # 每两帧搜一次
        if frame_idx % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 只在道路区域（下半部分）搜索，减小计算量
            roi = gray[height // 2:, :]
            offset_y = height // 2
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
    # 模板更新用
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
        pad = 50
        x1 = max(0, cx - anchor_box[2] // 2 - pad)
        y1 = max(0, cy - anchor_box[3] // 2 - pad)
        x2 = min(width, cx + anchor_box[2] // 2 + pad)
        y2 = min(height, cy + anchor_box[3] // 2 + pad)
        search = gray[y1:y2, x1:x2]
        # 多尺度
        scales = [curr_scale * 0.92, curr_scale * 0.96, curr_scale, curr_scale * 1.04]
        best_score = -1.0
        best_match = None
        for s in scales:
            sw = int(base_template.shape[1] * s)
            sh = int(base_template.shape[0] * s)
            if sw < 8 or sh < 8 or sw > search.shape[1] or sh > search.shape[0]:
                continue
            scaled = cv2.resize(base_template, (sw, sh))
            score, (lx, ly) = fast_ncc_match(search, scaled)
            if score > best_score:
                best_score = score
                best_match = (lx, ly, sw, sh, s)
        if best_score > 0.4:
            lx, ly, sw, sh, curr_scale = best_match
            bx, by = x1 + lx, y1 + ly
            cx, cy = bx + sw // 2, by + sh // 2
            history[i] = (bx, by, sw, sh)
            # 高置信度时略微更新模板（反向不更新模板本身，只更新位置）
        else:
            # 保持上一帧结果
            history[i] = history.get(i + 1, anchor_box)
            cx = history[i][0] + history[i][2] // 2
            cy = history[i][1] + history[i][3] // 2

    print("反向追踪完成。")

    # ---------- 正向追踪（适应镜头拉远） ----------
    print("开始正向追踪...")
    trajectory = deque(maxlen=2000)
    curr_tmpl = base_template.astype(np.float32)
    curr_scale = 1.0
    base_w, base_h = base_template.shape[1], base_template.shape[0]

    cx = anchor_box[0] + anchor_box[2] // 2
    cy = anchor_box[1] + anchor_box[3] // 2
    vx, vy = 0.0, 0.0
    # 尺度变化速度，用于预测下一帧的尺度
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
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            center = (bx + bw // 2, by + bh // 2)
            trajectory.append(center)
            # 补画轨迹
            if len(trajectory) > 1:
                pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
                cv2.circle(frame, center, 4, (255, 0, 0), -1)
            out.write(frame)
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx == anchor_idx:
            bx, by, bw, bh = anchor_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.putText(frame, "ANCHOR", (bx, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            trajectory.append((cx, cy))
            out.write(frame)
            frame_idx += 1
            continue

        # 预测位置（使用速度平滑，但减小惯性权重）
        pred_cx = cx + vx * 0.9        # 略微阻尼，防止超前
        pred_cy = cy + vy * 0.9
        pred_scale = curr_scale + vs * 0.9   # 预测尺度

        curr_w = int(base_w * pred_scale)
        curr_h = int(base_h * pred_scale)
        # 搜索区域随尺度自适应，尺度越小 pad 相对可稍大，但依然保持紧凑
        pad = max(15, int(curr_w * 0.4))
        x1 = max(0, int(pred_cx - curr_w // 2 - pad))
        y1 = max(0, int(pred_cy - curr_h // 2 - pad))
        x2 = min(width, int(pred_cx + curr_w // 2 + pad))
        y2 = min(height, int(pred_cy + curr_h // 2 + pad))
        search = gray[y1:y2, x1:x2]

        # 多尺度搜索：极宽范围覆盖快速缩小
        # 额外增加一个更小的尺度，确保能跟上
        scale_candidates = [pred_scale * 0.82, pred_scale * 0.90,
                            pred_scale * 0.96, pred_scale, pred_scale * 1.04]
        best_score = -1.0
        best_match = None
        for s in scale_candidates:
            sw = int(base_w * s)
            sh = int(base_h * s)
            if sw < 6 or sh < 6 or sw > search.shape[1] or sh > search.shape[0]:
                continue
            scaled_tmpl = cv2.resize(curr_tmpl, (sw, sh))
            score, (lx, ly) = fast_ncc_match(search, scaled_tmpl)
            if score > best_score:
                best_score = score
                best_match = (lx, ly, sw, sh, s)

        # 如果第一轮得分太低，可能是预测位置偏了，扩大搜索区域再试一次
        if best_score < 0.35:
            # 扩大 pad 和尺度范围重新搜
            pad2 = max(30, int(curr_w * 0.8))
            x1b = max(0, int(pred_cx - curr_w // 2 - pad2))
            y1b = max(0, int(pred_cy - curr_h // 2 - pad2))
            x2b = min(width, int(pred_cx + curr_w // 2 + pad2))
            y2b = min(height, int(pred_cy + curr_h // 2 + pad2))
            search2 = gray[y1b:y2b, x1b:x2b]
            scale_candidates2 = [pred_scale * 0.75, pred_scale * 0.85,
                                 pred_scale * 0.95, pred_scale, pred_scale * 1.05]
            best_score2 = -1.0
            best_match2 = None
            for s in scale_candidates2:
                sw = int(base_w * s)
                sh = int(base_h * s)
                if sw < 6 or sh < 6 or sw > search2.shape[1] or sh > search2.shape[0]:
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
                if consecutive_lost > 30:
                    cv2.putText(frame, "TARGET LOST", (width // 2 - 100, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                trajectory.append((cx, cy))
                out.write(frame)
                frame_idx += 1
                if frame_idx % 50 == 0:
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
        vx = 0.5 * vx + 0.5 * raw_vx
        vy = 0.5 * vy + 0.5 * raw_vy

        # 尺度变化速度更新
        raw_vs = s - curr_scale
        vs = 0.5 * vs + 0.5 * raw_vs

        cx, cy = new_cx, new_cy
        curr_scale = s

        # 模板更新：放宽条件和频率，让模板及时缩小
        if (frame_idx - last_update_frame) >= 8 and best_score > 0.55:
            patch = gray[by:by+sh, bx:bx+sw]
            if patch.shape[0] >= 6 and patch.shape[1] >= 6:
                curr_tmpl = cv2.resize(patch, (base_w, base_h)).astype(np.float32)
                # 重校尺度，确保 base 尺寸下的一致性（可选）
                # 这里保持 base_w/base_h 不变，仅更新 curr_tmpl
                # curr_scale 已在上面更新，表示当前框相对于原 base 的比例
                last_update_frame = frame_idx
        elif best_score > 0.7:
            # 常规微小混合更新
            patch = gray[by:by+sh, bx:bx+sw].astype(np.float32)
            if patch.shape[:2] == (sh, sw):
                patch_resized = cv2.resize(patch, (base_w, base_h))
                curr_tmpl = cv2.addWeighted(curr_tmpl, 0.95, patch_resized, 0.05, 0)

        cv2.rectangle(frame, (bx, by), (bx + sw, by + sh), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_score:.2f} scl={curr_scale:.2f}",
                    (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        trajectory.append((cx, cy))
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  处理进度: 第 {frame_idx} 帧")

    cap.release()
    out.release()
    print(f"完成。输出视频: {os.path.join(output_dir, 'tracked_result.mp4')}")

if __name__ == "__main__":
    main()