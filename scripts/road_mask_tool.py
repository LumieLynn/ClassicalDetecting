"""交互式路面蒙版标注工具。

锚点帧上点击 4 个点定义路面梯形区域。
左键点击、右键撤销、r 重置、Enter 确认。
"""

import cv2
import numpy as np
import os
import sys

ANCHOR_FRAME_IDX = 86
VIDEO_PATH = "data/task3/大疆无人机航拍骑车人.mp4"
OUTPUT_DIR = "output/masktool"


def _order_convex(pts):
    """按极角排序 4 个点 → 凸四边形（顺时针）。"""
    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    return sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    for _ in range(ANCHOR_FRAME_IDX + 1):
        ret, frame = cap.read()
    cap.release()

    if frame is None:
        print(f"错误: 无法读取帧 {ANCHOR_FRAME_IDX}")
        sys.exit(1)

    disp = frame.copy()
    points = []
    tmp_pt = None

    def draw_all():
        nonlocal disp
        disp = frame.copy()
        h, w = disp.shape[:2]

        # 画已确认的点
        for i, (px, py) in enumerate(points):
            cv2.circle(disp, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 画连线
        for i in range(1, len(points)):
            cv2.line(disp, points[i - 1], points[i], (0, 255, 255), 2)

        # 画临时点
        if tmp_pt is not None:
            cv2.circle(disp, tmp_pt, 4, (255, 0, 0), -1)

        # 画填充梯形（自动排序为凸四边形）
        if len(points) == 4:
            pts_sorted = _order_convex(points)
            overlay = disp.copy()
            pts_arr = np.array(pts_sorted, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
            cv2.polylines(disp, [pts_arr], True, (0, 255, 0), 2)

    def on_click(event, x, y, flags, param):
        nonlocal tmp_pt
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            tmp_pt = None
            draw_all()
            print(f"  点 {len(points)}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            removed = points.pop()
            draw_all()
            print(f"  撤销点: {removed}")
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(points) < 4:
                tmp_pt = (x, y)
                draw_all()

    draw_all()
    cv2.namedWindow("road mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("road mask", 960, 768)
    cv2.setMouseCallback("road mask", on_click)

    print("锚点帧 86 — 请沿路面边界点击 4 个点（梯形）")
    print("  左键: 加点   右键: 撤销   r: 重置   Enter/ESC: 确认退出")
    print("  建议: 左上、左下、右下、右上（或 右上、右下、左下、左上）")
    print()

    while True:
        cv2.imshow("road mask", disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:   # ESC
            break
        if key == 13:   # Enter
            if len(points) == 4:
                break
            else:
                print(f"  需要 4 个点，当前 {len(points)} 个")
        if key == ord('r'):
            points.clear()
            tmp_pt = None
            draw_all()
            print("  已重置")

    cv2.destroyAllWindows()

    if len(points) != 4:
        print("未完成标注，退出。")
        sys.exit(0)

    # 排序后保存
    pts_sorted = _order_convex(points)
    pts_str = repr(pts_sorted)
    print(f"\nMANUAL_ROAD_PTS = {pts_str}")

    # 保存标注截图
    draw_all()
    screenshot_path = os.path.join(OUTPUT_DIR, "road_mask_annotated.png")
    cv2.imwrite(screenshot_path, disp)
    print(f"标注截图: {screenshot_path}")

    # 保存坐标到文件
    coords_path = os.path.join(OUTPUT_DIR, "road_mask_pts.txt")
    with open(coords_path, "w") as f:
        f.write(pts_str + "\n")
    print(f"坐标文件: {coords_path}")


if __name__ == "__main__":
    main()
