import cv2
import numpy as np
from collections import deque

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)


def draw_tracking_overlay(frame, bbox, confidence, trajectory, cue_info=None,
                          lost=False, anchor=False):
    """统一的跟踪可视化。

    bbox: (x, y, w, h)
    cue_info: dict {cue_name: confidence} 或 None
    """
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2

    if anchor:
        color = COLOR_RED
        label = "ANCHOR"
    elif lost:
        color = COLOR_RED
        label = "TARGET LOST"
    else:
        color = COLOR_GREEN
        label = f"{confidence:.2f}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 8), FONT, 0.5, color, 2)

    if cue_info:
        y_offset = y + h + 16
        for name, conf in cue_info.items():
            text = f"{name}:{conf:.2f}"
            cv2.putText(frame, text, (x, y_offset), FONT, 0.4, (200, 200, 200), 1)
            y_offset += 14

    trajectory.append((cx, cy))
    if len(trajectory) > 1:
        pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, COLOR_RED, 2)
        cv2.circle(frame, (cx, cy), 4, COLOR_BLUE, -1)
