import os
import sys

import cv2
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.local_ocr import group_boxes_by_reading_lines, segment_character_boxes


def test_wide_connected_box_splitting():
    binary = np.zeros((48, 112), dtype=np.uint8)
    x0, y0, char_w, gap, h = 8, 8, 10, 8, 30
    for i in range(5):
        left = x0 + i * (char_w + gap)
        cv2.rectangle(binary, (left, y0), (left + char_w - 1, y0 + h - 1), 255, -1)

    raw_boxes = [(x0, y0, 5 * char_w + 4 * gap, h)]
    boxes = segment_character_boxes(binary, raw_boxes, box_size=112)

    assert len(boxes) == 5, f"expected 5 split boxes, got {len(boxes)}: {boxes}"
    assert boxes == sorted(boxes, key=lambda b: b[0])


def test_multiline_reading_order():
    boxes = [
        (40, 50, 10, 20),
        (10, 50, 10, 20),
        (35, 10, 10, 20),
        (5, 10, 10, 20),
    ]
    lines = group_boxes_by_reading_lines(boxes)

    assert lines == [
        [(5, 10, 10, 20), (35, 10, 10, 20)],
        [(10, 50, 10, 20), (40, 50, 10, 20)],
    ]


if __name__ == "__main__":
    test_wide_connected_box_splitting()
    test_multiline_reading_order()
    print("[+] segmentation tests passed")
