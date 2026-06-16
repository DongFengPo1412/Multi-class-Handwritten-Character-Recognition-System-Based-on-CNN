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


def test_single_large_contour_is_split_into_lines():
    binary = np.zeros((150, 120), dtype=np.uint8)
    y_positions = [10, 43, 76, 109]
    for y in y_positions:
        for x in [10, 32, 54, 76]:
            cv2.rectangle(binary, (x, y), (x + 9, y + 19), 255, -1)

    raw_boxes = [(8, 8, 88, 124)]
    boxes = segment_character_boxes(binary, raw_boxes, box_size=150)
    lines = group_boxes_by_reading_lines(boxes)

    assert len(lines) == 4, f"expected 4 text lines, got {len(lines)}: {lines}"
    assert all(len(line) == 4 for line in lines), f"expected 4 chars per line, got {lines}"


if __name__ == "__main__":
    test_wide_connected_box_splitting()
    test_multiline_reading_order()
    test_single_large_contour_is_split_into_lines()
    print("[+] segmentation tests passed")
