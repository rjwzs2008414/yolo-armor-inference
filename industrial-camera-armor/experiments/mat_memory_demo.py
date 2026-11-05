"""Small OpenCV/numpy memory allocation/release demonstration.

This substitutes for the classroom mat_memory sample if that file is not
available.  It shows that numpy/OpenCV buffers are released when references
are dropped; RSS is reported before/after.
"""
from __future__ import annotations

import gc

import cv2
import numpy as np
import psutil


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> None:
    print(f"start RSS: {rss_mb():.1f} MB")
    frames = []
    for i in range(30):
        frame = np.random.randint(0, 256, (1080, 1440, 3), dtype=np.uint8)
        frames.append(cv2.resize(frame, (640, 480)))
    print(f"after 30 frames RSS: {rss_mb():.1f} MB")
    frames.clear()
    gc.collect()
    print(f"after clear + gc RSS: {rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
