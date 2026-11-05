"""Supplementary motion-blur simulation on a real static camera frame.

The lab armor plate LED is broken, so a real moving-target sequence could not
be captured reliably.  This script applies horizontal motion-blur kernels to a
real camera frame and measures the detector confidence drop.  It is clearly a
simulation, not a replacement for the required moving-target experiment.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from detect.yolo_detector import ArmorDetector, enhance_for_detection


def blur_kernel(length: int) -> np.ndarray:
    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)
    kernel = np.zeros((1, length), dtype=np.float32)
    kernel[0, :] = 1.0 / length
    return kernel


def main() -> None:
    source = Path("results/detection_raw.jpg")
    img = cv2.imread(str(source))
    if img is None:
        raise SystemExit(f"cannot read {source}")
    out_dir = Path("results/motion_blur")
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = ArmorDetector("models/best.pt", imgsz=960, conf=0.05, device="cpu")
    rows = []
    for length in (1, 5, 11, 21, 31):
        blurred = cv2.filter2D(img, -1, blur_kernel(length))
        vis = enhance_for_detection(blurred, 0.7)
        result = detector.infer(vis)
        boxes = result.boxes
        conf = float(boxes.conf[0]) if boxes is not None and len(boxes) else 0.0
        cls = int(boxes.cls[0]) if boxes is not None and len(boxes) else -1
        count = 0 if boxes is None else len(boxes)
        out_path = out_dir / f"blur_{length:02d}.jpg"
        detector.draw(vis, result)
        cv2.imwrite(str(out_path), vis)
        rows.append(
            {
                "blur_length_px": length,
                "detections": count,
                "best_conf": round(conf, 3),
                "class_id": cls,
                "image": str(out_path),
            }
        )
        print(length, count, round(conf, 3), cls)
    csv_path = Path("results/motion_blur_simulation.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
