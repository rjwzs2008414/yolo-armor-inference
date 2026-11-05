"""Run YOLO on one saved image, optionally with gamma correction."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detect.yolo_detector import ArmorDetector, apply_gamma, enhance_for_detection


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--output", default="results/detect_image.jpg")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--enhance", action="store_true")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    vis = enhance_for_detection(img, args.gamma if args.gamma != 1.0 else 0.7) if args.enhance else apply_gamma(img, args.gamma)
    det = ArmorDetector(args.model, imgsz=args.imgsz, conf=args.conf, device="cpu")
    result = det.infer(vis)
    det.draw(vis, result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    n = 0 if result.boxes is None else len(result.boxes)
    print(f"detections={n} output={args.output}")


if __name__ == "__main__":
    main()
