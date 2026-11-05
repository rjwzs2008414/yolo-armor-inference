"""Continuously write results/preview.jpg for browser-based lens adjustment."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from camera.mvs_camera import MvsCamera
from detect.yolo_detector import apply_gamma, enhance_for_detection


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exposure", type=float, default=10000.0)
    p.add_argument("--gain", type=float, default=6.0)
    p.add_argument("--format", default="BayerRG8")
    p.add_argument("--out", default="results/preview.jpg")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--enhance", action="store_true")
    p.add_argument("--seconds", type=float, default=0.0)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with MvsCamera(0) as cam:
        cam.set_pixel_format(args.format)
        cam.set_exposure_us(args.exposure)
        cam.set_gain_db(args.gain)
        cam.start()
        print(f"camera {cam.info}", flush=True)
        n = 0
        while True:
            frame = cam.read()
            if args.enhance:
                img = enhance_for_detection(frame.bgr, args.gamma if args.gamma != 1.0 else 0.7)
            else:
                img = apply_gamma(frame.bgr, args.gamma)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.putText(
                img,
                f"exp {args.exposure:.0f}us gain {args.gain:.0f}dB gamma {args.gamma:.2f} mean {gray.mean():.1f}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )
            tmp = out.with_suffix(".tmp.jpg")
            cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            os.replace(tmp, out)
            n += 1
            if n % 20 == 0:
                print(f"frames={n} mean={gray.mean():.1f} fps~{n/(time.perf_counter()-started):.1f}", flush=True)
            if args.seconds > 0 and time.perf_counter() - started >= args.seconds:
                break
            time.sleep(0.05)


if __name__ == "__main__":
    main()
