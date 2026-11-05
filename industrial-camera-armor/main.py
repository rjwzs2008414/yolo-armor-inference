"""Real-time MVS camera + YOLO armor detection.

Run:
  ./run.sh main.py --model models/best.pt --exposure 1000 --gain 0 \
      --format BayerRG8 --headless --seconds 20 --output results/live_demo.mp4
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from camera.mvs_camera import MvsCamera
from detect.yolo_detector import ArmorDetector, apply_gamma, enhance_for_detection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--exposure", type=float, default=1000.0, help="exposure time in us")
    p.add_argument("--gain", type=float, default=0.0, help="gain in dB")
    p.add_argument("--format", default="BayerRG8", choices=["BayerRG8", "BGR8", "RGB8"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="gamma < 1 brightens dark images, e.g. 0.35")
    p.add_argument("--enhance", action="store_true",
                   help="unsharp + bloom compensation for an unlit armor plate")
    p.add_argument("--seconds", type=float, default=0.0, help="0 means until Ctrl+C")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--output", default="results/live_demo.mp4")
    p.add_argument("--headless", action="store_true", help="do not open a GUI window")
    p.add_argument("--snapshot-interval", type=float, default=0.0,
                   help="save a jpg every N seconds, 0 disables")
    p.add_argument("--log", default="")
    return p.parse_args()


def setup_logging(path: str) -> None:
    handlers = [logging.StreamHandler()]
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log or f"logs/run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot_interval > 0:
        Path("results/live").mkdir(parents=True, exist_ok=True)

    detector = ArmorDetector(
        args.model, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device
    )
    logging.info("model loaded: %s", args.model)
    logging.info("classes: %s", detector.names)

    writer = None
    frame_count = 0
    started = time.perf_counter()
    next_snapshot = args.snapshot_interval
    fps_window: deque[float] = deque(maxlen=30)

    cam = MvsCamera(args.camera)
    try:
        cam.open()
        cam.set_pixel_format(args.format)
        cam.set_exposure_us(args.exposure)
        cam.set_gain_db(args.gain)
        logging.info("camera: %s", cam.info)
        logging.info(
            "settings: format=%s exposure=%.1fus gain=%.1fdB",
            args.format, args.exposure, args.gain,
        )
        cam.start()

        while True:
            loop_start = time.perf_counter()
            frame_data = cam.read()
            if args.enhance:
                frame = enhance_for_detection(frame_data.bgr, args.gamma if args.gamma != 1.0 else 0.7)
            else:
                frame = apply_gamma(frame_data.bgr, args.gamma)
            result = detector.infer(frame)
            detector.draw(frame, result)

            elapsed = time.perf_counter() - started
            if args.seconds > 0 and elapsed >= args.seconds:
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot create output video: {out_path}")

            fps_window.append(1.0 / max(1e-6, time.perf_counter() - loop_start))
            fps = sum(fps_window) / len(fps_window)
            det_count = 0 if result.boxes is None else len(result.boxes)
            cv2.putText(
                frame,
                f"FPS {fps:5.1f}  det {det_count}  exp {args.exposure:.0f}us "
                f"gain {args.gain:.0f}dB gamma {args.gamma:.2f} "
                f"enhance {int(args.enhance)}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

            if args.snapshot_interval > 0 and elapsed >= next_snapshot:
                snap = Path("results/live") / f"snap_{elapsed:07.2f}s.jpg"
                cv2.imwrite(str(snap), frame)
                next_snapshot += args.snapshot_interval

            if not args.headless:
                cv2.imshow("armor", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                logging.info(
                    "frames=%d fps=%.1f det=%d convert=%.2fms",
                    frame_count, fps, det_count, frame_data.convert_ms,
                )
    finally:
        cam.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logging.info("stopped after %d frames", frame_count)


if __name__ == "__main__":
    main()
