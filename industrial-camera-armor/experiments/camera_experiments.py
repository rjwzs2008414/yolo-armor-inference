"""Exposure / gain / pixel-format experiments for the MVS camera.

Examples:
  ./run.sh experiments/camera_experiments.py --mode all --label static
  ./run.sh experiments/camera_experiments.py --mode exposure --label moving --seconds 5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from camera.mvs_camera import MvsCamera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["exposure", "gain", "format", "all"], default="all")
    p.add_argument("--label", choices=["static", "moving"], default="static")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out", default="results/experiments")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="if >0, capture for this many seconds instead of --frames")
    return p.parse_args()


def metrics(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "mean_gray": float(gray.mean()),
        "sat_ratio_ge250": float((gray >= 250).mean()),
        "p99_gray": float(np.percentile(gray, 99)),
        "sharpness_lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "min_gray": int(gray.min()),
        "max_gray": int(gray.max()),
    }


def capture(cam: MvsCamera, frames: int, warmup: int, seconds: float):
    for _ in range(max(0, warmup)):
        cam.read()

    process = psutil.Process()
    process.cpu_percent(None)
    started = time.perf_counter()
    convert_times: list[float] = []
    first_bgr = None
    first_raw = None
    count = 0
    while True:
        frame = cam.read()
        if first_bgr is None:
            first_bgr = frame.bgr.copy()
            first_raw = frame.raw.copy()
        convert_times.append(frame.convert_ms)
        count += 1
        if seconds > 0:
            if time.perf_counter() - started >= seconds:
                break
        elif count >= frames:
            break
    elapsed = time.perf_counter() - started
    cpu = process.cpu_percent(None)
    return {
        "frames": count,
        "elapsed_s": elapsed,
        "fps": count / elapsed if elapsed > 0 else 0.0,
        "cpu_percent": cpu,
        "convert_ms_avg": float(np.mean(convert_times)),
        "convert_ms_p95": float(np.percentile(convert_times, 95)),
        "image": first_bgr,
        "raw": first_raw,
    }


def save_run(out_dir: Path, name: str, cam: MvsCamera, result: dict, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{name}.png"
    raw_path = out_dir / f"{name}_raw.npy"
    cv2.imwrite(str(image_path), result["image"])
    np.save(str(raw_path), result["raw"])
    row = {**meta}
    row.update({k: v for k, v in result.items() if k not in ("image", "raw")})
    row.update(metrics(result["image"]))
    row["image"] = str(image_path)
    row["raw"] = str(raw_path)
    return row


def run_exposure(cam: MvsCamera, args, rows: list[dict]) -> None:
    cam.stop()
    cam.set_gain_db(0.0)
    cam.set_pixel_format("BayerRG8")
    for exposure in (500.0, 1000.0, 3000.0):
        cam.stop()
        cam.set_exposure_us(exposure)
        time.sleep(0.2)
        result = capture(cam, args.frames, args.warmup, args.seconds)
        row = save_run(
            Path(args.out) / "exposure",
            f"exposure_{int(exposure)}us_{args.label}",
            cam,
            result,
            {
                "mode": "exposure",
                "label": args.label,
                "setting": f"{exposure:.0f}us",
                "exposure_us": exposure,
                "gain_db": 0.0,
                "pixel_format": cam.pixel_format,
            },
        )
        rows.append(row)
        print(f"[exposure] {exposure:.0f}us mean={row['mean_gray']:.2f} "
              f"sat={row['sat_ratio_ge250']:.4f} fps={row['fps']:.2f}")


def run_gain(cam: MvsCamera, args, rows: list[dict]) -> None:
    cam.stop()
    cam.set_exposure_us(1000.0)
    cam.set_pixel_format("BayerRG8")
    for gain in (0.0, 6.0, 12.0):
        cam.stop()
        cam.set_gain_db(gain)
        time.sleep(0.2)
        result = capture(cam, args.frames, args.warmup, args.seconds)
        row = save_run(
            Path(args.out) / "gain",
            f"gain_{int(gain)}db_{args.label}",
            cam,
            result,
            {
                "mode": "gain",
                "label": args.label,
                "setting": f"{gain:.0f}dB",
                "exposure_us": 1000.0,
                "gain_db": gain,
                "pixel_format": cam.pixel_format,
            },
        )
        rows.append(row)
        print(f"[gain] {gain:.0f}dB mean={row['mean_gray']:.2f} "
              f"sat={row['sat_ratio_ge250']:.4f} fps={row['fps']:.2f}")


def run_format(cam: MvsCamera, args, rows: list[dict]) -> None:
    cam.stop()
    cam.set_exposure_us(1000.0)
    cam.set_gain_db(0.0)
    for fmt in ("BayerRG8", "BGR8"):
        cam.stop()
        try:
            cam.set_pixel_format(fmt)
        except Exception as exc:
            print(f"[format] {fmt} not supported: {exc}")
            rows.append(
                {
                    "mode": "format",
                    "label": args.label,
                    "setting": fmt,
                    "exposure_us": 1000.0,
                    "gain_db": 0.0,
                    "pixel_format": fmt,
                    "error": str(exc),
                }
            )
            continue
        time.sleep(0.2)
        result = capture(cam, args.frames, args.warmup, args.seconds)
        row = save_run(
            Path(args.out) / "format",
            f"format_{fmt.lower()}_{args.label}",
            cam,
            result,
            {
                "mode": "format",
                "label": args.label,
                "setting": fmt,
                "exposure_us": 1000.0,
                "gain_db": 0.0,
                "pixel_format": fmt,
            },
        )
        rows.append(row)
        print(f"[format] {fmt} fps={row['fps']:.2f} cpu={row['cpu_percent']:.1f}% "
              f"convert={row['convert_ms_avg']:.2f}ms")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with MvsCamera(0) as cam:
        print(f"camera: {cam.info}")
        if args.mode in ("exposure", "all"):
            run_exposure(cam, args, rows)
        if args.mode in ("gain", "all"):
            run_gain(cam, args, rows)
        if args.mode in ("format", "all"):
            run_format(cam, args, rows)
    if rows:
        fields = sorted({k for row in rows for k in row})
        csv_path = out_dir / f"experiment_{args.mode}_{args.label}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        json_path = out_dir / f"experiment_{args.mode}_{args.label}.json"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {csv_path}")
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
