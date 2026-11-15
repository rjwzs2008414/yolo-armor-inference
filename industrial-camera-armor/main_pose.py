"""Camera -> YOLO armor -> 4 corners -> PnP -> Kalman -> prediction.

Examples:
  # offline image
  ./.venv/bin/python main_pose.py --source image --image results/detection_raw.jpg \
      --calib configs/camera.yaml --enhance --headless --output results/pose_image.jpg

  # live MVS camera (XRDP GUI)
  ./.venv/bin/python main_pose.py --source camera --calib configs/camera.yaml \
      --exposure 30000 --gain 12 --enhance --imgsz 960 --conf 0.15

  # headless video
  ./.venv/bin/python main_pose.py --source camera --headless --seconds 20 \
      --output results/pose_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detect.armor_pose import (
    ArmorPose,
    CameraCalibration,
    PoseKalmanFilter,
    load_calibration,
    order_corners,
    solve_pose,
)
from detect.yolo_detector import ArmorDetector, apply_gamma, enhance_for_detection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["camera", "image", "video"], default="camera")
    p.add_argument("--image", default="results/detection_raw.jpg")
    p.add_argument("--video", default="")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--calib", default="configs/camera.yaml")
    p.add_argument("--armor-width", type=float, default=None,
                   help="override armor width in meters")
    p.add_argument("--armor-height", type=float, default=None,
                   help="override armor height in meters")
    p.add_argument("--exposure", type=float, default=30000.0)
    p.add_argument("--gain", type=float, default=12.0)
    p.add_argument("--format", default="BayerRG8", choices=["BayerRG8", "BGR8", "RGB8"])
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--enhance", action="store_true")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", default="results/pose_demo.mp4")
    p.add_argument("--jsonl", default="results/pose_log.jsonl")
    p.add_argument("--kalman-horizon", type=float, default=0.2,
                   help="prediction horizon in seconds")
    return p.parse_args()


def preprocess(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.enhance:
        return enhance_for_detection(
            frame, args.gamma if args.gamma != 1.0 else 0.7
        )
    return apply_gamma(frame, args.gamma)


def draw_pose_overlay(vis: np.ndarray, result, pose: Optional[ArmorPose],
                      calib: CameraCalibration,
                      filtered: Optional[np.ndarray],
                      predicted: Optional[np.ndarray],
                      future: Optional[np.ndarray],
                      velocity: Optional[np.ndarray],
                      best_index: Optional[int],
                      horizon: float,
                      fps: float) -> np.ndarray:
    h, w = vis.shape[:2]
    cv2.putText(vis, f"FPS {fps:5.1f}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if best_index is None or result.boxes is None:
        cv2.putText(vis, "no armor detection", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    boxes = result.boxes
    conf = float(boxes.conf[best_index].item())
    cls_id = int(boxes.cls[best_index].item())
    names = getattr(result, "names", {}) or {}
    name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_index].tolist())
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"{name} {conf:.2f}", (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    if pose is None:
        cv2.putText(vis, "pose failed", (x1, min(h - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    # draw the ordered quad
    quad = pose.image_points.astype(int)
    cv2.polylines(vis, [quad], True, (0, 255, 255), 2)

    try:
        cv2.drawFrameAxes(vis, calib.camera_matrix, calib.dist_coeffs,
                          pose.rvec, pose.tvec, 0.05, 2)
    except Exception:
        pass

    x, y, z = pose.tvec
    lines = [
        f"XYZ(m): {x:+.3f} {y:+.3f} {z:+.3f}",
        f"dist: {pose.distance_m:.3f} m  reproj: {pose.reprojection_error_px:.2f}px",
        f"yaw/pitch/roll: {pose.euler_deg[0]:+.1f} {pose.euler_deg[1]:+.1f} {pose.euler_deg[2]:+.1f} deg",
    ]
    if filtered is not None:
        lines.append(f"KF: {filtered[0]:+.3f} {filtered[1]:+.3f} {filtered[2]:+.3f}")
    if predicted is not None:
        lines.append(f"KF pred: {predicted[0]:+.3f} {predicted[1]:+.3f} {predicted[2]:+.3f}")
    if future is not None:
        lines.append(f"future +{horizon:.2f}s: {future[0]:+.3f} {future[1]:+.3f} {future[2]:+.3f}")
    if velocity is not None:
        lines.append(f"v(m/s): {velocity[0]:+.3f} {velocity[1]:+.3f} {velocity[2]:+.3f}")

    y0 = max(58, y1 - 10 - 18 * len(lines))
    for i, text in enumerate(lines):
        cv2.putText(vis, text, (12, y0 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def process_frame(frame: np.ndarray,
                  frame_index: int,
                  dt: float,
                  detector: ArmorDetector,
                  calib: CameraCalibration,
                  kf: PoseKalmanFilter,
                  args: argparse.Namespace,
                  log_file) -> tuple[np.ndarray, Optional[dict]]:
    """Run one full pipeline and return (visualization, json record)."""
    vis = preprocess(frame, args)
    result = detector.infer(vis)

    # Draw raw model keypoints only; the ordered quad and label are drawn
    # later in draw_pose_overlay to avoid duplicate boxes/labels.
    if result.keypoints is not None and result.keypoints.xy is not None:
        for person_kpts in result.keypoints.xy:
            for point in person_kpts:
                px, py = int(point[0]), int(point[1])
                if px == 0 and py == 0:
                    continue
                cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
                cv2.circle(vis, (px, py), 1, (255, 255, 255), -1)

    best_index = None
    pose = None
    filtered = predicted = future = velocity = None
    corners = None
    if result.boxes is not None and len(result.boxes) > 0:
        best_index = int(result.boxes.conf.argmax().item())
        if result.keypoints is not None and result.keypoints.xy is not None:
            try:
                kpts = result.keypoints.xy[best_index].detach().cpu().numpy()
            except Exception:
                kpts = np.asarray(result.keypoints.xy[best_index])
            if kpts.shape == (4, 2) and np.isfinite(kpts).all() and not np.allclose(kpts, 0):
                corners = order_corners(kpts)
                pose = solve_pose(corners, calib,
                                  width_m=calib.armor_width_m,
                                  height_m=calib.armor_height_m)
                if pose is not None:
                    predicted, filtered, velocity, _ = kf.update(pose.tvec, dt)
                    future = kf.predict_ahead(args.kalman_horizon)
    if pose is None:
        kf.predict_only(dt)

    fps = 1.0 / max(1e-6, dt)
    vis = draw_pose_overlay(vis, result, pose, calib, filtered, predicted,
                            future, velocity, best_index,
                            args.kalman_horizon, fps)

    record = None
    if pose is not None:
        record = {
            "time": time.time(),
            "frame": frame_index,
            "class_id": int(result.boxes.cls[best_index].item()),
            "class_name": (
                result.names.get(int(result.boxes.cls[best_index].item()), "")
                if isinstance(result.names, dict) else ""
            ),
            "confidence": float(result.boxes.conf[best_index].item()),
            "corners": corners.tolist(),
            "rvec": pose.rvec.tolist(),
            "tvec": pose.tvec.tolist(),
            "distance_m": pose.distance_m,
            "euler_deg": list(pose.euler_deg),
            "reprojection_error_px": pose.reprojection_error_px,
            "filtered": filtered.tolist() if filtered is not None else None,
            "predicted": predicted.tolist() if predicted is not None else None,
            "future": future.tolist() if future is not None else None,
            "velocity": velocity.tolist() if velocity is not None else None,
        }
        if log_file is not None:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
    return vis, record


def make_writer(path: str, shape, fps: float = 30.0):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (shape[1], shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video writer: {out}")
    return writer


def main() -> None:
    args = parse_args()
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None

    detector = ArmorDetector(args.model, imgsz=args.imgsz, conf=args.conf,
                             iou=args.iou, device=args.device)
    print(f"[model] {args.model} classes={detector.names}")

    calib = None
    kf = PoseKalmanFilter()
    frame_index = 0
    last_time = None
    writer = None
    start_time = time.perf_counter()

    try:
        if args.source == "image":
            frame = cv2.imread(args.image)
            if frame is None:
                raise SystemExit(f"cannot read image: {args.image}")
            calib = load_calibration(args.calib, frame.shape[1], frame.shape[0],
                                     args.armor_width, args.armor_height)
            print(f"[calib] {calib.source} fx={calib.fx:.1f} fy={calib.fy:.1f} "
                  f"armor={calib.armor_width_m:.3f}x{calib.armor_height_m:.3f}m")
            vis, record = process_frame(frame, 0, 1/30.0, detector, calib, kf,
                                        args, log_file)
            out = args.output if args.output.endswith((".jpg", ".png")) else "results/pose_image.jpg"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out, vis)
            print(f"[ok] saved {out}, record={'yes' if record else 'no'}")
            if not args.headless:
                cv2.imshow("armor pose", vis)
                cv2.waitKey(0)
            return

        if args.source == "video":
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"cannot open video: {args.video}")
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("cannot read first video frame")
            calib = load_calibration(args.calib, frame.shape[1], frame.shape[0],
                                     args.armor_width, args.armor_height)
            print(f"[calib] {calib.source} fx={calib.fx:.1f} fy={calib.fy:.1f}")
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
            while True:
                if frame_index > 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                vis, _ = process_frame(frame, frame_index, 1.0/fps_in,
                                       detector, calib, kf, args, log_file)
                if writer is None:
                    writer = make_writer(args.output, vis.shape, fps_in)
                writer.write(vis)
                if not args.headless:
                    cv2.imshow("armor pose", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if args.max_frames > 0 and frame_index + 1 >= args.max_frames:
                    break
                frame_index += 1
            cap.release()
            print(f"[ok] saved {args.output}")
            return

        # camera
        from camera.mvs_camera import MvsCamera
        with MvsCamera(args.camera) as cam:
            cam.set_pixel_format(args.format)
            cam.set_exposure_us(args.exposure)
            cam.set_gain_db(args.gain)
            cam.start()
            frame_data = cam.read()
            frame = frame_data.bgr
            calib = load_calibration(args.calib, frame.shape[1], frame.shape[0],
                                     args.armor_width, args.armor_height)
            print(f"[camera] {cam.info}")
            print(f"[calib] {calib.source} fx={calib.fx:.1f} fy={calib.fy:.1f} "
                  f"armor={calib.armor_width_m:.3f}x{calib.armor_height_m:.3f}m")
            while True:
                now = time.perf_counter()
                dt = 1/30.0 if last_time is None else (now - last_time)
                last_time = now
                vis, _ = process_frame(frame, frame_index, dt,
                                       detector, calib, kf, args, log_file)
                if writer is None:
                    writer = make_writer(args.output, vis.shape, 30.0)
                writer.write(vis)
                if not args.headless:
                    cv2.imshow("armor pose", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                frame_index += 1
                elapsed = time.perf_counter() - start_time
                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
                if args.seconds > 0 and elapsed >= args.seconds:
                    break
                frame_data = cam.read()
                frame = frame_data.bgr
            print(f"[ok] saved {args.output} ({frame_index} frames)")
    finally:
        if writer is not None:
            writer.release()
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
