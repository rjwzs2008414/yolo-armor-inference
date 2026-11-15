"""Chessboard camera calibration with OpenCV.

Example:
  ./.venv/bin/python calibration/calibrate_camera.py \
      --images calibration/images \
      --cols 9 --rows 6 --square-size 0.025 \
      --output configs/camera.yaml

`--cols` and `--rows` are the number of INNER corners, not squares.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="calibration/images")
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square-size", type=float, default=0.025,
                   help="square side length in meters")
    p.add_argument("--output", default="configs/camera.yaml")
    p.add_argument("--show", action="store_true", help="show detected corners")
    return p.parse_args()


def find_corners(gray: np.ndarray, pattern: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        # Newer OpenCV has a more robust sector-based detector.
        try:
            ok, corners = cv2.findChessboardCornersSB(
                gray, pattern,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
        except Exception:
            ok = False
    if not ok or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate_from_images(image_dir: str | Path, cols: int, rows: int,
                          square_size: float, show: bool = False):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise RuntimeError(f"no calibration images in {image_dir}")

    pattern = (int(cols), int(rows))
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size)

    object_points = []
    image_points = []
    image_size = None
    used = 0
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[skip] cannot read {path}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        else:
            if image_size != (gray.shape[1], gray.shape[0]):
                print(f"[skip] size mismatch {path}: {gray.shape[1]}x{gray.shape[0]}")
                continue
        corners = find_corners(gray, pattern)
        if corners is None:
            print(f"[skip] chessboard not found: {path.name}")
            continue
        object_points.append(objp.copy())
        image_points.append(corners)
        used += 1
        print(f"[ok] {path.name}")
        if show:
            vis = cv2.drawChessboardCorners(frame.copy(), pattern, corners, True)
            cv2.imshow("chessboard", vis)
            cv2.waitKey(300)

    if show:
        cv2.destroyAllWindows()
    if used < 5:
        raise RuntimeError(f"need at least 5 valid images, got {used}")
    if image_size is None:
        raise RuntimeError("no valid images")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    # Mean reprojection error over all images.
    total_error = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        error = cv2.norm(
            img.reshape(-1, 2).astype(np.float32),
            projected.reshape(-1, 2).astype(np.float32),
            cv2.NORM_L2,
        )
        total_error += float(error)
        total_points += len(obj)
    mean_error = total_error / max(1, total_points)

    print(f"\nRMS: {rms:.4f} px")
    print(f"Mean reprojection error: {mean_error:.4f} px")
    print("camera_matrix:\n", K)
    print("dist_coeffs:\n", dist.ravel())
    return {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms": float(rms),
        "mean_reprojection_error_px": mean_error,
        "images_used": used,
        "chessboard": {"cols": cols, "rows": rows, "square_size_m": square_size},
        "calibrated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def save_calibration(data: dict, output: str | Path,
                     armor_width_m: float = 0.225,
                     armor_height_m: float = 0.055) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["calibrated"] = True
    data["armor"] = {
        "width_m": float(armor_width_m),
        "height_m": float(armor_height_m),
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"\n[ok] wrote {out}")


def main() -> None:
    args = parse_args()
    data = calibrate_from_images(args.images, args.cols, args.rows,
                                 args.square_size, args.show)
    save_calibration(data, args.output)


if __name__ == "__main__":
    main()
