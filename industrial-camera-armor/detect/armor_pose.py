"""Armor corner ordering, PnP pose estimation and constant-velocity Kalman filter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int
    armor_width_m: float
    armor_height_m: float
    source: str = "placeholder"

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


@dataclass
class ArmorPose:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    euler_deg: tuple[float, float, float]
    reprojection_error_px: float
    distance_m: float
    image_points: np.ndarray
    object_points: np.ndarray


def placeholder_calibration(width: int, height: int,
                            armor_width_m: float = 0.225,
                            armor_height_m: float = 0.055) -> CameraCalibration:
    """Approximate intrinsics for a 1440x1080 industrial camera.

    A placeholder makes the code runnable without a chessboard, but the
    absolute XYZ distances are NOT trustworthy until you calibrate.
    """
    fx = fy = 0.85 * float(width)
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return CameraCalibration(
        camera_matrix=K,
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
        image_width=int(width),
        image_height=int(height),
        armor_width_m=float(armor_width_m),
        armor_height_m=float(armor_height_m),
        source="placeholder",
    )


def load_calibration(path: str | Path,
                     frame_width: Optional[int] = None,
                     frame_height: Optional[int] = None,
                     armor_width_m: Optional[float] = None,
                     armor_height_m: Optional[float] = None) -> CameraCalibration:
    p = Path(path)
    width = int(frame_width or 1440)
    height = int(frame_height or 1080)
    aw = 0.225 if armor_width_m is None else float(armor_width_m)
    ah = 0.055 if armor_height_m is None else float(armor_height_m)

    if not p.is_file():
        print(f"[warn] calibration file not found: {p}; using placeholder intrinsics")
        return placeholder_calibration(width, height, aw, ah)

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    K_raw = data.get("camera_matrix")
    dist_raw = data.get("dist_coeffs", [0, 0, 0, 0, 0])
    armor = data.get("armor", {}) or {}
    try:
        K = np.array(K_raw, dtype=np.float64)
        dist = np.array(dist_raw, dtype=np.float64).reshape(-1, 1)
        if K.shape != (3, 3) or not np.isfinite(K).all() or abs(K[0, 0]) < 1e-9:
            raise ValueError("invalid camera_matrix")
    except Exception as exc:
        print(f"[warn] invalid calibration ({exc}); using placeholder intrinsics")
        return placeholder_calibration(width, height, aw, ah)

    if armor_width_m is None:
        aw = float(armor.get("width_m", aw))
    if armor_height_m is None:
        ah = float(armor.get("height_m", ah))

    calibrated = bool(data.get("calibrated", "rms" in data))
    source = str(p) if calibrated else f"{p} (PLACEHOLDER, run calibration)"
    if not calibrated:
        print("[warn] calibration file is a placeholder; XYZ/distance are approximate")
    return CameraCalibration(
        camera_matrix=K,
        dist_coeffs=dist,
        image_width=int(data.get("image_width", width)),
        image_height=int(data.get("image_height", height)),
        armor_width_m=aw,
        armor_height_m=ah,
        source=source,
    )


def object_points(width_m: float, height_m: float) -> np.ndarray:
    """Four coplanar corners in TL, TR, BR, BL order, centered at origin."""
    w = float(width_m) * 0.5
    h = float(height_m) * 0.5
    return np.array([
        [-w, -h, 0.0],
        [ w, -h, 0.0],
        [ w,  h, 0.0],
        [-w,  h, 0.0],
    ], dtype=np.float32)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL in image coordinates.

    The model may emit keypoints in a different order; this function is
    independent of the network's keypoint ordering.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4 or not np.isfinite(pts).all():
        raise ValueError("need 4 finite corner points")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0).astype(np.float32)

    # In image coordinates (y down), TL -> TR -> BR -> BL is clockwise and
    # the cross product of the first two edges is positive.
    v1 = ordered[1] - ordered[0]
    v2 = ordered[2] - ordered[1]
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    if cross < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def _euler_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Return yaw, pitch, roll in degrees (ZYX convention)."""
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch), float(roll)


def solve_pose(image_points: np.ndarray,
               calib: CameraCalibration,
               width_m: Optional[float] = None,
               height_m: Optional[float] = None,
               use_ippe: bool = True) -> Optional[ArmorPose]:
    """Solve planar PnP from ordered TL,TR,BR,BL corners."""
    try:
        corners = order_corners(image_points)
    except ValueError:
        return None

    obj = object_points(width_m or calib.armor_width_m,
                        height_m or calib.armor_height_m)
    flags = cv2.SOLVEPNP_IPPE if use_ippe else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(
        obj, corners, calib.camera_matrix, calib.dist_coeffs,
        flags=flags,
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        obj, rvec, tvec, calib.camera_matrix, calib.dist_coeffs
    )
    reproj = float(np.mean(np.linalg.norm(
        projected.reshape(-1, 2) - corners, axis=1
    )))
    tvec = tvec.reshape(3).astype(np.float64)
    return ArmorPose(
        rvec=rvec.reshape(3).astype(np.float64),
        tvec=tvec,
        rotation=R.astype(np.float64),
        euler_deg=_euler_from_rotation(R),
        reprojection_error_px=reproj,
        distance_m=float(np.linalg.norm(tvec)),
        image_points=corners,
        object_points=obj,
    )


class PoseKalmanFilter:
    """Constant-velocity Kalman filter: state [x,y,z,vx,vy,vz]."""

    def __init__(self, process_noise: float = 1e-3,
                 measurement_noise: float = 5e-3):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * float(measurement_noise)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.initialized = False
        self.last_time: Optional[float] = None

    def _set_dt(self, dt: float) -> None:
        dt = float(np.clip(dt, 1e-4, 1.0))
        T = np.eye(6, dtype=np.float32)
        T[0, 3] = dt
        T[1, 4] = dt
        T[2, 5] = dt
        self.kf.transitionMatrix = T

    def update(self, measurement: np.ndarray, dt: float):
        """Return (predicted, filtered, velocity, future)."""
        z = np.asarray(measurement, dtype=np.float32).reshape(3, 1)
        if not self.initialized:
            self.kf.statePost = np.array([
                z[0, 0], z[1, 0], z[2, 0], 0, 0, 0
            ], dtype=np.float32).reshape(6, 1)
            self.kf.statePre = self.kf.statePost.copy()
            self.initialized = True
            self.last_time = None
            return (z.ravel().copy(), z.ravel().copy(),
                    np.zeros(3, dtype=np.float32),
                    z.ravel().copy())

        self._set_dt(dt)
        predicted = self.kf.predict().reshape(6).copy()
        filtered = self.kf.correct(z).reshape(6).copy()
        velocity = filtered[3:6].copy()
        future = filtered[0:3] + velocity * 0.2
        return predicted[0:3], filtered[0:3], velocity, future

    def predict_only(self, dt: float):
        if not self.initialized:
            return None
        self._set_dt(dt)
        predicted = self.kf.predict().reshape(6).copy()
        return predicted[0:3], predicted[3:6]

    def predict_ahead(self, horizon_s: float) -> Optional[np.ndarray]:
        if not self.initialized:
            return None
        state = self.kf.statePost.reshape(6)
        return state[0:3] + state[3:6] * float(horizon_s)
