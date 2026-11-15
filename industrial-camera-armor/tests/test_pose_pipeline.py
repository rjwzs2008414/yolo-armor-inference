"""Offline tests for corner ordering, PnP and the Kalman filter."""
from __future__ import annotations

import numpy as np
import cv2

from detect.armor_pose import (
    PoseKalmanFilter,
    load_calibration,
    object_points,
    order_corners,
    solve_pose,
)


def test_order_corners() -> None:
    expected = np.array([
        [10, 10], [110, 10], [110, 60], [10, 60],
    ], dtype=np.float32)
    shuffled = expected[[2, 0, 3, 1]]
    got = order_corners(shuffled)
    assert np.allclose(got, expected), got


def test_pnp_roundtrip() -> None:
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    obj = object_points(0.225, 0.055)
    rvec_true = np.array([0.08, -0.04, 0.02], dtype=np.float64)
    tvec_true = np.array([0.02, -0.03, 0.85], dtype=np.float64)
    projected, _ = cv2.projectPoints(obj, rvec_true, tvec_true, K, dist)
    image_points = projected.reshape(-1, 2)

    class C:
        pass
    calib = C()
    calib.camera_matrix = K
    calib.dist_coeffs = dist
    calib.armor_width_m = 0.225
    calib.armor_height_m = 0.055
    pose = solve_pose(image_points, calib)
    assert pose is not None
    assert np.linalg.norm(pose.tvec - tvec_true) < 1e-4, pose.tvec
    assert abs(pose.distance_m - np.linalg.norm(tvec_true)) < 1e-4
    assert pose.reprojection_error_px < 1e-3


def test_kalman_constant_velocity() -> None:
    kf = PoseKalmanFilter(process_noise=1e-4, measurement_noise=1e-4)
    dt = 0.05
    last_filtered = None
    for i in range(30):
        z = np.array([0.02 + 0.1 * i * dt, -0.01, 0.8])
        pred, filtered, vel, _ = kf.update(z, dt)
        last_filtered = filtered
    assert last_filtered is not None
    # constant velocity should converge to vx ~ 0.1 m/s
    assert abs(vel[0] - 0.1) < 0.02, vel
    future = kf.predict_ahead(0.2)
    assert future[0] > last_filtered[0]


def test_kalman_smoothing() -> None:
    rng = np.random.default_rng(0)
    kf = PoseKalmanFilter(process_noise=1e-5, measurement_noise=2e-2)
    measurements = []
    filtered = []
    for _ in range(60):
        z = np.array([0.2, 0.1, 1.0]) + rng.normal(0, 0.01, 3)
        measurements.append(z)
        _, f, _, _ = kf.update(z, 1 / 30)
        filtered.append(f)
    measurements = np.array(measurements)
    filtered = np.array(filtered)
    assert filtered[10:].std(axis=0).mean() < measurements.std(axis=0).mean()


def test_calibration_fallback() -> None:
    calib = load_calibration("configs/does_not_exist.yaml", 1440, 1080)
    assert calib.camera_matrix.shape == (3, 3)
    assert calib.dist_coeffs.shape == (5, 1)
    assert calib.armor_width_m > 0
    assert calib.armor_height_m > 0


if __name__ == "__main__":
    test_order_corners()
    test_pnp_roundtrip()
    test_kalman_constant_velocity()
    test_kalman_smoothing()
    test_calibration_fallback()
    print("All pose pipeline tests passed.")
