"""YOLO armor detector + drawing helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Brighten a dark image for display and for the detector.

    gamma < 1 lifts shadows (e.g. 0.35); gamma = 1 returns the input.
    """
    if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
        return frame
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)


def _unsharp(frame: np.ndarray, amount: float = 1.2, sigma: float = 3.0) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def _add_bloom(frame: np.ndarray, threshold: int = 180, ksize: int = 31,
               strength: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray >= threshold).astype(np.uint8) * 255
    glow = cv2.GaussianBlur(mask, (ksize, ksize), 0).astype(np.float32) / 255.0
    out = frame.astype(np.float32) + glow[..., None] * strength * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_for_detection(frame: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Compensate for a broken/unlit armor plate.

    Gamma lifts the dark scene, unsharp mask restores edges, and a small bloom
    around the white number/bars approximates the missing LED glow.
    """
    base = apply_gamma(frame, gamma)
    return _add_bloom(_unsharp(base, 1.2, 3.0), 180, 31, 1.0)


class ArmorDetector:
    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(self.model_path)
        self.names = self.model.names

    def infer(self, frame: np.ndarray):
        return self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        result = self.infer(frame)
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        names = self.names
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            cls_id = int(boxes.cls[i].item())
            name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": float(boxes.conf[i].item()),
                    "xyxy": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                }
            )
        return detections

    def draw(
        self,
        frame: np.ndarray,
        result,
        draw_center: bool = True,
        draw_keypoints: bool = True,
    ) -> np.ndarray:
        boxes = result.boxes
        names = self.names
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if draw_center:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"({cx},{cy})",
                        (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        if draw_keypoints:
            kpts = result.keypoints
            if kpts is not None and kpts.xy is not None:
                for person_kpts in kpts.xy:
                    for point in person_kpts:
                        px, py = int(point[0]), int(point[1])
                        if px == 0 and py == 0:
                            continue
                        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                        cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)
        return frame
