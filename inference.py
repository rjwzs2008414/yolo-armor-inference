from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 装甲板检测/关键点推理与性能测试")
    parser.add_argument("--model", default="models/best.pt", help="模型路径：.pt / .onnx / .torchscript")
    parser.add_argument("--source", default="data/test.mp4", help="测试图片或视频")
    parser.add_argument("--output", default="results/pt/result.mp4", help="结果图片或视频路径")
    parser.add_argument("--json", default=None, help="性能结果 JSON；默认与 output 同目录")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="cpu 或 0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示处理整个视频")
    return parser.parse_args()


def gpu_sync(device: str) -> None:
    if str(device).lower() != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def draw_result(frame: np.ndarray, result, names) -> int:
    """在帧上画检测框、类别、置信度和关键点，返回目标数量。"""
    count = 0
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            count += 1

    kpts = result.keypoints
    if kpts is not None and kpts.xy is not None:
        for person_kpts in kpts.xy:
            for point in person_kpts:
                px, py = int(point[0]), int(point[1])
                if px == 0 and py == 0:
                    continue
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)

    return count


def benchmark(model, frame: np.ndarray, args: argparse.Namespace):
    kwargs = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    for _ in range(max(0, args.warmup)):
        model.predict(frame, **kwargs)

    times = []
    last_result = None
    for _ in range(max(1, args.repeat)):
        gpu_sync(args.device)
        started = time.perf_counter()
        last_result = model.predict(frame, **kwargs)[0]
        gpu_sync(args.device)
        times.append((time.perf_counter() - started) * 1000)

    avg_ms = sum(times) / len(times)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return times, avg_ms, fps, last_result


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise SystemExit(f"模型不存在：{model_path}")
    if not source.exists():
        raise SystemExit(f"测试数据不存在：{source}")

    if str(args.device).lower() != "cpu" and not torch.cuda.is_available():
        print("[warn] 当前 PyTorch 不支持 CUDA，自动改用 CPU。")
        args.device = "cpu"

    model = YOLO(str(model_path))

    is_video = source.suffix.lower() in VIDEO_EXTS
    if is_video:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"无法打开视频：{source}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, first_frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("视频第一帧读取失败。")
        source_type = "video"
    else:
        first_frame = cv2.imread(str(source))
        if first_frame is None:
            raise SystemExit(f"图片读取失败：{source}")
        frame_count = 1
        fps_in = None
        height, width = first_frame.shape[:2]
        source_type = "image"

    print(f"[model] {model_path}")
    print(f"[source] {source} ({source_type}, {width}x{height})")
    print(f"[bench] warmup={args.warmup}, repeat={args.repeat}, device={args.device}")

    times, avg_ms, fps, bench_result = benchmark(model, first_frame, args)
    first_count = 0 if bench_result.boxes is None else len(bench_result.boxes)

    total_detections = 0
    frames_processed = 0
    max_detections = 0

    if is_video:
        cap = cv2.VideoCapture(str(source))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps_in, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"无法创建输出视频：{output}")

        limit = args.max_frames if args.max_frames > 0 else frame_count
        while frames_processed < limit:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            count = draw_result(frame, result, model.names)
            total_detections += count
            max_detections = max(max_detections, count)
            writer.write(frame)
            frames_processed += 1
            if frames_processed % 30 == 0:
                print(f"  已处理 {frames_processed}/{limit} 帧")

        cap.release()
        writer.release()
    else:
        result = model.predict(
            first_frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        count = draw_result(first_frame, result, model.names)
        total_detections = count
        max_detections = count
        frames_processed = 1
        cv2.imwrite(str(output), first_frame)

    avg_detections = total_detections / frames_processed if frames_processed else 0.0

    try:
        import onnxruntime

        onnxruntime_version = onnxruntime.__version__
    except Exception:
        onnxruntime_version = None

    payload = {
        "model": str(model_path),
        "format": model_path.suffix.lower().lstrip("."),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "source": str(source),
        "source_type": source_type,
        "width": width,
        "height": height,
        "video_fps": round(fps_in, 3) if fps_in else None,
        "frames_processed": frames_processed,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "times_ms": [round(t, 3) for t in times],
        "avg_ms": round(avg_ms, 3),
        "fps": round(fps, 2),
        "first_frame_detection_count": first_count,
        "total_detections": total_detections,
        "avg_detections_per_frame": round(avg_detections, 3),
        "max_detections_per_frame": max_detections,
        "output": str(output),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics": getattr(ultralytics, "__version__", "unknown"),
            "onnxruntime": onnxruntime_version,
        },
    }

    json_path = Path(args.json).resolve() if args.json else output.parent / "benchmark.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] 平均推理耗时：{avg_ms:.2f} ms，FPS：{fps:.2f}")
    print(f"[ok] 首帧目标数：{first_count}，平均每帧：{avg_detections:.2f}")
    print(f"[ok] 结果：{output}")
    print(f"[ok] 数据：{json_path}")


if __name__ == "__main__":
    main()
