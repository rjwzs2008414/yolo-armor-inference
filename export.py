from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

EXT_MAP = {"onnx": ".onnx", "torchscript": ".torchscript"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO .pt -> ONNX / TorchScript")
    parser.add_argument("--weights", default="models/best.pt", help="原始 .pt 权重")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "torchscript"],
        choices=sorted(EXT_MAP),
        help="要导出的格式",
    )
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        sys.exit(
            f"找不到权重文件：{weights}\n"
            "请先把上次作业的 .pt 放到 models/best.pt，"
            "或用 --weights 指定实际路径。"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[dict] = []

    for fmt in args.formats:
        print(f"\n[export] {fmt} <- {weights}")
        model = YOLO(str(weights))
        kwargs = {
            "format": fmt,
            "imgsz": args.imgsz,
            "device": args.device,
        }
        if fmt == "onnx":
            kwargs.update(
                opset=args.opset,
                simplify=not args.no_simplify,
                dynamic=args.dynamic,
            )

        try:
            try:
                exported = model.export(**kwargs)
            except Exception as exc:
                if fmt == "onnx" and kwargs.get("simplify"):
                    print(f"[warn] ONNX simplify 失败，改为不简化重试：{exc}")
                    kwargs["simplify"] = False
                    model = YOLO(str(weights))
                    exported = model.export(**kwargs)
                else:
                    raise
        except Exception as exc:
            print(f"[error] {fmt} 导出失败：{exc}")
            errors.append({"format": fmt, "error": str(exc)})
            continue

        src = Path(exported).resolve()
        if not src.exists():
            errors.append({"format": fmt, "error": f"导出返回路径不存在：{src}"})
            continue

        dst = (output_dir / f"model{EXT_MAP[fmt]}").resolve()
        if src != dst:
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

        size_mb = round(dst.stat().st_size / (1024 * 1024), 2)
        records.append({"format": fmt, "path": str(dst), "size_mb": size_mb})
        print(f"[ok] {dst} ({size_mb} MB)")

    manifest = output_dir / "export_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "weights": str(weights),
                "imgsz": args.imgsz,
                "device": args.device,
                "exports": records,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n清单：{manifest}")

    if not records:
        sys.exit("没有任何格式导出成功。")


if __name__ == "__main__":
    main()

