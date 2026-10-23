from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = "<!-- COMPARISON_START -->"
END = "<!-- COMPARISON_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比原始 .pt 与导出模型的推理表现")
    parser.add_argument("--pt", default="models/best.pt")
    parser.add_argument("--onnx", default="models/model.onnx")
    parser.add_argument("--torchscript", default="models/model.torchscript")
    parser.add_argument("--source", default="data/sample.jpg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--skip-run", action="store_true", help="只生成表格，不重新推理")
    return parser.parse_args()


def run_one(label: str, model: str, slug: str, args: argparse.Namespace) -> dict:
    model_path = Path(model)
    if not model_path.is_file():
        raise SystemExit(f"{label} 模型不存在：{model_path}")

    output = ROOT / "results" / slug / "sample_pred.jpg"
    json_path = ROOT / "results" / slug / "benchmark.json"
    if not args.skip_run:
        cmd = [
            sys.executable,
            str(ROOT / "inference.py"),
            "--model",
            str(model_path),
            "--source",
            args.source,
            "--output",
            str(output),
            "--json",
            str(json_path),
            "--device",
            str(args.device),
            "--repeat",
            str(args.repeats),
        ]
        print(f"\n[{label}] " + " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    if not json_path.is_file():
        raise SystemExit(f"缺少 {json_path}，请先运行 inference.py 或去掉 --skip-run。")
    return json.loads(json_path.read_text(encoding="utf-8"))


def rel(path: str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def build_markdown(rows: list[dict], prefix: str = "") -> str:
    headers = ["对比项目"] + [r["label"] for r in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    def add(name: str, values: list[str]) -> None:
        lines.append("| " + name + " | " + " | ".join(values) + " |")

    add("模型格式", [f"`.{r['data']['format']}`" for r in rows])
    add("模型文件大小", [f"{r['data']['model_size_mb']:.2f} MB" for r in rows])
    add("平均推理耗时", [f"{r['data']['avg_ms']:.2f} ms" for r in rows])
    add("推理帧率", [f"{r['data']['fps']:.2f} FPS" for r in rows])
    add("首帧检测数量", [str(r["data"]["first_frame_detection_count"]) for r in rows])
    add("平均每帧检测数", [f"{r['data']['avg_detections_per_frame']:.2f}" for r in rows])
    add(
        "检测效果",
        [
            f"[查看结果图]({prefix}{rel(r['data']['output'])})"
            for r in rows
        ],
    )

    env = rows[0]["data"]["environment"]
    lines += [
        "",
        f"> 测试数据：`{Path(rows[0]['data']['source']).name}`；"
        f"imgsz={rows[0]['data']['imgsz']}，conf={rows[0]['data']['conf']}，"
        f"iou={rows[0]['data']['iou']}，device={rows[0]['data']['device']}，"
        f"warmup={rows[0]['data']['warmup']}，repeat={rows[0]['data']['repeat']}。",
        f"> 运行环境：{env['platform']}；CPU：{env['processor']}；"
        f"Python {env['python']}；PyTorch {env['torch']}；"
        f"Ultralytics {env['ultralytics']}；ONNX Runtime {env['onnxruntime'] or '未安装'}。",
        f"> GPU：{env['gpu'] or '本次未使用 GPU（对比统一使用 CPU，保证公平）'}。",
    ]

    fastest = min(rows, key=lambda r: r["data"]["avg_ms"])
    smallest = min(rows, key=lambda r: r["data"]["model_size_mb"])
    first_counts = {r["data"]["first_frame_detection_count"] for r in rows}
    avg_counts = {round(r["data"]["avg_detections_per_frame"], 2) for r in rows}

    lines += [
        "",
        "## 自动分析",
        f"- 推理最快：**{fastest['label']}**，平均 {fastest['data']['avg_ms']:.2f} ms，"
        f"约 {fastest['data']['fps']:.2f} FPS。",
        f"- 文件最小：**{smallest['label']}**，{smallest['data']['model_size_mb']:.2f} MB。",
        "- 首帧检测数量："
        + ("各格式一致。" if len(first_counts) == 1 else "各格式不一致，需要检查预处理/NMS/精度差异。"),
        "- 平均每帧检测数："
        + ("各格式基本一致。" if len(avg_counts) == 1 else "各格式存在差异，需要检查阈值或后处理。"),
    ]

    return "\n".join(lines)


def update_readme(table: str) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return
    before = text.split(START, 1)[0]
    after = text.split(END, 1)[1]
    readme.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [
        {"label": "原始 .pt", "data": run_one("原始 .pt", args.pt, "pt", args)},
        {"label": "ONNX", "data": run_one("ONNX", args.onnx, "onnx", args)},
    ]
    if args.torchscript:
        rows.append(
            {"label": "TorchScript", "data": run_one("TorchScript", args.torchscript, "torchscript", args)}
        )

    root_md = build_markdown(rows, prefix="")
    results_md = build_markdown(rows, prefix="../")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "comparison.md").write_text(root_md + "\n", encoding="utf-8")
    (ROOT / "results" / "comparison.md").write_text(results_md + "\n", encoding="utf-8")
    update_readme(root_md)

    print("\n[ok] 已生成 comparison.md")
    print("[ok] 已生成 results/comparison.md")
    print("[ok] 已把对比表写回 README.md")


if __name__ == "__main__":
    main()
