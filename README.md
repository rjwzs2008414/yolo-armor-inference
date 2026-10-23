# YOLO 装甲板模型导出、推理与性能对比

## 1. 项目简介

在上一份 YOLOv26n 装甲板推理与关键点可视化作业的基础上，把 PyTorch 格式的 `best.pt` 导出为 ONNX 和 TorchScript，使用导出模型完成同一段视频/同一帧图片的推理，并与原始 `.pt` 做公平对比。

完成内容：`.pt -> .onnx/.torchscript` 导出；三种格式加载推理；检测框、类别、置信度、4 个关键点可视化；保存图片和视频；统计模型大小、平均推理耗时、FPS、检测数量；自动生成 `comparison.md`。

## 2. 模型与数据

| 项目 | 内容 |
|---|---|
| 模型 | YOLO26n-pose（Ultralytics） |
| 参数量 | 2,447,934 |
| 类别 | B1、B2、B3、R1、R2、R3 共 6 类 |
| 关键点 | 每个目标 4 个 |
| 原始权重 | `models/best.pt`（5.62 MB） |
| 测试视频 | `data/test.mp4`，1280x720，404 帧，30.15 FPS，约 13 秒 |
| 对比图片 | `data/sample.jpg`（test.mp4 第 100 帧） |
| 输入尺寸 | 640x640 |
| 导出格式 | ONNX、TorchScript |

## 3. 目录结构

```text
yolo-armor-inference/
├─ README.md
├─ requirements.txt
├─ export.py
├─ inference.py
├─ compare.py
├─ models/
│  ├─ best.pt
│  ├─ model.onnx
│  ├─ model.torchscript
│  └─ export_manifest.json
├─ data/
│  ├─ test.mp4
│  └─ sample.jpg
└─ results/
   ├─ pt/
   ├─ onnx/
   ├─ torchscript/
   └─ comparison.md
```

## 4. 运行环境

| 项目 | 内容 |
|---|---|
| 操作系统 | Windows 10 |
| Python | 3.11.5 |
| CPU | AMD Ryzen 9 8945HX with Radeon Graphics |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB |
| PyTorch | 2.14.1+cpu |
| Ultralytics | 8.4.177 |
| ONNX Runtime | 1.31.0 |
| OpenCV | 5.0.0 |

说明：本机虽然有 RTX 5060，但当前虚拟环境安装的是 CPU 版 PyTorch（2.14.1+cpu），所以本次 `.pt`、ONNX、TorchScript 三种格式统一使用 CPU 推理，保证对比公平。如果之后安装了 CUDA 版 PyTorch 和 `onnxruntime-gpu`，可以把三种格式都改成 `--device 0` 再做 GPU 对比，仍然要保持三种格式使用同一设备。

## 5. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果 PowerShell 禁止激活虚拟环境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 6. 模型导出

```powershell
python export.py --weights models/best.pt --formats onnx torchscript --device cpu --imgsz 640
```

| 格式 | 文件 | 大小 |
|---|---|---:|
| PyTorch | `models/best.pt` | 5.62 MB |
| ONNX | `models/model.onnx` | 9.72 MB |
| TorchScript | `models/model.torchscript` | 9.94 MB |

ONNX 使用 opset 12 并开启 simplify；如果 `onnxslim` 不可用，`export.py` 会自动回退到不简化导出。TorchScript 使用 Ultralytics 的 `format="torchscript"` 导出。

## 7. 模型推理

### 7.1 单张图片

```powershell
python inference.py --model models/best.pt --source data/sample.jpg --output results/pt/sample_pred.jpg --device cpu --warmup 3 --repeat 10
python inference.py --model models/model.onnx --source data/sample.jpg --output results/onnx/sample_pred.jpg --device cpu --warmup 3 --repeat 10
python inference.py --model models/model.torchscript --source data/sample.jpg --output results/torchscript/sample_pred.jpg --device cpu --warmup 3 --repeat 10
```

### 7.2 视频

```powershell
python inference.py --model models/best.pt --source data/test.mp4 --output results/pt/result.mp4 --device cpu --warmup 3 --repeat 10
python inference.py --model models/model.onnx --source data/test.mp4 --output results/onnx/result.mp4 --device cpu --warmup 3 --repeat 10
python inference.py --model models/model.torchscript --source data/test.mp4 --output results/torchscript/result.mp4 --device cpu --warmup 3 --repeat 10
```

结果视频：`results/pt/result.mp4`、`results/onnx/result.mp4`、`results/torchscript/result.mp4`。

### 7.3 一键生成对比表

```powershell
python compare.py --source data/sample.jpg --device cpu --repeats 10
```

该命令会依次运行三种模型，并自动更新 `comparison.md`、`results/comparison.md` 和本 README 中的对比表。

## 8. 性能对比

<!-- COMPARISON_START -->
| 对比项目 | 原始 .pt | ONNX | TorchScript |
|---|---|---|---|
| 模型格式 | `.pt` | `.onnx` | `.torchscript` |
| 模型文件大小 | 5.62 MB | 9.72 MB | 9.94 MB |
| 平均推理耗时 | 47.20 ms | 35.47 ms | 68.39 ms |
| 推理帧率 | 21.19 FPS | 28.20 FPS | 14.62 FPS |
| 首帧检测数量 | 2 | 2 | 2 |
| 平均每帧检测数 | 2.00 | 2.00 | 2.00 |
| 检测效果 | [查看结果图](results/pt/sample_pred.jpg) | [查看结果图](results/onnx/sample_pred.jpg) | [查看结果图](results/torchscript/sample_pred.jpg) |

> 测试数据：`sample.jpg`；imgsz=640，conf=0.5，iou=0.45，device=cpu，warmup=3，repeat=10。
> 运行环境：Windows-10-10.0.26300-SP0；CPU：AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD；Python 3.11.5；PyTorch 2.14.1+cpu；Ultralytics 8.4.177；ONNX Runtime 1.31.0。
> GPU：本次未使用 GPU（对比统一使用 CPU，保证公平）。

## 自动分析
- 推理最快：**ONNX**，平均 35.47 ms，约 28.20 FPS。
- 文件最小：**原始 .pt**，5.62 MB。
- 首帧检测数量：各格式一致。
- 平均每帧检测数：各格式基本一致。
<!-- COMPARISON_END -->

### 视频推理统计

| 模型 | 视频平均每帧检测数 | 视频结果 |
|---|---:|---|
| 原始 .pt | 1.32 | `results/pt/result.mp4` |
| ONNX | 1.31 | `results/onnx/result.mp4` |
| TorchScript | 1.31 | `results/torchscript/result.mp4` |

### 结果分析

1. ONNX 在 CPU 上最快：平均 35.47 ms，比原始 `.pt` 的 47.20 ms 快约 24.9%，FPS 从 21.19 提升到 28.20。
2. `.pt` 文件最小：5.62 MB。ONNX 和 TorchScript 分别为 9.72 MB、9.94 MB，因为导出文件除了权重还保存了计算图或序列化结构。
3. TorchScript 在本机 CPU 上反而最慢：68.39 ms。TorchScript 主要优势是脱离 Python 源码部署到 LibTorch/C++，并不保证在 CPU 上比原生 PyTorch 或 ONNX Runtime 更快；本次测量还包含 Ultralytics 的统一预处理和后处理封装。
4. 检测结果一致：三种格式在对比图片上都检测到 2 个目标，说明导出后的检测框和关键点结果基本一致。
5. 视频平均检测数略低于首帧：视频中存在目标出画、遮挡和角度变化，平均每帧约 1.31 到 1.32 个目标是合理的帧间差异，不代表模型导出错误。

## 9. 实验过程中遇到的问题与解决方法

1. 一开始找不到视频文件。原因：输入文件名把 `test` 误写成 `text`。解决：统一使用 `data/test.mp4`，并在 `inference.py` 中用 `Path` 检查文件是否存在。
2. ONNX 导出提示缺少 `onnxslim`。原因：Ultralytics 导出时自动做 ONNX 简化，需要额外依赖。解决：安装 `onnx`、`onnxruntime`、`onnxslim`；`export.py` 也保留了简化失败时自动回退的逻辑。
3. TorchScript 导出出现 `torch.jit.trace` 弃用警告。原因：新版 PyTorch 对 `torch.jit.trace` 给出 FutureWarning。解决：警告不影响导出和推理；本次 TorchScript 作为选做格式保留。
4. 有 RTX 5060 但推理仍然使用 CPU。原因：当前虚拟环境是 `torch 2.14.1+cpu`，没有 CUDA 支持。解决：本次统一用 CPU 做公平对比；如果要 GPU 对比，安装 CUDA 版 PyTorch 和 `onnxruntime-gpu`，三种格式都加 `--device 0`。
5. 视频结果文件较大。原因：OpenCV 的 `mp4v` 编码体积较大。解决：作业允许提交结果图片；本项目同时保留结果图片和视频，若 GitHub 空间紧张可只保留 `sample_pred.jpg` 和对比表。

## 10. 对不同模型格式的理解

- `.pt`：PyTorch 原生权重格式，保存模型参数和训练相关结构，继续训练、调试最方便，但部署时依赖 PyTorch/Python 环境。
- `.onnx`：开放的计算图交换格式，可被 ONNX Runtime、TensorRT、OpenVINO 等后端加载，跨框架部署方便；导出后计算图固定，预处理和后处理需要保持一致。
- `.torchscript`：PyTorch 的序列化部署格式，可由 LibTorch/C++ 加载，不依赖原始 Python 代码；跨框架能力不如 ONNX。
- `.engine`：TensorRT 引擎，NVIDIA GPU 上通常最快，但和 GPU 型号、CUDA/TensorRT 版本强绑定，通用性较差。
- OpenVINO：面向 Intel CPU/GPU/VPU 的部署格式，在 Intel 硬件上更有优势。

不同格式出现速度或检测数量差异，常见原因包括：NMS 实现不同、输入预处理和 letterbox 不同、fp32/fp16 精度不同、opset 版本不同、动态和静态 shape 不同，以及是否在同一硬件上测试。

## 11. GitHub 提交

本仓库已经关联远程地址：

```text
git@github.com:rjwzs2008414/yolo-armor-inference.git
```

日常提交：

```powershell
git status
git add .
git commit -m "feat: add YOLO export and performance comparison"
git push
```

模型文件较大时可以使用 Git LFS：

```powershell
git lfs install
git lfs track "models/*.pt" "models/*.onnx" "models/*.torchscript"
git add .gitattributes models
git commit -m "chore: track model weights with Git LFS"
git push
```

## 12. 提交前自查

- [x] `models/best.pt` 是上次作业的原始模型
- [x] `data/test.mp4` 是上次作业的同一段测试视频
- [x] ONNX 导出成功
- [x] TorchScript 导出成功
- [x] 三种格式都完成图片推理
- [x] 三种格式都完成视频推理
- [x] `results/` 中有带检测框和关键点的结果
- [x] `comparison.md` 包含模型大小、耗时、FPS、检测数量
- [x] README 包含运行命令、环境、结果和格式分析
- [x] 结果视频和模型文件已准备提交
