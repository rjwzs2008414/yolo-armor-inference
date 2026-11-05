# 工业相机装甲板实时识别

基于海康 MVS SDK + OpenCV + YOLO 的工业相机实时装甲板识别作业。

## 1. 实验环境

| 项目 | 值 |
|---|---|
| 系统 | Ubuntu 26.04.1 LTS（WSL2） |
| 相机 | Hikrobot MV-CA016-10UC |
| 相机序列号 | 00L28498186 |
| 相机固件 | V1.6.3 201222 551351 |
| 接口 | USB3 |
| MVS SDK | 4.9.0.3（`/opt/MVS`） |
| Python | 3.12.16（项目内 `.venv`） |
| OpenCV | 5.0.0 |
| PyTorch | 2.15.0+cpu |
| Ultralytics | 8.4.202 |
| ONNX Runtime | 1.31.1 |
| 模型 | `models/best.pt` / `models/model.onnx` |
| 类别 | B1、B2、B3、R1、R2、R3 |
| 分辨率 | 1440 × 1080 |

> 当前 WSL 没有可用 CUDA，推理使用 CPU。

## 2. 目录结构

```text
industrial-camera-armor/
├─ camera/mvs_camera.py          MVS 取流封装
├─ detect/yolo_detector.py       YOLO 加载、推理、画框、暗图增强
├─ experiments/
│  ├─ camera_experiments.py      曝光/增益/格式实验
│  ├─ error_handling.py          超时/错误格式与资源释放
│  └─ mat_memory_demo.py         OpenCV/numpy 内存演示
├─ tools/preview.py              浏览器实时预览
├─ tools/detect_image.py         单张图片检测
├─ main.py                       相机 + YOLO 实时识别
├─ models/                       best.pt / model.onnx
├─ results/                      实验图片、CSV、JSON、视频
├─ docs/实验记录.md              实验报告
├─ logs/                         运行日志
└─ run.sh                        统一入口（设置 MVS 环境变量）
```

## 3. 运行前准备

MVS SDK 默认安装在 `/opt/MVS`。`run.sh` 会自动设置：

```bash
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
export LD_LIBRARY_PATH=/opt/MVS/lib/64:$LD_LIBRARY_PATH
```

**运行程序时不要打开 MVS 客户端**，否则相机会被占用：

```text
MV_CC_OpenDevice failed: 0x80000203
```

关闭 MVS 客户端：

```bash
pkill -f '/opt/MVS/bin/MVS'
```

## 4. 实时预览与调焦

```bash
./run.sh tools/preview.py --exposure 30000 --gain 12 --enhance
```

程序持续写 `results/preview.jpg`，浏览器打开：

```text
http://<WSL-IP>:8000/results/preview.jpg
```

在 SSH 会话里可以这样起 HTTP 服务：

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

## 5. 实时 YOLO 识别

```bash
./run.sh main.py \
  --model models/best.pt \
  --exposure 30000 --gain 12 --format BayerRG8 \
  --enhance --imgsz 960 --conf 0.2 \
  --headless --seconds 15 \
  --output results/live_demo_enhance.mp4
```

| 参数 | 说明 |
|---|---|
| `--model` | `models/best.pt` 或 `models/model.onnx` |
| `--exposure` | 曝光时间，单位 μs |
| `--gain` | 增益，单位 dB |
| `--format` | `BayerRG8` / `BGR8` / `RGB8` |
| `--gamma` | gamma < 1 提亮暗图 |
| `--enhance` | 暗图增强：gamma + 锐化 + 高光 bloom |
| `--imgsz` | 推理输入尺寸 |
| `--conf` / `--iou` | 置信度和 NMS 阈值 |
| `--headless` | 不弹 GUI，直接写视频 |
| `--seconds` | 运行秒数 |
| `--max-frames` | 最大帧数 |

## 6. 曝光/增益/格式实验

```bash
# 曝光：500 / 1000 / 3000 us，增益 0 dB
./run.sh experiments/camera_experiments.py --mode exposure --label static --frames 60

# 增益：0 / 6 / 12 dB，曝光 1000 us
./run.sh experiments/camera_experiments.py --mode gain --label static --frames 60

# 格式：BayerRG8 / BGR8
./run.sh experiments/camera_experiments.py --mode format --frames 60
```

运动目标：

```bash
./run.sh experiments/camera_experiments.py --mode exposure --label moving --seconds 5
```

输出在 `results/experiments/`，包括 PNG、原始 `.npy`、CSV 和 JSON。

## 7. 异常处理与内存

```bash
./run.sh experiments/error_handling.py
./run.sh experiments/mat_memory_demo.py
```

异常日志在 `logs/error_*.log`。

## 8. 关键结果

### 8.1 图像格式

| 格式 | FPS | CPU 占用 | 转换耗时 |
|---|---:|---:|---:|
| BayerRG8（OpenCV 转 BGR） | 70.4 | 161.9% | 0.62 ms |
| BGR8（相机原生） | 70.4 | 46.9% | 0.61 ms |

BayerRG8 带宽小，但 CPU 转换开销大；BGR8 原生输出省 CPU。

### 8.2 装甲板识别

- 正常补偿组：`30 ms / 12 dB / enhance`，1 号装甲板检测置信度 **0.81**。
- 异常组：`500/1000/3000 μs` 下画面几乎全黑，YOLO 检测数 **0**。
- 演示视频：`results/live_demo_enhance.mp4`
- 识别截图：`results/detection_success.jpg`

### 8.3 设备限制

实验室装甲板 LED 灯条损坏，无法发光。因此：

- 作业要求的 500/1000/3000 μs 短曝光图像严重欠曝；
- 采用外部照明 + `30 ms` 长曝光 + `12 dB` 增益 + `enhance` 预处理作为补偿；
- 该补偿会引入噪声、拖影和额外耗时，报告中如实记录。

## 9. 提交清单

- [x] 相机取流封装
- [x] 曝光/增益/格式实验脚本与数据
- [x] 异常处理与资源释放日志
- [x] YOLO 实时识别、画框、置信度、中心点
- [x] 真实相机识别截图
- [x] 真实相机识别视频
- [x] 实验记录 `docs/实验记录.md`
- [ ] 按老师要求补充运动目标拖影对比图（需要人工移动装甲板）
