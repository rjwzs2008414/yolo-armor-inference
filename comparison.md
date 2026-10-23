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
