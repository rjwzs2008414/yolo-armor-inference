# 标定图片目录

把棋盘格标定图片放在这里（`.jpg` / `.png`）。

建议：

- 棋盘格：9×6 内角点（不是格子数）
- 方格边长：25 mm
- 拍摄 15～25 张，覆盖不同位置、不同角度、不同距离
- 棋盘格要完整出现在画面里，不要大面积反光/模糊
- 图片分辨率保持一致（建议 1440×1080）

然后运行：

```bash
cd ~/projects/industrial-camera-armor
./.venv/bin/python calibration/calibrate_camera.py \
  --images calibration/images \
  --cols 9 --rows 6 --square-size 0.025 \
  --output configs/camera.yaml
```

标定成功后，`configs/camera.yaml` 里的 `calibrated` 会变成 `true`，PnP 的 XYZ 和距离才具有物理意义。
