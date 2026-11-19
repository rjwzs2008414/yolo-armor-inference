# ROS 2 QoS Debugger 作业

同济导航课 Lecture 4 小作业：`Qos_debugger`。

对应课件：`slides-ros2-basic/`（ROS 2 基础课幻灯片，36 页）。  
本目录是作业代码和解答，不是课件本身。

## 包结构

```text
src/
  nav_hw_interfaces/     # 自定义消息 SensorData.msg
  qos_debugger/          # 发布者、订阅者、QoS 调试
```

## 编译

原作业环境是 ROS 2 Humble；本机在 ROS 2 Lyrical（Ubuntu 26.04）上验证通过。

```bash
source /opt/ros/lyrical/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 运行

终端 1：

```bash
ros2 run qos_debugger qos_debugger_sub
```

终端 2：

```bash
ros2 run qos_debugger qos_debugger_pub
```

预期：

```text
Subscriber on /sensor_data: reliability=reliable, depth=100
累计: 收到 ... 条, 丢失 0 条, 丢包率 0.00%
帧率: 约 100 Hz
```

## 三个任务

1. 修 QoS 兼容性：发布者默认 `best_effort` 改为 `reliable`。
2. 修丢包：回调默认 5ms、队列深度 100，并正确累计 `lost_count_`。
3. 在 `report()` 定时器里计算并打印帧率。

完整原因、命令和实测输出见：

```text
README_解答.md
```

## Lyrical 适配

ROS 2 Lyrical 不再提供作业原来的 `ament_target_dependencies` 宏，因此 CMake 改为：

```cmake
target_link_libraries(qos_debugger_pub
  rclcpp::rclcpp
  nav_hw_interfaces::nav_hw_interfaces
)
```

并补充 `rcl_interfaces` 依赖。
