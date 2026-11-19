# 同济导航课 Lecture 4 小作业：QoS Debugger 解答

> 这周的 `slides-ros2-basic/` 是 ROS 2 基础课幻灯片，属于讲课资料；  
> `homework/` 才是作业。作业来源是“同济导航课 Lecture 4”，主题是 **ROS 2 QoS 调试**，不是同济的某个项目。

## 0. 环境与编译

原作业写的是 ROS 2 Humble，本机验证环境是 ROS 2 Lyrical（Ubuntu 26.04）。

```bash
cd ~/projects/homework_ws
source /opt/ros/lyrical/setup.bash
colcon build --symlink-install
source install/setup.bash
```

如果使用原始目录：

```bash
cd /mnt/e/机甲视觉学习/homework
source /opt/ros/lyrical/setup.bash
colcon build
source install/setup.bash
```

> 注：ROS 2 Lyrical 里 `ament_target_dependencies` 不可用，本解答已改成现代写法  
> `target_link_libraries(... rclcpp::rclcpp ...)`，并补了 `rcl_interfaces` 依赖。

## 1. 任务一：pub 和 sub 收不到消息

### 原因

发布者默认：

```text
reliability = BEST_EFFORT
```

订阅者默认：

```text
reliability = RELIABLE
```

QoS 兼容规则：

| 发布者 | 订阅者 | 能连上 |
|---|---|---|
| RELIABLE | RELIABLE | 是 |
| RELIABLE | BEST_EFFORT | 是 |
| BEST_EFFORT | BEST_EFFORT | 是 |
| BEST_EFFORT | RELIABLE | **否** |

订阅者的要求不能高于发布者，所以两者无法建立连接，订阅回调一次都不会进。

### 修复

`qos_debugger_pub.cpp`：

```cpp
// 原来
this->declare_parameter("reliability", "best_effort");

// 改为
this->declare_parameter("reliability", "reliable");
```

这样发布者提供 `RELIABLE`，默认订阅者也是 `RELIABLE`，可以通信。

### 复现与验证

故意让发布者用 `BEST_EFFORT`：

```bash
ros2 run qos_debugger qos_debugger_pub --ros-args -p reliability:=best_effort
```

订阅端会打印：

```text
[WARN] New publisher discovered on topic '/sensor_data', offering incompatible QoS.
No messages will be sent to it. Last incompatible policy: RELIABILITY_QOS_POLICY
```

### 用到的命令

```bash
# 查看话题的 QoS
ros2 topic info -v /sensor_data

# 查看节点参数
ros2 param list /sensor_subscriber
ros2 param get /sensor_subscriber reliability
ros2 param get /sensor_subscriber depth
```

## 2. 任务二：为什么收到的消息会丢包？

### 原因

发布者：

```text
rate = 100 Hz   → 每 10 ms 一条
```

订阅者回调：

```cpp
std::this_thread::sleep_for(std::chrono::milliseconds(30));
```

单线程执行器下，一个回调要 30 ms，最多处理约：

```text
1000 / 30 ≈ 33 Hz
```

生产 100 Hz、消费 33 Hz，队列只保留 10 条，约 100 ms 就填满，后面的消息被丢弃。

另外原代码有一个计数 bug：

```cpp
const uint32_t lost = msg->seq - expected_seq_;
// 只打印，没有 lost_count_ += lost;
```

所以 `report()` 里的丢包率永远是 0。

### 修复

`qos_debugger_sub.cpp`：

```cpp
// 1) 回调不再人为 sleep 30ms
this->declare_parameter("callback_delay_ms", 5);

// 2) 队列留一点余量，吸收短时突发
this->declare_parameter("depth", 100);

// 3) 真正累计丢包
lost_count_ += lost;
```

### 重要理解

- 把 `depth` 调大只能吸收**短时突发**；
- 如果生产速度长期高于消费速度，队列迟早还会满；
- 彻底解决要么提高消费速度（本作业默认 5 ms），要么降低发布频率，要么按传感器数据策略主动丢旧帧。

### 复现丢包

```bash
# 订阅端故意慢下来
ros2 run qos_debugger qos_debugger_sub --ros-args \
  -p callback_delay_ms:=30 -p depth:=10

# 另一个终端
ros2 run qos_debugger qos_debugger_pub
```

本机实测输出：

```text
检测到丢包: 期望 seq=478, 实际 seq=480, 丢失 2 条
累计: 收到 162 条, 丢失 319 条, 丢包率 66.32%
帧率: 32.93 Hz（本周期收到 33 条）
```

### 修复后

默认参数运行：

```text
Subscriber on /sensor_data: reliability=reliable, depth=100
累计: 收到 485 条, 丢失 0 条, 丢包率 0.00%
帧率: 99.98 Hz（本周期收到 100 条）
```

## 3. 任务三：计算并打印帧率

在 `report()` 定时器回调中，用两次报告之间的收到条数差除以时间差：

```cpp
const auto now = std::chrono::steady_clock::now();
const double elapsed =
    std::chrono::duration<double>(now - last_report_time_).count();
const uint32_t delta = received_count_ - last_received_count_;
const double fps = (elapsed > 0.0) ? static_cast<double>(delta) / elapsed : 0.0;

RCLCPP_INFO(
    this->get_logger(),
    "帧率: %.2f Hz（本周期收到 %u 条）",
    fps, delta);

last_received_count_ = received_count_;
last_report_time_ = now;
```

成员变量已经在原代码里预留：

```cpp
uint32_t last_received_count_{0};
std::chrono::steady_clock::time_point last_report_time_{std::chrono::steady_clock::now()};
```

## 4. 一键运行

终端 1：

```bash
source /opt/ros/lyrical/setup.bash
source ~/projects/homework_ws/install/setup.bash
ros2 run qos_debugger qos_debugger_sub
```

终端 2：

```bash
source /opt/ros/lyrical/setup.bash
source ~/projects/homework_ws/install/setup.bash
ros2 run qos_debugger qos_debugger_pub
```

## 5. 本作业涉及的命令总结

```bash
ros2 -h
ros2 topic list
ros2 topic info -v /sensor_data
ros2 topic hz /sensor_data
ros2 node list
ros2 param list /sensor_subscriber
ros2 param get /sensor_subscriber depth
ros2 param set /sensor_subscriber callback_delay_ms 30
ros2 run qos_debugger qos_debugger_pub
ros2 run qos_debugger qos_debugger_sub
```

## 6. 修改文件

```text
src/qos_debugger/src/qos_debugger_pub.cpp   发布者默认 RELIABLE
src/qos_debugger/src/qos_debugger_sub.cpp   默认 5ms/100 深度、累计丢包、帧率
src/qos_debugger/CMakeLists.txt             Lyrical 的 target_link_libraries 写法
src/qos_debugger/package.xml                增加 rcl_interfaces 依赖
```
