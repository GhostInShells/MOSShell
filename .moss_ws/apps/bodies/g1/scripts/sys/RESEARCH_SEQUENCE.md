# G1 系统线调研时序

AI 协作者引导人类工程师逐项执行系统线实验的对话脚本。每个步骤包含：执行命令 → 观察目标 → 决策影响。

第一轮验证后，这些脚本将改造为 MOSS channel 命令，AI 可在会话中直接调用。

---

## 步骤 0: 网络准备

**背景**: PC2 被隔离在交换机后，WiFi 默认关闭。如果这是断电重启后的首次连接，需要用以太网路径进入 PC2 再开启 WiFi。

**执行**: 人类自行用以太网连接 PC2（参考 `docs/hardware.md` 中的静态 IP 配置）。

**检查**: `ssh moss@<pc2-wifi-ip>` 是否可达？不可达则用以太网 IP 进入，跑 `network/03_wifi_status.sh` 检查 WiFi 状态。

**决策**: WiFi 不通 → 后续所有步骤阻塞。先修复网络。

---

## 步骤 1: 系统身份确认

**命令**: `bash system/01_jetson_info.sh`

**为什么**: 确认我们操作的是正确的机器（Jetson Orin NX），L4T 版本与文档记录一致。

**观察**: 
- 输出中的 R 版本号（如 R35.x）— 决定了后续依赖包兼容性
- 内核版本 — cyclonedds 需要特定内核模块

**决策**: 如果型号或版本与 `docs/hardware.md` 记录不一致 → 更新文档。

---

## 步骤 2: 基线性能快照

**命令**: `bash performance/05_idle_baseline.sh`

**为什么**: 在 MOSS 启动前记录 PC2 的空载状态。后续所有性能对比以此为准。没有基线的性能数据无法判断"正常还是异常"。

**观察**:
- CPU idle 百分比
- 可用内存
- Python 进程数（应为零或仅系统进程）

**决策**: 这只是记录，不做判断。保存输出，留待步骤 9 对比。

---

## 步骤 3: 网络拓扑验证

**命令**:
1. `bash network/01_interfaces.sh`
2. `bash network/02_reachability.sh`

**为什么**: 确认 PC2 的网络接口状态和三节点（PC1/LiDAR/外网）连通性。这决定了 DDS 通讯路径是否完整。

**观察**:
- 哪个接口持有 192.168.123.x 地址？接口名是什么（eth0/enpXsY）？
- PC1 (192.168.123.161) 是否可达？— 不可达则 LocoClient RPC 无法工作
- LiDAR (192.168.123.120) 是否可达？— 不可达则条件反射层无数据源

**决策**: 
- PC1 不可达 → G1 可能未开机或交换机未通电。阻塞所有运动控制相关实验
- LiDAR 不可达 → 条件反射层方案暂缓
- 记录网卡接口名 → 配置 `CYCLONEDDS_URI` 时需要

---

## 步骤 4: Python 与 MOSS 环境

**命令**:
1. `bash system/03_python_env.sh`
2. `bash system/04_moss_check.sh`

**为什么**: 这是阶段 D（MOSS 装机）的验收检查点。确认 `uv sync` 完成、venv 正确、moss 命令可用。

**观察**:
- `moss --ai start` 输出是否正常？
- `moss --ai all-commands` 是否返回命令树？
- `VIRTUAL_ENV` 是否指向正确的 `.venv/`？
- cyclonedds 包是否在 pip list 中？

**决策**:
- MOSS 启动失败 → 需重跑 `uv sync --active --all-extras`
- cyclonedds 未安装 → SDK 线所有 DDS 相关实验阻塞

---

## 步骤 5: DDS 环境

**命令**: `bash dds/01_env_check.sh`

**为什么**: DDS 是 G1 内外部通讯的统一总线。cyclonedds 安装、环境变量、网卡配置三者任一不对，DDS 发现和通讯都会静默失败。

**观察**:
- `CYCLONEDDS_URI` 是否设置？是否指定了正确的网卡？
- 共享内存配置是否正确（外部开发应为 `false`）？

**决策**:
- 环境变量缺失 → 配置 `CYCLONEDDS_URI`，网卡为步骤 3 确认的 192.168.123.x 接口
- 共享内存配置错误 → 修正配置文件

---

## 步骤 6: 音频硬件摸底

**命令**:
1. `bash audio/01_list_devices.sh`
2. `bash audio/02_bluetooth_hw.sh`

**为什么**: 这是 G1 音频架构的关键决策点。PC2 是否能独立发声和录音，决定了音频链路走 PC2 本地还是 PC1 API。

**观察**:
- aplay -l 有输出吗？— 有声卡 = PC2 可独立播放
- arecord -l 有输出吗？— 有录音设备 = PC2 可本地录音
- hciconfig 有输出吗？— 有蓝牙适配器 = 可连蓝牙耳机
- rfkill 是否 blocked？— soft block 可解，hard block 不可解

**决策**:
- 有声卡且有播放设备 → MOSS 音频可直接走 PC2，绕过 PC1 AudioClient API 的状态不确定性（PlayStream 回调/TTS 耗时/Cancel）
- 有蓝牙适配器 → 下一步扫描设备
- 无声卡且无蓝牙 → 音频唯一路径是 PC1 VuiClient RPC。接受 API 约束
- 无录音设备 → 语音输入只能走 G1 四麦阵列（PC1 UDP 组播）

---

## 步骤 7: 蓝牙音频设备扫描

**前置**: 步骤 6 确认蓝牙适配器存在且未被 hard block

**命令**: `bash audio/03_bluetooth_scan.sh`

**为什么**: 确认周边是否有可用的蓝牙音频设备（耳机/音箱）。决定蓝牙音频方案的实操可行性。

**观察**: 是否发现了 Class=Audio/Headset 的设备？

**决策**:
- 发现音频设备 → 准备配对。后续可写配对+连接脚本
- 未发现 → 距离太远或周边无蓝牙设备。蓝牙方案暂不适用

**人类操作**: 扫描前将蓝牙耳机/音箱置于配对模式。

---

## 步骤 8: 实际发声测试

**命令**: `bash audio/04_speaker_test.sh`

**为什么**: 设备列表不等于能发声。内核模块、PulseAudio 路由、权限都可能导致静默失败。实际播放是唯一可靠的验证方式。

**观察**: 人类是否听到了测试音？

**决策**:
- 听到声音 → PC2 音频输出通路完整。这是最好的结果——MOSS 音频可直接走 PC2
- 听不到 → 即使设备列表中有声卡，通路某处有断点。暂时走 PC1 API 方案，后续可排查

---

## 步骤 9: USB 接口与摄像头

**命令**:
1. `bash usb_camera/01_list_devices.sh`
2. `bash usb_camera/02_camera_check.sh`

**为什么**: G1 没有内置面向外部的摄像头。视觉方案依赖外接。先看 PC2 暴露了哪些 USB 接口，再插入摄像头验证。

**观察**:
- lsusb 中有几个 Root Hub？USB 2.0 还是 3.0？
- 插入 USB 摄像头后，`/dev/video*` 是否出现？
- 支持的分辨率和帧率（v4l2-ctl --list-formats-ext）？

**决策**:
- 有 USB 3.0 接口 + v4l2 识别摄像头 → USB 摄像头直连 PC2 方案可行
- 无 USB 3.0 或 v4l2 不识别 → 需要网络摄像头（IP Camera）或外部视觉处理方案
- 支持的分辨率/帧率低 → 视觉处理可能需要降采样

**人类操作**: 准备一个 USB 摄像头，在步骤 9 执行前插入 PC2。

---

## 步骤 10: 资源概况

**命令**: `bash system/02_resources.sh`

**为什么**: 快速了解 PC2 的 CPU/内存/磁盘/温度。为步骤 11 的性能对比提供宏观参照。

**观察**: 
- 内存可用量 — 8GB Orin NX 中系统占用后通常剩 5-6GB
- 磁盘可用量 — 日志和存储的容量上限
- 温度 — Jetson 被动散热，高温会触发降频

**决策**: 这只是信息收集。与步骤 11 的 MOSS 运行时数据对比才有意义。

---

## 步骤 11: MOSS 运行时性能

**前置**: MOSS 已启动

**命令**:
1. `bash performance/01_process_tree.sh`
2. `bash performance/02_cpu_profile.sh`
3. `bash performance/03_memory_profile.sh`
4. `bash performance/04_disk_io.sh`

**为什么**: 了解 MOSS 在 Jetson 上的实际资源消耗。对比步骤 2 的基线，判断 MOSS 是否可在 PC2 上长期稳定运行。

**观察**:
- MOSS 进程数 — Matrix/Cell/Channel 各占多少进程？
- CPU 增量 — MOSS 吃掉了多少 CPU？是单核还是多核分布？
- 内存增量 — MOSS 的 RSS 占用。是否有内存泄漏迹象？
- IO 压力 — 日志写入是否频发？

**决策**:
- CPU 增量 < 20%、内存增量 < 1GB → PC2 有余量同时跑 MOSS + DDS + Ghost
- CPU 或内存紧张 → 考虑将 Ghost 推理分散到外部机器（Mac/服务器）
- 磁盘 IO 高 → 日志级别调整或写入限速

---

## 验证记录

| 步骤 | 日期 | 结论 | 备注 |
|------|------|------|------|
| 0 | | | |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |
| 11 | | | |
