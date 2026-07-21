# G1 Body App — 开发说明

## 当前状态

**阶段 A: 云端文档摸底** (已完成)

**当前: 阶段 E — SDK 脚本验证** (03-08/13/14 已跑通，09/15 + channel 脚本待明天实机)

上一个会话（2026-06-16）完成了 RUN_ORDER 阶段一二 + PlayStream 流式通路确认。计划见 FEATURE.md。

## 环境准备

```bash
# 1. 克隆 Unitree SDK2 Python (gitignored, 需手动)
cd .moss_ws/apps/bodies/g1
mkdir -p src
git clone https://github.com/unitreerobotics/unitree_sdk2_python src/unitree_sdk2_python

# 2. 安装依赖 (包括 ghoshell-moss + unitree_sdk2py)
uv sync
```

SDK 不在版本控制中。`pyproject.toml` 通过 `{ path = "./src/unitree_sdk2_python", editable = true }` 引用。

**macOS 注意**: 不需要在 macOS 上编译 SDK。开发流程：macOS 上读源码 + 文档做规划，G1 PC2 (Linux) 上实装验证。`docs/` 和 `scripts/` 通过 git 同步。

## 目录结构

```
.moss_ws/apps/bodies/g1/
├── APP.md                  # 应用说明 (面向使用者/模型)
├── README.md               # 本文件 — 开发计划与设计决策
├── pyproject.toml          # 独立 venv 依赖声明
├── main.py                 # 入口: 进程启动 → 构造 channel → Matrix 注册
├── .gitignore              # 排除 src/, .venv/, uv.lock
├── docs/                   # 技术文档 (活的，随代码更新)
│   ├── index.md            # 云端文档 URL 映射 + 关键概念索引
│   ├── sdk-api.md          # SDK 接口分析 (loco/arm/audio)
│   ├── comms.md            # DDS 通讯模型
│   ├── hardware.md         # 硬件连接、网络拓扑、PC2 环境
│   ├── safety.md           # 安全控制方式 (急停、限位、力控)
│   ├── moss-on-pc2.md      # MOSS 装机过程与问题日志
│   └── channel-design.md   # Channel 体系设计 (脚本验证后产出)
├── scripts/                # 安全原子化验证脚本
│   └── ...                 # 每个脚本验证一个原子能力
├── src/                    # gitignored — 手动 clone SDK
│   └── unitree_sdk2_python/
└── runtime/                # 运行时日志 (gitignored)
```

## 开发哲学

### 安全优先，脚本先于 channel

在全尺寸人形机器人上，先理解安全机制再设计 channel 不是可选项。用独立 Python 脚本验证基线能力，人类反馈确认后，再从验证结果提炼 channel 设计。不要直接写 channel 然后猜测硬件行为。

### Channel 最简

App 进程 = 生命周期管理器。进程由 Circus 管理，挂了就重启。因此 channel 不需要 bootstrap/cleanup 生命周期 hook、不需要 factory 模式、不需要状态声明。

一个 channel 就是一个 Python 对象：构造里连硬件，方法暴露命令。G1 是最简 channel 的示范。

### macOS 规划，PC2 实装

macOS 上不需要编译 cyclonedds。所有文档和脚本在 macOS 上编写，通过 git 同步到 G1 PC2 执行。

### 技术文档与博客分离

- `docs/` — 技术文档，活的，随代码迭代更新
- `.ai_partners/blogs/posts/` — 博客，时间点快照，写决策的 why

## 开发阶段

### 阶段 A: 云端文档摸底
**产出**: 填充 `docs/index.md` + 创建 `docs/sdk-api.md` + `docs/comms.md`
**性质**: 纯阅读，不写代码
**目标**: 理解 G1 的技术全貌 — 自由度配置、传感器、控制模式、SDK 架构、DDS 通讯模型
**方法**:
1. 遍历 Unitree 官方文档站 (https://support.unitree.com/home/zh/G1_developer) 的关键页面
2. 在 `docs/index.md` 中建立 URL → 本地摘要的映射表
3. 从文档中提取 API surface，写入 `docs/sdk-api.md`
4. 理解 DDS topic 发现、QoS、RPC/service 调用方式，写入 `docs/comms.md`
**备选**: 如果文档站是纯 SPA（WebFetch 抓不到），直接以 SDK 源码 + examples 为主要信息源

### 阶段 B: 代码仓库摸底
**产出**: 补充 `docs/sdk-api.md`，记录实际 API surface
**内容**: 读 `src/unitree_sdk2_python/` 源码，重点：
- `example/` 目录 — 最直接的 API 用法参考
- loco client 相关模块 — 运动控制接口
- arm client 相关模块 — 手臂操作接口
- audio client 相关模块 — 音频播放接口
- DDS IDL 定义（如有）— 数据类型和 topic 结构

### 阶段 C: 硬件环境记录
**产出**: `docs/hardware.md`
**内容**:
- G1 机器人与 PC2 的网络拓扑
- PC2 规格（OS、Python 版本、网卡配置）
- DDS 网卡选择与 IP 配置
- 机器人连接方式（以太网/WiFi、IP 地址）
- 不涉及帐号信息
**目标**: 一份可复现的环境准备流程

### 阶段 D: MOSS 装机
**产出**: `docs/moss-on-pc2.md`
**内容**: MOSS 安装到 G1 PC2 的过程记录
- Python 环境准备
- 系统依赖安装（cyclonedds 等）
- `uv sync` 过程与问题
- 网络权限配置
- 这是一份"问题日志"——每个异常都值得记录

### 阶段 E: 基线实验
**产出**: `scripts/` 下的安全原子化验证脚本
**原则**:
- 独立 Python 脚本，直接跑在 PC2 上，不经过 MOSS channel 体系
- 每个脚本验证一个原子能力
- 验证点来源于 SDK examples + docs 分析结果
- 人类执行脚本并反馈验证结果
- 脚本命名: `01_<topic>.py`, `02_<topic>.py` ...
**验证闭环**: 脚本 → 执行 → 人类观察 → 反馈记录 → 调整理解

### 阶段 F: 安全理解
**产出**: `docs/safety.md`
**内容**: 必须在 channel 设计之前完成
- 急停机制（硬件急停 + 软件急停）
- 关节限位（角度/力矩限制）
- 力控限制
- 遥控器优先级（遥控器能否覆盖模型指令）
- 模式切换的安全约束
- 错误恢复流程

### 阶段 G: Channel 设计
**产出**: `docs/channel-design.md`
**前置**: 阶段 E 验证结果 + 阶段 F 安全理解
**内容**:
- 基于验证结果，确定哪些能力暴露为 CTML 命令
- 遵循最简原则：构造连硬件，方法 = 命令
- 命令签名设计（Code as Prompt）
- 多级模式概念与切换逻辑
- 安全边界：哪些命令在哪些模式下可用

### 阶段 H: 多级模式迭代
**产出**: Channel 实现 + 模式体系
**模式渐进**:
1. **debug 模式** — 最小权限，只读状态，不做任何运动
2. **sit 模式** — 机器人保持坐姿，可读状态 + 音频交互，不做运动
3. **遥控交互模式** — 遥控器控制行动，模型可读状态 + 音频交互，不控制运动
4. **模型控制急停模式** — 模型可控制运动，但人类保留急停权
5. **多模式切换** — 以上模式间的安全切换
**约束**: 全部以 G1 基线能力为验证对象，不做高阶开发（步态规划、全身协调等）

## Reachy Mini 经验

从上一个机器人集成中携带的关键经验：

| 经验 | G1 应对 |
|------|---------|
| 硬件连接延迟到 bootstrap | 不需要 — app 进程即生命周期 |
| Channel 过度复杂 | 最简原则: 构造 + 方法 |
| 依赖体系分层 | 已做 — app 独立 venv |
| Matrix 错误传播 | 连接失败明确退出，不静默降级 |
| 音频不同步(Reachy 特有问题) | 关注 DDS 的实时性保证 |

## 参考

- [Unitree G1 FEATURE.md](../../../.ai_partners/features/workstreams/2026/06/unitree-g1-integration/FEATURE.md)
- [Reachy Mini Integration FEATURE.md](../../../.ai_partners/features/workstreams/2026/05/reachymini-integration/FEATURE.md)
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [Unitree G1 开发者文档](https://support.unitree.com/home/zh/G1_developer/about_G1)
- [AI Partner Blog](../../../.ai_partners/blogs/)
