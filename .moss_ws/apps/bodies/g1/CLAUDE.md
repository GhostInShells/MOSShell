# G1 Body App

Unitree G1 人形机器人身体控制 App。MOSS 接入的第二个人形机器人平台。

**所属 workstream**: `unitree-g1-integration`（状态追踪见 FEATURE.md，范式真相在本文件）

## 首次进入

1. 先读本文件 — 理解方法论
2. 再读 `README.md` — 了解当前阶段和下一步行动
3. 如需回溯决策历史，读 FEATURE.md

## 必要知识（必须先加载）

这四份蓝图是理解 G1 开发的前提，不是可选项：

```bash
moss --ai codex blueprint channel_builder    # new_channel + decorator + 命令签名反射
moss --ai codex blueprint states_channel     # with_state 排他切换 / with_module 永久叠加
moss --ai codex blueprint matrix             # Cell/进程/通讯：provide_channel + scopes
moss --ai ctml read                          # CTML 语法：命令 / 流式参数 / 作用域 / Observe
```

## 方法论（范式真相）

以下决策不由代码自然表达，每个进入 G1 的实例必须内化：

### 安全先于设计，脚本先于 channel

全尺寸人形机器人上，先理解安全机制再设计 channel。用独立 Python 脚本（`scripts/`）验证原子能力，人类反馈确认后，从验证结果提炼 channel 设计。**不要直接写 channel 然后猜测硬件行为。**

验证闭环：AI 在 macOS 写脚本 → git 同步到 PC2 → 人类执行观察 → 反馈 → 调整理解。

### App 进程 = 生命周期管理器

channel 不需要 bootstrap/cleanup/factory。构造里直接连 DDS，连接失败抛异常 → 进程退出 → Circus 重启。这是对 Reachy Mini 经验的关键修正 —— `bootstrap()` 延迟连接是进程内嵌入模式的历史遗留，app 模式天然解耦。

### Channel 最简

一个 Python 对象：构造连硬件，方法暴露命令。没有 factory，没有 lifecycle hook，没有状态声明。G1 是最简 channel 的示范。

Channel 的核心价值只有两件事：Code as Prompt（Python 函数签名即模型看到的命令接口）和并发有序（同 channel 内顺序执行，跨 channel 并行）。其他机制是历史补丁，不是设计意图。

### macOS 规划，PC2 实装

macOS 上不需要编译 cyclonedds。`docs/` 和 `scripts/` 在 macOS 编写，git 同步到 G1 PC2 执行。SDK (`src/unitree_sdk2_python`) gitignored，手动 clone。

### 文档与博客分离

- `docs/` — 技术文档，活的，随代码迭代更新，保持"当前真相"
- `.ai_partners/blogs/posts/` — 博客，时间点快照，写决策的 why，写完不改

### 其他约定

- 目录命名 `g1` 而非 `unitree_g1`：Unitree 是厂商，G1 是型号。app 寻址 `bodies/g1`，避免绑定厂商
- SDK 不在版本控制中：开发者通过 README.md 的 `git clone` 命令手动获取
- APP.md 面向使用者/模型，README.md 面向开发者

## 目录

```
.moss_ws/apps/bodies/g1/
├── APP.md               # 面向使用者/模型：能力 + CTML 调用示例
├── README.md            # 面向开发者：当前阶段 + 开发计划（会话起点）
├── CLAUDE.md            # 本文件 — AI 认知入口（范式真相）
├── main.py              # 入口：构造 channel → Matrix 注册
├── docs/                # 技术文档（活的）
│   ├── index.md         #   云端文档 URL 映射 + 概念索引
│   ├── sdk-api.md       #   SDK 接口分析 (loco/arm/audio)
│   ├── comms.md         #   DDS 通讯模型
│   ├── hardware.md      #   硬件连接 + 网络拓扑 + PC2 环境
│   ├── safety.md        #   急停/限位/力控/遥控器优先级
│   ├── moss-on-pc2.md   #   MOSS 装机过程与问题日志
│   └── channel-design.md#   Channel 体系设计（阶段 E+F 后产出）
├── scripts/             # 安全原子化验证脚本，不经 MOSS channel
└── src/                 # gitignored — unitree_sdk2_python 手动 clone
```

## 开发阶段

八阶段渐进 (A-H)，详情见 `README.md`。核心约束：

- 安全理解 (F) 在 channel 设计 (G) 之前
- 基线实验 (E) 是所有代码工作的前提
- 阶段 H 的模式渐进：debug → sit → 遥控交互 → 模型控制急停 → 多模式切换

当前：**阶段 B — SDK 源码验证** (待 clone `src/unitree_sdk2_python`)。阶段 A 已完成。

## 已知问题

- **moss_static 强缓存 + 动态节点不进 instruction**：`main.py` 中通过 `channel.build.instruction()` 声明的进度说明可能被 static 缓存，导致模型看到的 instruction 与实际阶段不同步。需通过 MCP 实测验证缓存行为，探索修正方案。如果在 session 内 instruction 不刷新，则需要回归 CLAUDE.md 作为进度真相的单一来源。
