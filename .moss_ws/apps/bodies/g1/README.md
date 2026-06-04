# G1 Body App — 开发说明

## 当前状态

**Phase 0: 目录结构定义 + SDK 分析** (进行中)

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

## 迭代计划

### Phase 0 — 目录结构 + SDK 分析 (当前)

- [x] 创建 app stub (`bodies/g1`)
- [x] 创建 pyproject.toml (独立 venv)
- [x] 完成目录结构定义
- [ ] SDK 接口分析: G1 的 API surface (loco/arm/audio)
- [ ] SDK 通讯模型分析: DDS topic、RPC/service 调用方式
- [ ] macOS 编译验证: cyclonedds 能否在本地构建和 import

### Phase 1 — 最小可 import

目标: 在 app venv 中 `import unitree_sdk2py` 成功。

- [ ] 解决 cyclonedds 的 macOS 编译 (C 扩展 + 系统依赖)
- [ ] `uv sync` 在 app 目录下通过
- [ ] 验证基础 DDS 通讯 (helloworld publisher/subscriber)

### Phase 2 — G1 API 理解

目标: 理解 G1 的三组 API，确定 Channel 命令设计。

- [ ] Loco API: 运动控制接口 (站立、行走、步态切换)
- [ ] Arm API: 手臂操作接口 (5/7 自由度、动作规划)
- [ ] Audio API: 音频播放接口
- [ ] 确定哪些 API 适合暴露为 CTML 命令

### Phase 3 — Channel 实现

目标: 实现 `G1Channel`，按 Code as Prompt 原则暴露命令。

- [ ] 参考 reachymini contrib 模式，在 `ghoshell_moss_contrib/` 下实现 channel wrapper
- [ ] 运动控制命令 (stand, walk, stop...)
- [ ] 手臂控制命令 (move_arm, grip...)
- [ ] 音频命令 (say, play...)
- [ ] 状态感知命令 (joint_states, imu...)

### Phase 4 — App 集成验证

目标: app 通过 Matrix 注册，CTML 端到端验证。

- [ ] `main.py` 入口实现
- [ ] `moss apps test bodies/g1` 前台调试
- [ ] 通过 MCP + CTML 验证控制链路

### Phase 5 — 真机测试与加固

目标: 连接真实 G1 机器人，验证全链路。

- [ ] 网络配置 (DDS 网卡、机器人 IP)
- [ ] 真机运动控制验证
- [ ] 安全机制 (急停、限位、力控)
- [ ] 错误处理与恢复

## 目录结构

```
.moss_ws/apps/bodies/g1/
├── APP.md              # 应用说明 (面向使用者/模型)
├── README.md           # 本文件 — 开发计划与设计决策
├── CLAUDE.md           # AI 开发上下文 (后续创建)
├── pyproject.toml      # 独立 venv 依赖声明
├── main.py             # 入口: Matrix.discover().run(provide_channel)
├── .gitignore          # 排除 src/, runtime/, uv.lock, .venv/
├── src/                # gitignored — 手动 clone SDK
│   └── unitree_sdk2_python/   # git clone https://github.com/unitreerobotics/unitree_sdk2_python
├── runtime/            # 运行时日志 (gitignored)
└── .discuss/           # 深度设计讨论 (按需)
```

## 参考

- [Reachy Mini Integration FEATURE.md](../../../.ai_partners/features/workstreams/2026/05/reachymini-integration/FEATURE.md)
- [Reachy Mini Contrib FEATURE.md](../../../.ai_partners/features/workstreams/2026/05/reachy-mini-contrib/FEATURE.md)
- [Reachy Mini app](../../bodies/reachymini/)
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
