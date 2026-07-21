---
title: Node Migration — .moss_ws/apps → nodes/ 开箱架构
status: in-progress
priority: P0
created: 2026-07-21
updated: 2026-07-22
depends:
  - matrix-cell-governance
  - cells-cli
milestone: 0.1.0
description: >-
  将旧 .moss_ws/apps 体系迁移到新 nodes/ 目录。设计开箱分类架构，APP.md→NODE.md
  声明转换，适配新 Matrix API。.moss_ws 目录 0.1 正式版全删除。
---

# Node Migration — 开箱架构

> 人类架构师 + claude-opus-4-7。将旧的 22 个 app 迁移到新的 nodes 体系，
> 设计分类架构，建立开箱即用的 node 能力集。

## Motivation

`.moss_ws/` 目录整体在 0.1 正式版要删除。其中 `apps/` 体系承载了 22 个已实现的
能力单元，需要迁移到新的 `nodes/` 目录结构。迁移不仅是文件搬运——声明格式从
APP.md 变为 NODE.md，CLAUDE.md 删除，exec 格式直书，实现代码适配新 Matrix API。

这是一次开箱架构整顿：去掉死代码，重新分类，建立 node 开发的标准模板。

## Design Index

- 参考 node 实现：`nodes/skins/desktop-gui/` (NODE.md + main.py 标准模板)
- cells-cli 决策 (§R rewrite)：`cells-cli/FEATURE.md` — NODE.md 格式、exec 直书、
  specification 删除、五类 cell 分类删除
- matrix-cell-governance：三域模型 (Manifest/Record/Presence)、膜承诺、
  NodeManager/NodeManifest 新 API

## Key Decisions

### 1. 六分类架构

```
nodes/
├── bodies/          ← 物理身体 + 虚拟身体
│   └── live2d/         虚拟角色 (miku)
├── sensors/        ← 感知/输入设备
├── tools/          ← 工具能力 (含 web 搜索)
├── games/          ← 游戏与互动
├── im/             ← 即时通讯集成
└── ui/
    ├── frontend/   ← 直接渲染像素 (给人看)
    └── servers/    ← 提供可渲染界面 (给 AI 用)
```

**设计原则**：
- `bodies` 不区分物理/虚拟——都是 Ghost 的身体
- `ui/` 拆 frontend/servers 两个二级目录：desktop-gui 和 reflex 是 frontend，
  playwright 是 server（AI 用它打开 n 个 mermaid 图）
- `tools` 收纳浏览器自动化以外的工具，含 web 搜索 (trafilatura)
- 旧 `browsers/` `genkits/` `web/` 三个单 app 分类折叠消除
- 旧 `skins/` 并入 `ui/frontend/`

### 2. 迁移清单：第一梯队（12 个必须迁）

| 旧路径 | 新路径 | 备注 |
|--------|--------|------|
| `bodies/g1` | `bodies/g1` | 真机 G1 人形机器人 |
| `bodies/g1_sim` | `bodies/g1_sim` | G1 纯软件仿真 |
| `bodies/reachymini` | `bodies/reachymini` | Reachy Mini 手臂机器人 |
| `live2d/miku` | `bodies/live2d/miku` | 虚拟角色身体，需重建到能跑 |
| `sensors/audio_capture` | `sensors/audio_capture` | 系统音频采集 |
| `sensors/listener` | `sensors/listener` | Volcengine ASR |
| `sensors/ptt_listener` | `sensors/ptt_listener` | PTT 语音输入 |
| `sensors/vision` | `sensors/vision` | OpenCV 相机 + 人脸检测 |
| `tools/moss_self` | `tools/moss_self` | MOSS CLI 自举 |
| `im/feishu` | `im/feishu` | 飞书 WebSocket 长连接 |
| `games/ai_eye` | `games/ai_eye` | AI 眼球 pygame 化身 |
| `games/gomoku` | `games/gomoku` | 五子棋 |
| `games/minecraft_bot` | `games/minecraft_bot` | Mineflayer JS bridge |

### 3. 迁移清单：第二梯队（7 个）

| 旧路径 | 新路径 | 备注 |
|--------|--------|------|
| `tools/screen_capture` | `tools/screen_capture` | 截图工具 |
| `tools/image_importer` | `tools/image_importer` | 图片批量导入 |
| `tools/video_importer` | `tools/video_importer` | 视频批量导入 |
| `web/trafilatura` | `tools/trafilatura` | URL → Markdown |
| `sensors/waveform` | `sensors/waveform` | 终端波形可视化 |
| `ui/reflex` | `ui/frontend/reflex` | 流式 GUI 页面 |
| `browsers/playwright` | `ui/servers/playwright` | AI 控制浏览器 |
| `skins/desktop-gui` | `ui/frontend/desktop-gui` | 人类观察界面 (已迁) |

### 4. 不迁移的应用（5 个）

| 应用 | 原因 |
|------|------|
| `sensors/voice` | 空目录，只有 .venv，无代码 |
| `genkits/image` | 骨架，"当前未开始实现" |
| `genkits/video` | 骨架，"当前未开始实现" |
| `ui/moshi` | 独立 project，未来独立发版 |
| `web/resource_server` | 资源体系保留原位，暂不迁移 |

### 5. 声明文件转换规则

每个迁移的 node 执行以下变换：

- `APP.md` → `NODE.md`：旧 frontmatter (executable/script/arguments/respawn/workers/max_age) 
  → 新 frontmatter (name/description/singleton/exec: {command, args, env})。cells-cli §R-1 定义。
- `CLAUDE.md` → **删除**。模型入口信息进 NODE.md body (instruction)。
- `README.md` → 保留。面向人类开发者。
- `main.py` → 适配新 Matrix API：
  - `matrix.provide_channel(channel)` (新) 替代旧启动模式
  - `matrix.cell_workspace.configs()` → `matrix.workspace.configs()`
  - 移除 `moss apps test` 相关的旧入口约定
- `pyproject.toml` → 保留。独立依赖声明不依赖框架迁移。
- `runtime/` → 保留运行时目录结构。
- 按需创建 `INSTALL.md`（依赖安装指南）。

### 6. miku 特殊处理

`live2d/miku` 是最早的 channel 躯体实现，当前在 `nodes/live2d/miku/`。
迁移到 `nodes/bodies/live2d/miku/` 后，需要**重建到可运行状态**——不仅是声明文件转换，
还包括让 miku_channels 的 channel 体系在新 Matrix API 下工作。

## §trafilatura Pilot (2026-07-22)

以 `web/trafilatura` → `tools/trafilatura` 为第一个迁移实例，端到端验证了
create → install → run → mesh accept → CTML 调用的完整闭环。

### 摩擦点记录

1. **NODE.md instruction 不要写 CTML 命令**。channel 是自解释的（interface 反射），
   instruction 只写：节点解决什么问题、提供什么资源。mesh 里 channel 名是 fullname
   (`tools_trafilatura`)，和代码中 `channel.name` (`web_trafilatura`) 不同，
   所以 instruction 不应该猜测 CTML 路径。

2. **独立 venv 节点 exec.command 用 `.venv/bin/python`**。`command: python` 会
   解析为 `sys.executable`（MOSS 环境 Python），节点自己的依赖找不到。有
   `pyproject.toml` 的节点必须指到自己的 `.venv/bin/python`。

3. **ghoshell-moss 依赖优先用 `[matrix]`**。`[host]` 带 TUI/音频/进程管理等全套，
   节点通常只需进网。`[matrix]` 只有 `eclipse-zenoh`。

4. **`python-dotenv` 缺失**。`project.py` 引用了 `dotenv` 但没在 `[matrix]`
   可选依赖中声明，靠 `[host]` 的 transitive 掩盖。已加到 `[matrix]`。

5. **`CellNamePattern = '^[a-zA-Z0-9_]+$'`** 不支持连字符。`desktop-gui` 需改名
   `desktop_gui`。

6. **system_test mode 的 `node_paths`** 默认只有 `.moss/system_test_nodes/`，
   不含根 `nodes/`。已在 HOST.md 补充。

7. **`moss nodes list` 加了 Installed 列**，一眼可见安装状态。

### Matrix API 补充

在 Matrix ABC 和 MatrixImpl 上新增两个 API（迁移必要）：

- `cell_workspace` (property) — 返回 cell home 的 `Workspace`（节点自己的 configs）
- `resources` (property) — 返回 `ResourceRegistry`（跨 scheme 资源路由）

## Implementation Notes

- 本 feature 的职责：**规划与监督**。每个 node 的迁移由独立任务完成，不在此 feature 内一把梭。
- 迁移顺序：先迁简单的 (tools/sensors) 建立模板，再迁复杂的 (bodies/games/ui)。
- trafilatura 是第一个迁移成功的 node，后续节点参照其模式。
- 后续迁移用 `git mv` 移动文件到 `nodes/` 下，保留 git 历史。
- `live2d/miku` 目录移动后，`nodes/live2d/` 目录删除（该分类并入 bodies）。
- sensors 内部实现的"开箱适合度"由人类判断——迁移是机械操作，但实现质量审查是人工 gate。
- `.moss_ws/apps/` 在所有迁移完成后整体删除，.moss_ws 其他子目录同时清理。
