---
title: Node Migration — .moss_ws/apps → nodes/ 开箱架构
status: in-progress
priority: P0
created: 2026-07-21
updated: 2026-08-14
depends:
  - matrix-cell-governance
  - cells-cli
milestone: 0.1.0
description: >-
  将旧 .moss_ws/apps 体系迁移到新 nodes/ 目录。轻依赖归并共享 venv、
  重依赖独立；NODE.md 声明、适配新 Matrix API。
---

# Node Migration — 开箱架构

> 人类架构师主导。本 feature 只做**规划与监督**，每个分组的迁移是独立任务，
> 不一把梭。分组方式与迁移细节随实践迭代，本文件只记录当前共识与状态。

## 当前方案 (2026-08-14 共识)

### 依赖分组: A + C — 轻归并、重独立

不按固定分类，按**真实依赖画像**分组。轻依赖 node 归并到一个共享 venv
父目录；重依赖/独有依赖 node 独立保留自己的 venv。

```
nodes/
├── tools/            ← tools 共享 venv 父目录 (样板已落地)
│   ├── pyproject.toml    # 聚合所有 tools 子 node 依赖
│   ├── .venv/
│   └── trafilatura/      # 子 node，无自身 venv
├── sensors/          ← sensors 共享 venv 父目录 (规划中)
└── ...               重依赖独立: desktop-gui(reflex)、qt_screen(pyside6)、
                       g1/control(unitree-sdk2py)、vision(cv2)
```

### 不穿透规则（共享组约定，不写进机制）

- 共享组父目录持有 `pyproject.toml` + `.venv/` + 一份权威 `INSTALL.md`
- 子 node 不携带 `INSTALL.md` / `.venv` / `.installed` → `installed` 恒 True，
  实际状态由父 venv 是否 sync 决定
- `exec.command` 用相对 node cwd 的 `../.venv/bin/python`（已验证可解析）
- INSTALL.md 语义**不加路径概念**（显式 project 路径声明暂不做）

### 分类漂移 — 六分类设计已失效

早期六分类（bodies/sensors/tools/games/im/ui{frontend,servers}）未照实执行。
现实是 `nodes/{live2d, screens, sensors, skins, tools, unitree, webview_apps}`，
screen-node / g1 / text-blocks 各自建了新目录。**保留现状，不重组回六分类**，
迁移只按依赖画像分组落位。

## 迁移状态清单

### 已开箱（nodes/ 内，6 个）

| node | 路径 | 来源 | 状态 |
|---|---|---|---|
| trafilatura | `nodes/tools/trafilatura` | `.moss_ws/apps/web/` | ✅ tools 共享组样板 |
| voice | `nodes/sensors/listener` | `.moss_ws/apps/sensors/` | ✅ 独立 venv |
| desktop_gui | `nodes/skins/desktop-gui` | `.moss_ws/apps/skins/` | ✅ 独立 venv |
| control | `nodes/unitree/g1/control` | `.moss_ws/apps/bodies/g1` | ✅ 独立 venv |
| screen | `nodes/screens/qt_screen` | screen-node 新建 | ✅ |
| text_blocks | `nodes/webview_apps/text_blocks` | text-blocks 新建 | ✅ |

### 待迁（按分组）

| 组 | node | 依赖 | 可推进 |
|---|---|---|---|
| tools 共享组 | screen_capture / image_importer / video_importer | `[matrix]` + mss + Pillow | ✅ 立即 |
| sensors 共享组 | audio_capture / waveform / ptt_listener（listener 已迁） | `[host,matrix]` + scipy + numpy + click | 待 tools 样板验证 |
| 独立 | vision | cv2 重依赖 | 待定 |
| 复杂/待评估 | g1_sim / reachymini / feishu / ai_eye / gomoku / minecraft_bot / reflex / playwright | 各自 | 人工评估 |

### 不迁移（已确认）

`sensors/voice`(空)、`genkits/image`、`genkits/video`(骨架)、`ui/moshi`(独立项目)、
`web/resource_server`(保留原位)。

## 历史轨迹（压缩）

早期设计轨迹——六分类架构、第一/二梯队迁移清单、trafilatura pilot 摩擦点
（NODE.md 不写 CTML 命令、独立 venv 用 `.venv/bin/python`、`[matrix]` 优先、
python-dotenv 缺失、CellNamePattern 连字符、system_test node_paths、
`moss nodes list` Installed 列）、Matrix API 补充（`cell_workspace`/`resources`）——
**已压缩**。完整轨迹用 git log 查看：

```bash
git log -p -- .ai_partners/features/workstreams/2026/07/node-migration/FEATURE.md
```

操作约定已固化在 `src/ghoshell_moss/stubs/node/` 模板与共享组 INSTALL.md 中，
无需重复记录。

## 下一步（可推进）

1. **tools 共享组收尾**：迁入 screen_capture（pyproject 追加 mss+Pillow）、
   image_importer、video_importer
2. **sensors 共享组**：audio_capture / waveform / ptt_listener 并入 `nodes/sensors/`
3. **vision 独立** + 复杂 app（feishu/games/bodies/ui）人工评估
4. **全部迁完后**：删除 `.moss_ws/apps/`
