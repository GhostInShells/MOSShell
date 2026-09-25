---
created: 2026-07-21
depends:
- matrix-cell-governance
- cells-cli
description: 将旧 .moss_ws/apps 体系迁移到新 nodes/ 目录。轻依赖归并共享 venv、 重依赖独立；NODE.md 声明、适配新
  Matrix API。
milestone: 0.1.0
priority: P0
status: completed
status_note: nodes/ 开箱架构全量落地, 迁移收口; .moss_ws/apps 余档留原地待清
title: Node Migration — .moss_ws/apps → nodes/ 开箱架构
updated: '2026-09-20'
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

> 已被 2026-09-20 的「收口」取代 —— 下表是当时的计划，别当现状读，保留只为对齐历史决策。

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

## 收口（2026-09-20）

**完成。** nodes/ 开箱架构落地并稳定运行 —— 迁移不再是进行中的工程，而是既成事实的目录约定。

固化下来的约定（本 feature 长出来的，现在自立）：

- 一个 node = 一个目录 + `NODE.md`（声明 + 给运行中 ghost 的 instruction 正文）+ 入口脚本。
  `INSTALL.md` 存在与否推导 `installed` 语义（有则靠 `.installed` 标记，无则天然已装）。
- 轻依赖归并**共享 venv 组**（`nodes/tools`、`nodes/visions`：父目录持 `pyproject.toml` +
  `.venv` + 权威 `INSTALL.md`，子 node 用 `exec.command: ../.venv/bin/python`）；
  重依赖独立 venv（`live2d/avatar`、`os/*`、`screens/*`、`browsers/playwright`）。
- 不分发资产（模型、专有 SDK）走 node 内 `INSTALL.md` + gitignore，不进仓库。

**现实与当初规划的偏差**（记下来，免得下次再按旧表读）：

- 六分类从未执行。最终按域落位：`browsers / deepseek-harness / live2d / os / screens /
  tools / unitree / visions / webview_apps`。
- **sensors 共享组从未成立**：`nodes/sensors/listener` 随 voice 收编进 host 内核后删除
  （`f4fbdc68`），`nodes/sensors/` 目录空置。
- **tools 共享组只落了一个成员**（trafilatura）。共享组的价值是"多个轻依赖 node 共用一个
  venv"，不是目录美学 —— 没有第二个成员时它就不该存在。
- 当初开箱清单里的 `desktop_gui` / `text_blocks` 都没留在原目录：前者并入 MOSS OS Control
  （`nodes/os/*`），后者被 artifacts 吸收（`a39c95c1`）。

`.moss_ws/apps/` 余档（96 个 git 文件）**无待迁项**：

| 余档 | 归宿 |
|---|---|
| bodies/g1 | → `nodes/unitree/g1/control` + `contrib/unitree/g1` |
| bodies/reachymini | → `contrib/moss_in_reachy_mini` |
| ui/reflex | → 被 webview artifacts 取代（text_blocks 曾用 reflex，已删） |
| tools/screen_capture | → `contrib/channels/screen_capture.py`（通道，不是 node） |
| genkits/image | 已确认不迁 |
| bodies/g1_sim | 废弃实验（仓库内无任何引用） |
| im/feishu | 不在本 feature：归 `feishu-channel-integration` workstream（pending） |

`.moss_ws/` 已整体删除（2026-09-20，人类拍板）—— 要捞从 git 历史捞。