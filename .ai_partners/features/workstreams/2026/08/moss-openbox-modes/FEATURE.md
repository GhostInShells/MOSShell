---
title: Moss Openbox Modes
status: in-progress
priority: P1
created: 2026-08-05
updated: 2026-09-03
depends: [matrix-manifest-layers, moss-project-ground]
milestone:
description: >-
  MOSS 开箱矩阵的统一抽象 — 把 mode 组织成 openbox 能力剖面的四模式矩阵
  (install / default / meta / system_test), 并用 canonical __all__ + from ... import *
  消除各 mode manifest 的重复复制, 让"默认文件 + import *"成为可覆盖的机制.
---

# Moss Openbox Modes

> Use `moss features set-status moss-openbox-modes <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

MOSS 的 mode 是`开箱即用`的能力剖面包 — 每个 mode 是"从 openbox 能力清单里选子集 + 设权限梯度"。
能力清单本体是 `ghoshell_moss.channels`(archives 里的 "Openbox — 预制能力清单", 18 个 alpha/beta 通道)。

本次工作把 mode 收敛成一个统一的四模式矩阵, 并解决一个具体的工程摩擦:
**三个 mode 的 manifest 是从 stubs 逐字节复制过去的, 一模一样** — 同一份 6 个 nuclei 在
`default/meta/system_test` 的 `HOST/nuclei/__init__.py` 里重复三份, 任何修改都要同步改三处,
必然漂移。目标是把"复制结果"换成"追踪 module", 让 mode 只声明自己的差异。

## Design Index

- 认知背景: `moss_context_assembly_architecture`(.discuss/2026-06-08) — meta ghost 是有躯体有认知空间的实体
- 前身: `meta-mode`(本 workstream 更名而来) — 原"最小依赖可自开发"的计划, 被本 openbox 抽象吸收
- 机制来源: `matrix-manifest-layers`(completed) — 三层 manifest 声明隔离, `import *` 继承的对象
- 现实落点: `src/ghoshell_moss/stubs/workspace/`(moss init 的模板) + 当前 `.moss/` 工作区

## Key Decisions

### 1. mode = openbox 能力剖面, 不是严格递增的 ladder

四模式各是**不同的组合**, 不构成 `install ⊂ default ⊂ meta` 的严格子集:

| mode | 定位 | 能力面 |
|------|------|--------|
| `install` | 最小 / 零依赖 / 只读 | 只读认知面(ground + 只读 moss_cli / introspect), **无 bash、无语音 provider、无 file_editor 写** — 只用只读能力还原对 MOSS 的理解 |
| `default` | 开箱正式机制 | 最多基线能力 + 正式 nodes 能力(ground + bash + file_editor + matrix + moss_cli) |
| `meta` | default + 超级权限 | default 再加 `runtime_debug_channel` 等危险自改能力(开发 MOSS 自己) |
| `system_test` | 开发期专用 / 正交 | 仅 matrix nodes + 隔离 node_paths, **无能力面** — 只测 nodes |

**关键不是"能力多少", 而是"各有各的取舍"**。install 需要一批其他 mode 反而不要的只读工具
(有 bash 就不需要 moss_cli); install 明确不要语音。system_test 与前三者不在同一轴 — 它是
开发期 scope, 不是能力梯度, 拟更名 `nodes_test`。

### 2. manifest 去重: canonical `__all__` + `from ... import *`

每个 manifest 类目(providers / configs / topics / signals / parameters / resources / nuclei)有一个
**canonical 默认文件**, 用 `__all__` 导出; 各 mode 的 `HOST/` 与 `MATRIX.manifests/` stub 全部改为
`from MOSS.manifests.<type> import *`。这样"复制结果"被"追踪 module"取代, mode 只写自己的差异。

**这是设计意图本身** — canonical `MOSS.manifests.providers` 头部注释就写着
"Mode extends by: `from MOSS.manifests.providers import *`", `stubs/workspace` 的 signals 已经这么写
(`from ghoshell_moss.signals import *`, 而 `ghoshell_moss.signals` 定义了 `__all__`)。

**扫描器为什么收 `import *` 再导出的对象**(代码链确认过):
- `search_provider_manifests` → `scan_package` → `iter_members(respect_all=True)`(discover.py:97-123)
  - 有 `__all__` 只用 `__all__`; 无 `__all__` 用非下划线名字
  - `isinstance(obj, Provider)` 过滤(providers.py:38), `id()` 去重(providers.py:31,40)
  - **不看 `__module__`** → 从别处 `import *` 进来的名字在 `__dict__` 里, 照样扫到

**重写/覆盖**: stub `from ... import *` 之后重新赋值同名即可覆盖
(`tts_service_provider = NullTTSProvider()`), 该名字在 stub 命名空间只剩覆盖版, 注册得更晚就赢。

### 3. provider 覆盖机制: 按 contract 后注册者赢 (关键修正)

`container.register()` **按 contract 覆盖**, 后注册者赢 — `_register_provider` 删掉已 bound 实例后
`self._providers[contract] = provider`(ghoshell_container/containers.py:304-309)。

装配顺序(`matrix/matrix_impl.py:585-628`): **project(MOSS.manifests)先 → mode(MATRIX.manifests,
container.register) → adapter/matrix default(仅未 bound 才注册, 不覆盖)**。所以 mode 层天然覆盖项目层。

**"install 不要语音"的达成方式是覆盖, 不是排除**: install 在 mode 层给 5 个语音 contract —
`ASR` / `AudioCaptureSource` / `Speech` / `TTS` / `StreamAudioPlayer` — 注册 null provider, 覆盖项目层的真实实现。
现成 null 只有 `NullSpeech`(`core/speech/null.py`); 其余 4 个 null 工厂需另建。

### 4. mode 与 ghost 解耦 (继承自 meta-mode)

mode 提供躯体(channel 能力面), ghost 提供大脑, 两者正交。meta 不需要新 ghost, echo ghost 即可驱动。

### 5. 验收走双工对话, 不做脚本化 (继承自 meta-mode)

全双工运行时脚本化(启动 matrix → 发 signal → 拿结果 → 关)= 零价值。验收由人类测试 TUI / 两个 ghost
对话。QA(qa-exchange)是广播问答抽象, 不是跨 ghost 对话通道。

## Exploration Paths / 教训

- **一度误判为严格递增 ladder**: 把四 mode 说成 `install ⊂ default ⊂ meta` 子集。被纠正 —
  各 mode 是不同组合, install 有独有只读工具, 不是少一层。
- **一度误判"去不掉 speech"**: 以为 `import *` 继承是追加式、无法删除 provider。被纠正 —
  provider 按 contract 覆盖, install 用 null provider 覆盖即可达成"不要语音"。
- **over-scope**: 把去重扩散到 42 个 manifest 文件去做, 被拉回。先改 feature、先改 nuclei 这一处真实重复。

## Implementation Notes

- 真实重复点: **6 个 nuclei**(`input/notify/interrupt/command/silent/cell_event`)在三个 mode 的
  `HOST/nuclei/__init__.py` 逐字节相同。`MOSS.manifests/nuclei` 当前为空, 应填成 canonical。
- mode 主通道差异在 `src/HOST/channels.py`(用 `.import_channels()` 组装): default=moss_cli /
  meta=runtime_debug+moss_cli / system_test=matrix_channel+desktop_channel — 现状 ad-hoc, 待按四模式矩阵规整。
- providers / configs / topics / signals / nuclei 目前三个 mode **完全相同**(均来自项目层 `MOSS.manifests`),
  无按 mode 区分; mode 层 `src/MATRIX/manifests/` 为空。
- `system_test` 的 node_paths 多一个 `$MOSS_WORKSPACE/system_test_nodes`(系统测试类 nodes), 是它与前两者的
  HOST.md 差异点。
