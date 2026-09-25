---
created: 2026-08-07
depends:
- node-migration
description: '将 manifest 从两层 (MOSS.manifests / HOST) 扩展为三层: MOSS.manifests (通讯必需,
  任何 cell 入网) / MATRIX.manifests (环境能力, per-mode, 跨 cell 共享) / HOST (mode 专属). MATRIX.manifests
  初始全空, 未来承载音频等环境能力 provider.'
milestone: 0.1.0
priority: P0
status: completed
status_note: 人类判断可以 completed
title: Matrix Manifest Layers — 三层 manifest 声明隔离
updated: '2026-08-30'
---

# Matrix Manifest Layers — 三层 manifest 声明隔离

> 人类架构师 + Claude Opus 4.7。两个关键设计场景驱动三层剥离:
> (1) node cell 独立入网时拿不到 HOST 层的环境能力 provider;
> (2) 不同 mode 需要共享环境能力但不应互相污染。

## Motivation

当前两层的结构问题:

**MOSS.manifests(通讯必需)**: zenoh/topic/logger/session/qa/config/subprocesses/jobs。
任何 cell 入网都需要,职责清晰。

**HOST(mode 专属)**: 音频三件套 (AudioPlayerProvider / TTSServiceProvider /
TTSSpeechServiceProvider) 在这里。但 HOST 只在 host 启动时才叠加,node cell
独立 `matrix.discover()` 时拿不到。这意味着:

- **capture 拆成独立 node** 做采集 → 拿不到 AudioCaptureSource
- **player 拆成独立 node** 做蓝牙播放 → 拿不到 StreamAudioPlayer
- **TTS 在独立进程里分析音频文件** → 拿不到 TTS

node 可以直接 new 实例(不走 provider)——但长期来看,"跨 cell 共享的环境能力"
缺少一个明确的声明归属层,会导致应用者各自做脏解。

通讯协议走 scope,mode a 和 mode b 仍能组网——组网层是 mode 无关的,不需要三层。
但**能力装配层不是**:ConfigStore 是 mode-aware 的,ModeHome 在 matrix 上可见。
matrix 从来不是真正 mode 无关的,只是在通讯面保持了独立。

## Key Decisions

### KD1: 三层划分 — 语义按 cell 角色 × mode 关系切

| 层 | 包路径 | 语义 | 装配时机 | 初始内容 |
|---|---|---|---|---|
| **通讯必需** | `MOSS.manifests` | 任何 cell 入网都需要 | 无条件 | zenoh/topic/logger/session/qa/config |
| **环境能力** | `MATRIX.manifests` | per-mode, 跨 cell 共享 | mode 激活时 | **全空** |
| **mode 专属** | `HOST` | host 编排专属 | host 启动时 | channel/nuclei/mode provider |

MATRIX.manifests 的抽象形状与 MOSS.manifests (ProjectManifest) 相同:
providers/configs/topics/signals/parameters/resources/nuclei。

### KD2: 方案 B — MATRIX.manifests 放在 mode 包下, per-mode

```
.moss/modes/<mode>/src/
├── MATRIX/manifests/     ← 环境能力 (全空, 未来放 audio 等)
│   ├── providers/
│   ├── configs/
│   └── ...
└── HOST/                 ← mode 专属
    ├── providers/
    ├── channels.py
    └── ...
```

**接受**: per-mode 的 MATRIX,不同 mode 各自声明所需环境能力。default mode
的 MATRIX 初始全空,未来 default 加音频 provider,蓝牙 mode 加蓝牙 player
覆盖 default。

**拒绝**: 全局 MATRIX (`.moss/src/MATRIX/manifests/`)——跨 mode 共享在
HOST 的 `from xxx import *` 继承路径上已经成立,不需要全局 MATRIX 再引入
manifest 层级的跨 mode 共享。

### KD3: 装配序 — MATRIX 在 HOST 之前, 允许 HOST 覆盖

matrix 装配序:

```
project.container ← MOSS.manifests (通讯必需, 无条件)
if not env.no_mode:
    mode.MATRIX.manifests scan + register  ← 环境能力 (register 语义)
    mode.HOST scan + register              ← mode 专属 (覆盖同 contract)
adapter + matrix default (兜底)
```

register 语义天然去重——HOST 里的 provider 和 MATRIX 里同 contract 的,
HOST 覆盖。IoC container 的 register 本身幂等。

### KD4: no_mode 防御 — 显式跳过

`env.no_mode` 时,MATRIX 和 HOST 两层直接跳过,不扫描。

当前 (2026-08-07) `_wire_mode_overlays` 没有显式的 no_mode 检查,靠
"包路径不存在→扫描返回空"隐式成立。三层方案要求改为显式:
`if env.no_mode: return`。

no_mode 时的 matrix 是真正的最小形态——只有通讯必需,没有环境能力,
没有 mode 覆盖。独立 node cell 入网就是这种。

## 5 个实施注意项

1. **blueprint project 声明 Matrix Manifest 抽象** — 形状与 ProjectManifest
   相同,可能共用类,区别只在 root_package (`MATRIX.manifests` vs `MOSS.manifests`)。
2. **Project 暴露三个懒加载函数** — matrix_manifests / project_manifests /
   mode manifests (或 host 相关)。
3. **MatrixImpl._prepare_container 增加 MATRIX.manifests 装配** — 在
   MOSS.manifests 之后,adapter default 之前。
4. **manifests CLI 改造** — 展示三层及每层的所属 mode。
5. **stubs 搬迁 + mode 增加 MATRIX 层** — `.moss/` 下 mode stub 增加
   `MATRIX/manifests/` 目录结构,所有子包初态为空 `__init__.py`。

## 与音频的关系

本 feature **不含音频 provider 搬迁**。音频三件套当前在
`modes/default/src/HOST/providers`,未来 (独立 feature) 搬到 MATRIX.manifests 的
环境能力层。本 feature 只建好三层壳,MATRIX 层全空,为后续搬迁预留。

音频的四个基础能力 (TTS / StreamAudioPlayer / AudioCaptureSource /
VolcengineASR) 后续进 MATRIX.manifests,Speech (tts+player 组合) 保留在 HOST。

---

*架构讨论: 人类架构师与 Claude Opus 4.7, 2026-08-07*