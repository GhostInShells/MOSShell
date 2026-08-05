---
title: Ground Channel — 认知场的运行时落点
status: draft
priority: P1
created: 2026-08-05
updated: 2026-08-05
depends: [ghost-ground]
milestone:
description: >-
  Ground 在 Ghost 运行时里的薄 channel：唯一职责是让"当前场的法链"跨 compact
  存活。全部内容交付转函数（CLI 等价），无子 channel、无对账、无帧覆写。
---

# Ground Channel — 认知场的运行时落点

> Use `moss features set-status ground-channel <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

ghost-ground 的 CLI 层（spec/init/frame/meta/observe/validate）已完成并 dogfood 验证。
遗留的 K14 是运行时落点 —— 旧设计是"父 channel + 每场一个 command-less virtual channel，
context_messages=帧"。推进前做了一次全量调研 + 与人类架构师的设计碰撞，结论是旧 K14
整体作废，换一个薄得多的模型：

**ground 协议 = 一个"读目录"的函数库。ground channel = 这个库中唯一必须跨 compact
存活的那一片：当前场的法链。其余一切（pin 内容、frame、对账、walk 渲染）全是函数调用，
与 CLI 逐字等价 —— 模型爱用 bash 调 `moss ground` 也行。**

核心事实：`read`/`frame` 的命令结果活在对话历史里，**历史会被 compact 压掉**；唯一跨
compact 不丢的载体是 channel static（moss_static 前置、跨 fold 缓存，channel-meta-dyn-static
的 LSM 事实）。所以 ground channel 的实质 = "把一个根的法链放进 static"。channel 与 CLI 的
差异仅此而已（static 自动进上下文 vs 模型手动 `moss ground frame`）。

## Design Index

- 上游协议：`ghost-ground`（`src/ghoshell_moss/ground/SPECIFICATION.md`）
- 参照框架：
  - `context-cache-engineering` — cold/warm/hot 三层 + ContextMonitor V2
  - `interleaved-ctml-thinking` — K5/K6/K8（拉/推两时钟、动词原子/上下文极简、append=observe）
  - `channel-meta-dyn-static` — LSM 静态前置 + delta append + fold 合并
- 本 workstream 的讨论轨迹：与人类架构师 2026-08-05 一场收敛（记录于本文件 Key Decisions）

## Key Decisions

### D1. ground = 读目录的函数库；channel = 法链进 static 的唯一载体

协议与 CLI 保持不变（ghost-ground 已完成那层）。channel 只做一件事：把当前场的法链放进
自己的 instruction/static。其余全部降为函数调用（= CLI 转发），模型可经 CTML command 或
bash 调用，体验等价。**channel 与 bash 的唯一差异 = static 自动进上下文且跨 compact 存活。**

### D2. 无子 channel（K14 旧设计作废）

否决"父 channel + 每场一个 command-less virtual channel + context_messages=帧"。那正是
"滥用 context messages"的形态：每场挂一帧，且 frame（k≈1 的 hot）被当 cold 放。
单一 channel 根在创建时传入的 path。virtual channel 的残余价值（第一次生产 warm 数据）
不成立 —— 拉模式下内容由模型按需 `read`，无需预置。

### D3. 无对账（hash shadow / stale / update 全部删除）

对账存在的唯一理由是"内容被推进上下文后，告诉模型心理拷贝过期"。拉模式下内容从不被推，
模型每次读到的都是新鲜的 —— 读到即承认，update 动词消失。这顺带消解了 ghost-ground 遗留
的 open/update 语义未对齐问题。变更感知降级为可选 `watch` 动作（见 D5），默认不给。

### D4. 内容交付全走 command；context_messages 默认空

pin 内容（file/glob/frontmatter/ls/exec 的观察结果）不进 context。模型 `read <pin>` /
`frame` 按需拉取，结果经 `<result>` 入对话历史 —— **对话历史就是 warm 层，模型自己当作者**。
hot 层仅当 `watch` 开启时存在：一屏变更信号（`label: changed/missing/added`），不带内容。

### D5. walk 反推 ground，导航无锚点

`walk(dir, ground=None)`：从 dir 向上找 GROUND.md 即 ground root，法链从 root 再向上收集。
$GROUND/$CWD/$HOME 锚点语法从导航动词消失，只残留在 CLI 的 pin path 参数内部。显式 `ground=`
仅在需要强制指定锚时用。

### D6. static = 创建时的根的法链（最后验证点，已定）

**当前法链 = 初始化时的根，不是最后停留的 cwd。** 理由：
1. 根子树内两者不分叉 —— 子场法链向上继承（K57），在场内下钻法不变；
2. 分叉点（跨场）是"改变栖息地"这个语义事件，该由显式 re-root 声明（场是开出来的不是走
   进去的，K57），而非让 static 偷偷跟随光标；
3. static 稳定性 = 缓存前缀稳定性。跟随 cwd 的 static 每次跨场都是一次 fold + 前缀重写。

接受的代价：跨场后 static 仍显示旧根法链，模型靠 breadcrumb 提示 + 显式 re-root 切换。
跨场是罕见且意图明确的动作，该付重拉成本；场内连续工作该享 zero-churn。

### D7. channel 的 pin = 目录级，细粒度 pin 降为 CLI 命令

channel 状态里的 pin 坍缩为目录：static = 当前根法链 + 最近 k 个 visited 根的 breadcrumb
（一行一个：$id/label/ground 路径）。细粒度 file/glob/frontmatter/ls/exec pins 全部留在
CLI 侧，模型按需调用。per-file 常驻 = 滥用 context；目录级法链 = 认知。

### 冷/温/热归位（context-cache-engineering 框架下的完整映射）

| 层 | 内容 | 载体 |
|---|---|---|
| cold | 当前根法链 + identity + k 面包屑 | channel static（付一次，恒 0.1） |
| hot | （可选）变更信号，一屏 | context_messages（默认空） |
| warm | pin 内容 —— 模型 `read` 拉进 | 对话历史（command result，模型当作者） |

## Implementation Notes

- **channel 形态**（薄到近乎透明）：
  ```
  state:        current_ground_root + visited_k 列表
  static:       当前根法链 + k 面包屑
  commands:     walk(dir, ground=None) / read / ls / frame / pin / unpin（= CLI 转发）
  context_messages: 空（或 watch 开启时的一屏变更信号）
  ```
- **唯一 integration 问题（待定）**：channel 怎么知道"现在在哪" —— Ghost 显式 `walk`
  驱动，还是 channel 读 Ghost 当前工作目录。决定 static 何时重绘。
- **推进前应修的 CLI P1 bug**（2026-08-05 调研发现，与 ghost-ground CLI 层相关）：
  1. `init --template 不存在` 静默降级成空 GROUND.md，应报错/警告；
  2. `range: "0-N"`（1-indexed 语法下非法 0 起点）validate 通过但 frame 静默渲染空，
     `_hash.py` 与 `_render.py` 两份 `_parse_range` 行为不一致；
  3. `ExecPin`/`ExecArguments` 未在 `ghoshell_moss.ground.__init__` 导出，`from
     ghoshell_moss.ground import ExecPin` 失败。
