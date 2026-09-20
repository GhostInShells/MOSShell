# Dolores 自迭代能力地图 (下阶段)

> status: **next-phase** — 本阶段接线完即收尾, 下一阶段重点做 ground 与认知工具. 这张图是下一阶段的种子, 先钉在这里免得忘.

## 背景

自迭代 (ghost 改自己) 的落点是一张**路由表**: 「改这类东西 → 走这个机制」. 现在它只散落在 prompt 的 Where You Are / Matrix / Deepseek Harness 几段散文里, 没有收束.

裁决 (2026-09-17): **这张表不进 prompt (成本高, 且只在自迭代那一刻才用到), 也不做成 skill (它是决定做什么之前的元上下文, 不是动作配方). 落 ground** —— 下一阶段作为 ground 的一个 field (自迭代地图), wake 初见读一遍, 之后按需重读. prompt 只保留五大支柱的「脊」+ 一行指针.

五大支柱 (MOSS 与别的体系不一样的地方): ctml / mindflow / matrix / ground + memento.

## 七方向 (原文)

1. **project coding** — per project, 主要看 dsh 能力.
2. **os 脚本能力** — 重点走 project / ghost 的 skills 体系, 不用很重.
3. **认知 / 记忆 / 经验 (尤其 wake 初见)** — 走基于文件系统的 ground.
4. **人机共享空间, 双向实时交互** — matrix nodes 体系, 包括 gui / 机器人 / 感知等.
5. **moss 启动状态** — 走 moss cli 提供的知识, 有权限时管理 moss workspace. 核心是 ioc provider, 感知核, 源码等.
6. **外部能力接入** — skills / mcp 等. moss node 是一种技术选项.
7. **推理内核优化** — dsh plugin, 在 ghost home.

## 落地 (下一阶段)

- 落 ground: 一个 field (如 `self-iteration`), 走 fields 索引自发现.
- prompt: 五大支柱收成「脊」(每支柱一两行), 加一行指针指向 ground 的这张图.
- 字段名别用 `unsorted` —— 自解释体系不接受杂物抽屉.

## 自驱 idle 方案 (2026-09-21 裁决)

ghost 永久自驱 (自己写的 loop) 的反身性机制. 三种候选——① nucleus 低优 impulse ② mindflow idle 的 python 驱动 ③ 旁路 agent 同上下文想一帧——**选 nucleus 触发 + idle 回调**; 机制 ② 明确否掉: 反身性自驱应是 ghost 在上下文里"想"要不要动, 不是执行一段脚本 (脚本把自驱外部化, 与 harness/状态机同形状, 只是搬进自己的 sandbox).

### 机制 (大部分已落地, 不默认开启, ghost 自己调)

| 部件 | 现成件 |
|---|---|
| 反身性 channel | `DoloresEgoNucleus` (self-wake 信号) |
| idle 回调 | `Mindflow.when_idle(callback)` (`_mindflow.py:267`, 转入 idle 时触发) |
| 闲时逻辑按 startup 同理 | `startup/` 场 (ground 治理 + 文件加载 + ghost 自改) |
| 低优信号 | nucleus impulse 路径 (BACKGROUND → attended → INFO 注意力) |

**唯一新件 = N 秒节流**: `when_idle` 在"转入 idle 那一刻"触发, 不是"持续 idle N 秒"才触发. 回调挂延迟任务——N 秒后检查"是否仍 idle", 是才发一次低优 signal, 发完不重发 (每段 idle 只 poke 一次, 不是持续发). N 秒可加随机 jitter, 避免固定节律. 单个阈值不是参数化状态机.

### 配置形态 = 固有发现场

闲时逻辑 (连同 startup / frame) 收成一个 ground 场, 简单解释机制, **不自动枚举、非唯一、允许自建子目录**——"固有发现场" = 默认发现点, 不是唯一目录. macro / skill / feature 同性质: 把整个 moss 迭代协作机制 (三元工程第三元) 往 ghost 里搬. 本期 skills / macro 可以有 (标定), features 等治理完再说.

### 边界

所有 dolores ghost channel 能约束 scope 的, 边界一律 = **project home**, 不做默认泄漏. frame 已接线为 `project_home/.ai_partners/frames`.

### 未来升级

闲时逻辑升级到 exec 脚本时, 复用 ground 的 exec pin 协议 (SPEC §5.5)——脚本配置在别处 (被引用的场件, 不内联), 协议一致, 不新开协议.
