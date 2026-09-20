# Dolores Ghost Home Governance — 认知场的形态与治理

> dolores 的子任务。ghost_home 的**形态裁决**: 哪些场是认知核心 (第一眼看见)、
> 哪些只做存在性发现、时间线如何以目录分区自治、机制可以在哪里替 ghost 写盘。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引, 不追加进主 feature。

## Motivation

ghost_home 是 Dolores 的认知领地, 但它的拓扑与治理从未被正规化。三处悬空:

1. **子场不会物化。** 根场没声明 `groundset`, 实测 `DefaultGroundSet` 只开 root
   (`active grounds: ['deepseek']`), 于是 existence 的 identity / timeline pin 一个都
   不渲染。子场只剩"名字 + 一行描述"。
2. **"第一眼看见什么"没有定义。** 哪些场必须 render、哪些只报存在, 目前是设计的
   巧合而非裁决。
3. **机制写盘边界不清。** 同一目录有多个写入者 (stubs / dsh_stubs / plugin+preset /
   runtime / ghost 自己), 谁的领地在哪没有声明; 而时间线的自动建文件又要引入新写入。

本阶段只求**模样对**: 把拓扑和边界钉死, 不追机制优化。

## Design Index

- Ground 协议: `src/ghoshell_moss/ground/SPECIFICATION.md` (groundset §3, 物化 §7.2, exec §5.5)
- Ghost runtime 装配: `src/ghoshell_moss/ghosts/dolores/_runtime.py`
- 根 channel 组装: `src/ghoshell_moss/ghosts/dolores/channel.py`
- ghost home 骨架: `src/ghoshell_moss/ghosts/dolores/stubs/`
- 层叠图与缺口盘点: 见本文件 §场拓扑

## Key Decisions

### K1. 两档可见性 = 两个已存在的送达面, 不是新机制

| 档 | 机制 | 送达时机 |
|---|---|---|
| **A 存在性发现** | root `GROUND.md` 的 frontmatter pin (`*/GROUND.md`) → root render → `memories()` 的 `<memory><ground>` | session 创建 |
| **B 挂载** | root 声明 `groundset: [...]` → `DefaultGroundSet` 构造期物化 (一层) → ground channel `startup` 挂成子 channel (instruction=meta, notice=渲染帧) → **facade** (`epoch_start_point` 全量 + `facade_delta` 每帧增量) | epoch 起点 + 内容变化时 |

- **判据**: 不变的"法"走静态面; 会变的"内容"走 facade 自动重发。
- **A 档已在跑, 零新代码**; B 档开关就是 `groundset`, 只是根场没声明。
- **代价是预算问题, 不是一次性成本**: 挂载场内容一旦变化就重发整帧暖数据。8 个场
  全挂 = facade 背 8 帧。存在性档是对 token 的预算选择, 不是省事。
- **索引与挂载会重叠** (root 的 glob 不区分谁挂了 groundset)。**裁决: 接受重叠** —
  地图本就该完整, 重复的只是 name + description 两行。不引入 `ignore` 排除
  (其语义是"不属于本场", 挪用即污染), 也不改成逐条显式 pin (啰嗦且失发现性)。

### K2. 自治优先 — 机制进 ground, 不进 dolores 源码

模型要自治, 机制就得落在**场里** (GROUND.md 的纯文本规则 + 场内脚本), 而不是
dolores 源码里。改场里的文件 = 自治; 改 dolores 源码 = 改自身源码 + 重启。这条
是 K3/K4/K5 的总纲。

### K3. 时间线走 groundset + exec pin (推翻早期 memories 化)

- **早期裁决被推翻**: 本子任务一度倾向"timeline 直接 memories 化" (运行时约定
  逻辑按时序塞进 `memories()`)。现在否定 — memories 化会把时间线形状写死进运行时
  代码, 违背 K2 自治。
- **载体 = exec pin**: `ref` 指向场内的 timeline 脚本 (ghost-owned), `mode=python`,
  `cwd=$GROUND`, env 注入 `GROUND`/`CWD`。脚本即机制。
- **为何 exec 是唯一对自治成立的载体** (契约见 SPEC §5.5): `ref` 必须是场内相对路径
  脚本 (不可 `..` 逃逸、不接 inline shell)、失败可见。脚本住在场里 → ghost 可自改。

### K4. 时间线目录分区 (Y/M/D) + metadata

```
journal/
  Y2026/
    yearly.md
    M09/
      monthly.md
      D20/
        daily.md
```

- **时间分割在目录**, memento 同款。glob 即人生轨迹; 缺天 = 缺目录, 诚实。
- **每文档 frontmatter**: `summary` (默认空, <100 字) + `status` — `pending`(自动创建)
  → `writing`(撰写中) → `closed`(已经结束)。有篇幅限制。
- **呈现**: yearly 存在性 / monthly 存在性 / daily **今天展开**。
- **GROUND.md 只标定**如何用这三个文档 + 对应目录, **纯 prompt**。ghost 改了它只改
  规则, 文件照常自动生成 (规则是纯文本)。
- 逐层不跳级 (daily → monthly → yearly); 提炼规则的数据打磨留下一阶段。

### K5. exec 脚本 = 存在性物化 (write-on-read)

- **exec 是 ground 里唯一执行代码的动词, 其余动词纯读** → "脚本写 memory 文件"正好
  落在 exec 被授权的副作用面上, 边界干净。
- **compute-on-observe**: 每次 render 的 observe 阶段跑一次进程, **无跨 render 缓存**
  (`_render.py:87`)。→ 脚本必须**便宜 + 幂等** (glob + 缺了建 + print)。
- **触发 = 渲染即"看见时创建"**, 不是定时器。这天没启动就不创建 (可缺), 由 memento
  的时间区间机制回溯补建 — 未来旁路/睡眠也能重建日记。
- 机制只做**存在性物化** (模板实例化成空壳), **内容仍归 ghost 写**; 模板是场内真件
  (如 `_template.md`), 机制读它、不内置。

### K6. architect 取消; architect / stage / regressions 一并收进 features

- **architect 场不要**。它的初衷是给 existence 做区分的对照, 不需要。
- 后续 architect 体系 + stage + regressions 一起并入 features 体系。
- **features 场只提醒一句**: 这里用 `moss features --dir ...`; 模型读 `specification`
  就能自建自己的 features 体系。
- **已验证可行**: `moss features` 全命令带 `--dir` (指向 `.ai_partners/features/`);
  `moss features init -p <root>` 可脚手架; ghost 侧有 `moss_cli:exec` 能跑 moss 命令。
- **架构理解不落第二份地图**: `src/ghoshell_moss/architecture.py` (`moss codex
  architecture`) 是单一事实源。

### K7. frame 不进场 (不进 ground)

frame 自带自解释机制 (`spec()` + instruction), 不需要一个场来标记它存在。
它的落点是 channel, 不是 field。

### K8. 时间线是 memento-adjacent, 不合并

时间线是一种"更强耦合时间线"的记忆 — 与 memento 同族: memento 的时间区间机制让
任何记忆可还原, 值得记的东西可以直接放这里。但**不急于并入 memento**: 两个机制
各自演进, 保持边界。

## 场拓扑 (本子任务的交付物)

```
ghost_home/
├── GROUND.md            root — A 档 (memories, 法)     ← 声明 groundset
├── existence/           B 档 挂载                      identity / purpose / behaviors
├── people/              B 档 挂载 (总索引 + 概要)
├── journal/             B 档 挂载 (exec pin = timeline)  Y/M/D 时间线
├── startup/             A 档 存在性                    ← 补 frontmatter (已修)
├── skills/              A 档 存在性                    (同时是 .dolores.yml 的 dirs 落点)
├── features/            A 档 存在性                    ← 提醒 moss features --dir
└── frames/              非场 — channel root            ghost.frame 的帧根
```

`groundset: [existence, people, journal]` — 挂载这三个。

## Open Problems

> 已结案: O1 场名 = `journal`; O1b status = `pending`/`writing`/`closed`;
> O3 startup frontmatter 已补 (待修 → 已修)。

- **O5. frame 机制优化 (下一阶段)** — ① 无 list/发现面: instruction 说"list the frame
  root"但通道里没有该命令; ② 无 reload: `load` 对已载入 frame 返回 "already loaded",
  磁盘改了也不重读 — 与"会话中追加"直接冲突; ③ 多帧重发成本: 一次 `resolve` 重发
  **所有**已载入帧的全部问答 (named notice fragment 是现成解法)。

## Implementation Notes

- **VERSION bump 遗留 todo**: 骨架同步机制要区分"从零创建拷贝 (seed)" vs "硬机制升级
  (machine upgrade)"。`copytree` + version gate 全量覆写会吞 ghost 的自改 (identity /
  behaviors / GROUND.md 等)。plugin 复制做成旁路 (always override 的开发件) 就是这层
  意思; dsh 运行时加环境变量 option 是为后续铺路。
- **exec 每次 render 跑一次进程** (compute-on-observe 无跨 render 缓存) — 脚本须便宜
  + 幂等; payload 缓存 / materialize-read 分离是下一阶段优化。
- **timeline.py 已迁出 existence/** → 落 `journal/` 场 (exec pin); existence 只留
  identity / purpose / behaviors。
- `memories()` 在每次 `create_session` 调用 (session 可重建), 所以 A 档内容随 session
  重建刷新; B 档内容随 facade (epoch + 变化) 刷新 — 两档刷新时机不同, 是刻意的。
- **write-on-read 会污染 seed 目录**: 对 `stubs/journal` 就地 render 会执行 timeline.py,
  在 seed 里生成 `Y*/` 数据。所以开发/验证时对 journal 场用 tmp 或副本 render, 不要
  就地 render stubs。长期答案是 ghost-home 独立 git (journal 数据归 ghost 自己的仓库
  提交, 不进源码 seed); 现况 `.moss/ghosts/deepseek/` 已被 gitignore, 无碍。
