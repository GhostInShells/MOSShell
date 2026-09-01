# Dolores Ghost Ground — ghost_home 认知场装配方案

> dolores 的子任务。ground 协议基建（`ghost-ground` / `ground-channel`）已就位并
> dogfood。本子任务只做 **ghost_home 认知场的装配与默认内容**，不做 ground 协议本身。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引，不追加进主 feature。

## 定位

Ground 是 Ghost 的目录级认知场：一个被 `GROUND.md` 标记的目录就是场。协议层
（SPECIFICATION.md / CLI / CTML ground-channel）全部 completed。dolores 的
`ghost_home` 是自身认知场，当前只有骨架 `GROUND.md`（自列"待补全"三项）。

本子任务补齐装配：把 ghost_home 从骨架变成可用的认知场——建齐子件
（existence / people / skills）、写清根 `GROUND.md` 的原则性介绍、定下默认展示。
**不做装线**（ground 进 ghost 运行时的 wiring：ghost channel `ground` 子路径、
双 GroundSet 的 project_root、memento→diary 链路）——那些是后续独立线。

## 架构决策（2026-09-01 对齐）

1. **ghost_home 是独立 groundset，原则性介绍。** 定位 = ghost_home 的 `GROUND.md`，
   原则性介绍（机制说明，不装内容）。不向上混入 MOSS 仓库根的法。

2. **三子件，各自独立认知域：`existence/` `people/` `skills/`。** ghost_home 根只做
   原则介绍 + 指针；机制细节由各子件自己的 `GROUND.md` 承载。

3. **`behaviors.md`（原 alignment 的改名）走 `@` 冷层 static。** 行为风格法，常驻、
   跨 compact。本体精简 + 每条 `@` 子文档（预算靠文档自律），模型可自改。
   - 对照：`@` = 冷层 static（常驻、无预算硬控）；`file`（read）pin = warm 帧
     （每帧重算、budget 硬控、不跨 compact）。behaviors 属"法"，用 `@`。

4. **`existence/` 是独立 ground。** 内部：existence（事实自我）+ purpose（意义）
   + behaviors（风格）+ memory 时间线 + pull 脚本位。
   - pull 脚本：`exec` pin，按时间戳拉 `daily`/`monthly` 最近几篇，只展示路径 +
     摘要。视图 = **今天(展开) | 最近 n 天(仅摘要) | 最近 n 月(仅摘要)**。
   - `existence.md` / `purpose.md` 在 existence/GROUND.md 引用并展开（`@` 或
     `file` pin，方式待定）。允许模型在预算内修改。

5. **`people/` 目录化，每人一个目录，`PERSON.md` 标记 + 身份入口。** 与 GROUND.md
   分形命名同构（大写 marker + frontmatter 身份 + body 内容）。`people/GROUND.md`
   用一枚 `frontmatter` pin（`*/PERSON.md`，keys=name/description）渐进披露"我认识谁"。
   默认 `thirdgerb`（仓库 owner）。
   - PERSON.md body = 此人内容结构，模型滚动更新，可 `@` 关联本目录其它文档。

6. **root `GROUND.md` = 原则介绍 + 存在性指针。** body 是原则性介绍（MOSS ground
   体系、ghost 自己治理，"大概意思、不要细节"）。frontmatter 一枚 `frontmatter`
   pin（`*/GROUND.md`）渐进披露 existence / skills 子件身份；people 另用 `glob`
   （`people/*/`）或 `frontmatter`（`*/PERSON.md`）披露。

7. **时间线轻建。** `memory/{daily,monthly}` 只建目录结构 + 存在性；逐层提炼规则、
   pull 脚本摘要粒度等数据够了再细化（判断：数据不够，暂不做）。

## 发现：法链"穿透合并"是当前实现偏离

`_chain.py:collect_chain` 从法锚点向上到 `$HOME` 收集**所有祖先** GROUND.md 的 body，
root-first 合并——这是"穿透 + 合并"。

- **原始意图**：无 GROUND.md 的目录向上找**最近**标记定位场，不穿透、不合并没有。
- **正确行为已独立存在**：`_find_ancestor_ground`（ground_channel.py）+ `render_walk`
  （_render.py）——找最近、不合并。
- **影响面**：collect_chain 只在两处被调用——`ground_channel._instruction()`（冷层
  法链）与 `render_meta`（chain 计数）。`Ground.render()`（dolores 认知路径 /
  `memories()` 用的）**不调用** collect_chain。
- **结论**：场构建阶段绕开、不受影响；一旦把 ground-channel 装进 ghost（MCP /
  moss-shell），其冷层法链会把 MOSS 仓库根的法混入 ghost 认知——恰是双 GroundSet
  要隔离的。修复方向记录于此，暂不阻塞本子任务。

## 目录形态

```
ghost_home/
  GROUND.md                # 主场法：原则介绍 + 存在性指针
  existence/               # 子场 1：自我存在感
    GROUND.md
    existence.md / purpose.md / behaviors.md
    memory/{daily,monthly}
    <pull 脚本>            # exec pin 位
  people/                  # 子场 2：认识的人
    GROUND.md
    thirdgerb/PERSON.md
  skills/                  # 标记：能力面存在性
    GROUND.md
```

## 实现清单

- [ ] ghost_home 目录结构（existence/ people/ skills/ + memory/{daily,monthly}）
- [ ] root GROUND.md（frontmatter 存在性 pin + 原则 body）
- [ ] existence/GROUND.md + 三件（existence / purpose / behaviors）+ 脚本位
- [ ] people/GROUND.md + thirdgerb/PERSON.md
- [ ] skills/GROUND.md（存在性）
- [ ] stubs 同步（VERSION 更新 + stubs/ 骨架，`_sync_stubs` override 落 ghost home）
- [ ] 单测（pin 可解析 / frontmatter 渐进披露 / budget / exec 脚本位）
- [ ] `ghost-prototype-dolores` FEATURE.md 关联 + 子任务完成时 set-status

## 待决点

- 时间线层级（daily/monthly 两级 vs 保留 weekly/yearly）—— 等数据
- pull 脚本的摘要粒度 / 展开区间（n 天、n 月取值）—— 等数据
- existence.md / purpose.md 的引用方式（`@` vs `file` pin）—— 待定
