# Dolores Ghost Ground — ghost_home 认知场装配方案

> dolores 的子任务。ground 协议基建（`ghost-ground` / `ground-channel`）已就位并
> dogfood。本子任务只做 **ghost_home 认知场的装配与默认内容**，不做 ground 协议本身。
> 由 `ghost-prototype-dolores` FEATURE.md 关联索引，不追加进主 feature。

## 定位

Ground 是 Ghost 的目录级认知场：一个被 `GROUND.md` 标记的目录就是场。dolores 的
`ghost_home` 是自身认知场。本子任务把 ghost_home 从骨架变成可用的认知场——建齐
子件（existence / people / skills）、写清根 `GROUND.md` 的原则性介绍、定下默认展示。
**不做装线**（ground 进 ghost 运行时的 wiring：ghost channel `ground` 子路径、
双 GroundSet 的 project_root、memento→diary 链路）——那些是后续独立线。

## 架构决策

1. **ghost_home 是独立 groundset，原则性介绍。** 定位 = ghost_home 的 `GROUND.md`，
   原则性介绍（机制说明，不装内容）。不向上混入 MOSS 仓库根的法。

2. **三子件，各自独立认知域：`existence/` `people/` `skills/`。** ghost_home 根只做
   原则介绍 + 指针；机制细节由各子件自己的 `GROUND.md` 承载。

3. **`behaviors.md`（原 alignment 的改名）走 `@` 冷层 static。** 行为风格法，常驻、
   跨 compact。本体精简 + 每条 `@` 子文档，模型可自改。

4. **`existence/` 是独立 ground。** 内部：identity（事实自我）+ purpose（意义）
   + behaviors（风格）+ memory 时间线 + timeline 视图。
   - 引用分层：identity 走 `file` pin（warm 帧 + budget 2k，滚动事实）；purpose
     与 behaviors 走 `@`（冷层 static，常驻，法）。
   - timeline 视图：`timeline.py`（exec pin, mode=python）输出
     今天(全文) | 最近 n 天(摘要) | 最近 n 月(摘要)，倒序。

5. **`people/` 目录化，每人一个目录，`PERSON.md` 标记 + 身份入口。** 与 GROUND.md
   分形命名同构。`people/GROUND.md` 用一枚 `frontmatter` pin（`*/PERSON.md`，
   keys=name/description）渐进披露"我认识谁"。默认 `thirdgerb`（仓库 owner）。

6. **root `GROUND.md` = 原则介绍 + 存在性指针。** frontmatter 一枚 `frontmatter`
   pin（`*/GROUND.md`）渐进披露三个子件的**类别**身份——不穿透进子件内部。

7. **时间线轻建。** `memory/{daily,monthly}` 先建目录结构；逐层提炼规则等数据
   够了再细化。

8. **git 即证据层（方向，非本期）。** ghost_home 独立 git 是真实动机：历史不可篡改
   是 commit 语义免费的，identity/monthly 随便重写、旧状态在 history。化身分叉 =
   branch。git init 落 sync/bootstrap 逻辑，但**不在场构建阶段**做。

9. **写机器与 ground 分离。** ground 的 body 是**法**，写清"这里欢迎写、写成什么
   形状、什么周期"（可写思维提示），不混入写协议。写工具归 agent 自带（dsh）+
   MOSS warrant。

10. **exec 支持 mode（shell/python/ts）。** 脚本文件不依赖 `+x`/shebang——mode 指定
    解释器（`python` 用 `sys.executable`，`shell` 用 `sh`）。协议层已落地。

11. **anchors 守恒（纪律，非 frontmatter）。** 不可压缩锚点放 body（读详情/改时才
    有意义），逐层提炼时原样上浮，是压缩中的守恒量。锚点集变了才重审 purpose。

12. **frontmatter 惯例 = description。** daily/monthly 的 frontmatter 只留
    `description`（一行摘要，被 timeline 视图消费）；date 在文件名（天然有序）。
    去 sources（memento 与 ground 是两个机制，ground 产物不是轨迹，不溯源）。

## 法链修复（2026-09-02）

`collect_chain` 原为"向上合并所有祖先 GROUND.md body"（claude.md 式穿透合并），
是历史遗留 bug——原始意图是"向上查找最近场，不穿透不合并"，只要 walk 行为。

- 已修复：`collect_chain` 改为"向上找最近的 GROUND.md，返回其 body（单层）"。
- 场的法 = 场自身的 body；无 GROUND.md 子目录向上找最近场（walk 定位）。
- 每个子场渲染自己的根，向上合并会造成大量重复。
- SPEC §7.3 同步改为 "Law — Nearest Ground, No Merge"；render_meta 的 chain 计数
  改为 law 存在性；ground_channel instruction 的"法链"改为"本场 body"。

## 目录形态

```
ghost_home/
  GROUND.md                # 主场法：原则介绍 + 存在性指针
  existence/               # 子场 1：自我存在感
    GROUND.md
    identity.md / purpose.md / behaviors.md
    timeline.py            # exec pin, mode=python — 时间线视图
    memory/{daily,monthly}
  people/                  # 子场 2：认识的人
    GROUND.md
    thirdgerb/PERSON.md
  skills/                  # 标记：能力面存在性
    GROUND.md
```

## 实现清单

- [x] ghost_home 目录结构（existence/ people/ skills/ + memory/{daily,monthly}）
- [x] root GROUND.md（frontmatter 存在性 pin + 原则 body）
- [x] existence/GROUND.md + 三件（identity / purpose / behaviors）+ timeline.py
- [x] people/GROUND.md + thirdgerb/PERSON.md
- [x] skills/GROUND.md（存在性）
- [x] exec 支持 mode（shell/python/ts）
- [x] 法链穿透修复（向上查找不合并）
- [x] 单测（渐进披露不穿透 / @ 冷层装载 / timeline 视图）
- [ ] stubs 同步（VERSION 更新 + stubs/ 骨架，`_sync_stubs` override 落 ghost home）
- [ ] `ghost-prototype-dolores` FEATURE.md 关联 + 子任务完成时 set-status

## 待决点

- 时间线层级（daily/monthly 两级 vs 保留 weekly/yearly）—— 等数据
- timeline 展开区间（n 天、n 月取值）—— 当前 14 天 / 6 月，可调
