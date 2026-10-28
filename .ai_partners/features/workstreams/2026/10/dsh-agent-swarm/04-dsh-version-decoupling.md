---
name: 04-dsh-version-decoupling
description: dsh 具体版本的依赖解耦 — packages/ 与独立版本线
---

# 04 — dsh 具体版本的依赖解耦

独立的小任务，不是并线前提，可以穿插进行。

## 他说

- 围绕这次开发，deepseek harness 本身的 moshell 兼容应该建立独立的 `packages/`，
  在 monorepo 中迭代，但通过**独立版本号**发布。
- 版本号：**头两位对应 dsh 的版本变更，第三位留给我们做 fix**。

## 待确认（我推断，未确认）

- dsh 现在是 `0.x`，字面上的大版本是 0，所以实际规则应为 **dsh 的 major.minor 占前两位**：
  `0.2.7` = 面向 dsh 0.2 的第七次自身修正；dsh 到 0.3 另开 `0.3.0` 一条线，`0.2.x` 继续可装。
- **依赖方向单向**：兼容包依赖 `ghoshell_moss` 核心，核心不依赖它。
  （依据：现有纪律是 `deepseek_harness` 只被 `ghosts/dolores` 引用，核心 / host / matrix 不直接依赖。）

## 待他卸载

- 发布到哪（PyPI 还是先只做版本线）。
- 与 `ghost-replaceable-agent-driver` 的关系：那一条定的是"ghost 不被某个运行时定义"，
  这一条定的是"这个运行时自己怎么版本化"。两者相邻，但边界要写清，否则会互相长进去。

## 不做什么

待补。
