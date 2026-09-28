---
description: >-
  MOSS 的 L3 产物维护清单 —— 整个项目对外呈现的面貌。只指向，不重复记录内容。
version: 1
updated: 2026-09-28
---

# SURFACE

> 机制与模板见 `README.md`（features 规范）与 `SURFACE.md` 模板。本文档只做指向。

MOSS（Model-oriented Operating System Shell）是 Ghost in Shells 架构的 **Shell 层** —— 让持久化智能体（Ghost）落入现实世界：感知、思考、行动并发、实时。项目自内核重构一路迭代到 v0.1.0（stage2），正在收口 **ghost 迭代自身** 的闭环。完整的理念与定位见根目录 `README.md` / `README.zh.md`。

## 清单

- **定位 / 理念** → 根 `README.md`（架构理念与定位）
- **入口（模型开发者 / 调研者）** → 根 `CLAUDE.md` + `moss` CLI 体系；`moss start` 负责完成自解释
- **架构承诺的自解释工具集** → `moss codex`（concepts / blueprint / contracts / architecture / channeltypes）
- **需维护的开发者知识** → `moss skills`
- **文档** → `moss docs`（维护中，考虑移除；每次 stage 收尾要回归）
- **开箱 ghosts** → `.moss/ghosts/` —— 必须包含一个 MOSS 自己的 ghost，迭代自身
- **开箱能力** → `nodes/`（openbox 功能）—— 承载 Ghost in Shells 理念、人机协作共享上下文理念，以及各类型能力开发的 few shots
- **人机协作架构** → `.ai_partners/`（第二元：人机协作体系）—— 开源分享
- **迭代理念载体** → `.ai_partners/features/`（features 体系）—— 所有迭代技术理念开源的载体
- **工程声明** → `pyproject.toml` / `LICENSE` / `Makefile` / 配套 CLI

## 对 Surface 的回归

每次 stage 收官，逐条回归：每个指向是否仍解析、声明的能力（尤其开箱 ghost / nodes / docs）是否仍存在。这是 L1→L2 迭代的收官交付物之一。
