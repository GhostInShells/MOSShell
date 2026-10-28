---
name: ghost-prototypes
description: ghost 原型与 Dolores 的实现代码 — 这一层往下的 GROUND.md 都是随包分发的模板桩，不是项目认知场
---

# Ghost Prototypes

这里是 **ghost 原型**的实现代码（Dolores 及其它）：runtime、channel、测试，以及随包分发的
stubs。它属于能力实现，不属于项目的认知场。

## 这一层的 GROUND.md 为什么存在

两个作用，都很小：

1. **挡穿透**：发现协议在「直接命中 GROUND.md 的目录」处止步（`stop_on_match`），
   所以项目索引不会继续走进 `dolores/stubs/`，把一个模板桩列成项目的一个场。
2. **说法**：读到这里的人应该知道，下面那些 GROUND.md 是**模板**——它们用来初始化一个
   新的 ghost home，不是这个项目的知识。
