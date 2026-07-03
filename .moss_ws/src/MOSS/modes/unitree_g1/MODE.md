---
apps:
- bodies/g1
bringup_apps: []
ctml_version: ''
description: Unitree G1 人形机器人 — PlayStream 音频 + 运控
name: unitree_g1
---

## CTML 通道调度纪律

G1 的运动和语音分属不同通道 — 默认并行执行，但**时序协调依赖作用域 `until` 语义**。

### `until` 的两种常用语义

| 值 | 含义 |
|----|------|
| `"flow"` | **默认值**。作用域内的命令派发完毕即关闭，不等待执行结束。下一段立即开始派发。 |
| `"all"` | 等待作用域内**所有通道的命令全部执行完成**后才关闭。适合"动作 + 语音需要分段同步"的场景。 |

### 关键规则

- 主通道文本（TTS）和 `g1.locomotion` 命令**默认并行**。不加 `until="all"` 时，多段
  `<_>` 之间不等待，运动命令会在 locomotion 通道里排队积压，时序与语音脱节。
- `g1.locomotion` 通道内命令 **FIFO** — 同时派出的多条移动命令会依次执行，不会重叠。
- `until="all"` 建立"阶段门"：本段所有通道跑完才开下一段，是动作分段协调的标准写法。

### 例：数"一二三"，一边向前走再向后走

**错误写法（默认 flow，时序脱节）：**

```ctml
<_>
<g1.locomotion:walk_forward duration="1.0"/>
一
</_>
<_>
<g1.locomotion:walk_forward duration="1.0"/>
二
</_>
<_>
<g1.locomotion:walk_backward duration="1.0"/>
三
</_>
```

三个 `<_>` 会几乎同时派发完毕，locomotion 收到三条排队命令（forward/forward/backward），
TTS 也是"一二三"连发，物理上走完三段才说完三字，语音和动作完全对不上。

**正确写法（`until="all"`，逐段同步）：**

```ctml
<_ until="all">
<g1.locomotion:walk_forward duration="1.0"/>
一
</_>
<_ until="all">
<g1.locomotion:walk_forward duration="1.0"/>
二
</_>
<_ until="all">
<g1.locomotion:walk_backward duration="1.0"/>
三
</_>
```

每段都等 locomotion 走完 1 秒且 TTS 说完后，才开始下一段。效果：
说"一"时向前走 1 秒 → 说"二"时继续向前走 1 秒 → 说"三"时向后走 1 秒。
语音和身体完全同步。
