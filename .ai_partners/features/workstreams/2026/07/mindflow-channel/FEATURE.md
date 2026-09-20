---
created: 2026-07-23
depends: []
description: Mindflow 反身控制 Channel — 将 mindflow 从 opaque 调度器变为 ghost 可感知、 可操纵的透明面.
  自解释 + 注意力管理 + 优先级干预 + nucleus claim.
milestone: 0.1.0
priority: P1
status: completed
status_note: 'v3 收口: claim_impulse 上 Mindflow ABC, 命令面去 instruction/notice/context, 收敛为
  nuclei/peek/claim/specification + set-* 治理命令; InputSignalNucleus 补 NAME 常量.'
title: Mindflow Channel
updated: '2026-09-21'
---

# Mindflow Channel

> Use `moss features set-status mindflow-channel <status> -m "note"` to update state.

## Motivation

Mindflow 当前没有 channel. 它的上下文（nucleus 状态、当前 attention、perspective）
是强行拼装到 perspective func 里的 — ghost 看不到自己的感知系统内部状态.
这是一个"感知幽闭"问题: ghost 知道自己有感官, 有消息在排队, 但感官对它是黑箱,
只能等 signal 够强了自行破门而入.

Mindflow channel 是 Mindflow 的反身控制面. 做完后:
- Mindflow 的自解释、注意力状态、优先级调整、nucleus pull 全部通过 channel 体系暴露
- ghost 获得一个关于自身注意力的**可操作模型** — 不只是"收到 impulse → 反应",
  而是"我看见我的感知系统里有什么、我的注意力被什么占据、我可以选择改变它"
- Mindflow 从 push-only 变为 push + pull 双路径

这是 Dolores ghost channel 认知场的核心内容: ghost 通过自己的 channel 看见
并调控自身感知状态.

## Capability Surface

四层递进:

### 1. 自解释 (Self-Explanation)

Channel `instruction` 动态反映当前 Nucleus 拓扑. 每个 NucleusMeta 贡献一行
自描述 (name + signals + description). 模型一瞥就知道自己有哪些感官、各自在
监听什么信号, 不需要系统层单独拼 perspective.

输出形状类似:
```
mindflow nuclei:
  vision (NOTICE): 监听 visual_frame, camera_status
  audio  (NOTICE): 监听 asr_partial, asr_final
  system (BACKGROUND): 监听 health_check, config_change
```

### 2. 注意力管理 (Attention Management)

暴露当前 attention 状态, ghost 可主动操作:

- **状态查询**: 当前谁占着 attention、什么 priority、剩余 protection_time
- **释放**: 主动结束当前 attention (release)
- **降级**: 降低当前 attention priority, 让外部信号更容易抢占
- **屏蔽**: 临时 suppress 特定 nucleus 的 impulse, 不让它参与 challenge

### 3. 优先级干预 (Priority Intervention)

基于 Mindflow 已预留的接口 (`set_signal_priority_bar` / `set_impulse_priority_bar`)
做细粒度操作:

- **per-nucleus 提权/降权**: 对特定 nucleus 的 priority +1/-1
- **全局水位**: 抬高最低可挑战门槛 (如 NOTICE 为最低, BACKGROUND 直接丢弃)
- **临时加权**: 对特定 signal name 临时加权, 使匹配的 signal 更容易抢占

mindflow 仲裁时取修改后的权重做 challenge 校验. 这些修改是 ghost 的运行时行为,
不污染 nucleus 本身的声明式 priority.

### 4. Nucleus Pull (主动拿 Impulse)

不通过 mindflow challenge 管道, 而是 ghost 主动从 nucleus 里拿高优 impulse:

```
channel:mindflow pull-nucleus --name vision
  → 返回 nucleus 当前最高优 impulse
  → 通过 command 返回值 → observe 体系直接递给 ghost
  → 同时反身更新当前 attention 状态 (标记已阅/消费)
```

这条路径不走 push (Signal → Nucleus → Impulse → Challenge → Attention).
低中优信号不参与 challenge 竞争, 安静排队等 ghost 来取. FATAL/CRITICAL
依然走 push 抢断. 两条路径各管各的优先级带.

## Key Decisions

- **Mindflow.as_channel() + Nucleus.as_channel() 已预留接口.** 实现时优先走
  这两个预留点, 不做新的抽象切口.
- **自解释走 instruction, 不走 context_messages.** instruction 是静态面 (每
  refresh 更新), context_messages 是动态面 (每帧更新). nucleus 拓扑变更是低频
  事件, 走 instruction 即可.
- **优先级干预是运行时 overlay, 不修改 nucleus 声明.** nucleus 的 priority
  是声明式基准, ghost 的提权/降权是运行时 overlay. mindflow 仲裁时合并计算.
- **Pull 路径复用 observe 体系, 不新造通讯协议.** command 返回值 →
  observe → 下一帧 context, 链路已存在.
- **首版 scope: 自解释 + 注意力管理 + 优先级干预.** Pull 路径实现复杂度
  更高 (需要定义 nucleus 的 poll 接口), 按需后做.

## Open Problems

- **Nucleus poll 接口** — 当前 Nucleus 只有 `peek()` (不消费) 和 `pop_impulse()`
  (mindflow 回调). Pull 路径需要 "poll + consume" 语义. 是否在 Nucleus ABC 加
  `poll()`, 还是用 `peek()` + 独立 ack 机制, 待定.
- **优先级 overlay 的 persist 范围** — 临时加权是否跨 attention? 是否跨 session?
  需要定义 overlay 生命周期.
- **屏蔽 vs background_notice** — 屏蔽一个 nucleus 和把它设到 BACKGROUND + notify
  有什么区别? 前者是硬切断, 后者是"不抢占但留痕". 两条路径各自适用场景需厘清.

## Implementation Notes

- `Mindflow.as_channel()` 返回一个 MutableChannel, 用 Builder 注册上述命令.
- 每个 Nucleus 的 `as_channel()` (如果返回非 None) 作为 mindflow channel 的
  虚拟子通道, 提供 per-nucleus 粒度的控制.
- 自解释的 instruction 通过 `Builder.instruction()` 注册, 每次 `refresh_meta`
  时重新生成字符串 (从 mindflow.faculties() 取当前 nuclei).
- `set_signal_priority_bar` / `set_impulse_priority_bar` 的默认实现是 noop
  (见 mindflow.py), 需要在 Mindflow 实现中补上.

## v1 实装 (2026-08-28)

首版落地「自解释 + 注意力管理 + 优先级干预」;Nucleus Pull 以 `pull` 命令的简化形态
实现 (peek → attended → 返回 messages), per-nucleus priority overlay 未做.

`build_mindflow_channel(mindflow, *, enable_priority=True, enable_bar=True,
enable_pull=False, enable_red_dot=False)`: 三个注意力机制用 build flag 门控, 开启时
命令 `available()` 通过并配套展示 context, 关闭时命令不可见、context 不带该状态.

| 面 | 载体 | 性质 |
|---|---|---|
| 心智模型 | `instruction` | 静态, 不罗列命令 |
| nucleus 拓扑自解释 | `help` | 动态 (每次 refresh), 拓扑变更自动 diff |
| 可变状态 (bars/attn/红点) | `context_messages` | 动态, 按 flag 门控 |
| 子通道挂载 | `virtual_children` | decorator, 运行时按 running 状态动态挂载 |

命令面: `status` (always_observe=True, 自省) / `set-priority` `set-signal-bar`
`set-impulse-bar` (always_observe=False, 确认语义) / `pull` (try, 100% 不等 next impulse).

关键偏离原计划:

- **priority bar 逻辑没丢** — 重构后仍在 `_mindflow.py` 且被 `add_signal` /
  `_rank_best_impulse_from_nuclei` 消费, 补了 getter + 单测.
- **per-nucleus priority overlay 从未实现过** (原计划提出但无对应 commit), 是独立
  follow-up, 不是被重构丢的.
- **`with_nucleus` 去掉旧 channel import** — 注册 nucleus 不再 `import_channels` 子通道
  (旧设计), 改由 `virtual_children` decorator 运行时按 running 状态动态挂载; 这同时解决
  `BaseMindflow.as_channel()` 懒构建与 `__init__` 里 `_mindflow_channel` 尚未赋值的时序冲突.
- **`BaseMindflow.as_channel()` off-switch 打开** — 从 `return None` 改为返回真 channel,
  使 `MindflowInShell` / `ghost_runtime` 拿到 channel.

## 未来方向: idle 自驱 (self-drive)

> 定案方向, 未实装, 独立 follow-up, 不属本 feature v1.

自驱的最小种子: "one more check then stop" — 闲时自检 N 次 (含间隔), 用 prompt 给自己,
到 0 停.

**职责分离 (关键)**:

- **mindflow 的 `when_idle` 只做一件事**: 闲了 `wait_time` 后**广播一个 idle signal**.
  它真正核心的部分是「signal 发送的等待时间」(idle 阈值) — 这是 lazy 触发, 与 push 通道正交.
- **拦截该 signal 的 nucleus 自己理解 idle、治理行为**: 业务逻辑 (prompt / 次数 / 停止
  条件) 落在 **nucleus channel** (`as_channel`), 不在 mindflow channel.
- **「次数」是 attended 计数, 不是递减计数器**: 复用 `Nucleus.attended(impulse)` 生命周期
  — 每自检 impulse 被 attended 一次记一次, 到 N 停, 不引入新计数机制.

自检 signal 应走低优先带 (notify / 静默), 不抢真实外部信号. count 上限是硬护栏;
「永不安息」(self-check → 思考 → 又 self-check) 的冷却语义留待 follow-up.
## v2 重开 (2026-09-13): gate 元机制 + 反身面收敛

### 为什么重开

v1 的反身面把「注意力治理」直接做成 mindflow channel 的顶层命令 (`set-priority` /
`set-signal-bar` / `set-impulse-bar` / `pull`), 并用构建期 flag (`enable_priority` /
`enable_bar` / `enable_pull` / `enable_red_dot`) 门控。问题:

- 命令面与「感知/思考」的主面混在一起, 模型一上来就看到全部控制能力, 没有渐进式披露。
- `notice` / `context` / `status` 各拼一遍 nucleus 列表, 逻辑重叠。
- 红点机制放在 context 里是错位 — 红点本应是子 channel 的形态。

目标形态: mindflow 的高功能控制面 (注意力治理) 和所有 nucleus channel **都作为
virtual child**; mindflow channel 实例化默认走 gate, 模型不打开的子通道只在 notice
里可见。gate 是 **prime channel 的元机制**, 不是 mindflow 专属。

### 两步走 (硬顺序, step1 完成才进 step2)

**Step 1 — py_channel 落地 gate 元机制**

核心接口开 `gate` flag, 默认 `false` (全仓库现行为零变化)。踩既有 states 机制的路径,
不新造抽象:

- 蓝图 `states_channel.py`: `PrimeChannel` 增抽象访问器 `gate() -> bool`;
  `new_prime_channel(name, description="", gate=False)` 透传。
- `py_channel.py`: `PyChannel(..., gate=False)` → `BaseStateChannel(..., gate=False)`
  存 `_gate`; `StatefulChannelRuntimeImpl` 读 `channel.gate()`, 持
  `_opened_children: set[str] = set()` (**默认全关**)。
- **过滤点唯一**: `virtual_sub_channels()` — gate 为真时只保留名字在 `_opened_children`
  的虚拟子通道, 其余不交给 tree (tree 自动卸载, 见 `runtime/tree.py` `_refresh_structure`)。
- **自动绑命令** (仿 `switch_state`: runtime public 方法 → `PyCommand` → `_own_commands`):
  `mount_channel(name)` / `unmount_channel(name)`, 改 `_opened_children` 后
  `await refresh_metas()` 触发 tree 挂载/卸载。available 受 gate 与目录/打开集合约束。
  **命令名待人类架构师拍板** (要求: 不能用一眼通用的 open/close, 名字要稍稍特化)。
- **notice 自动拼装目录**: `_get_notice()` 在 gate 为真时追加声明的虚拟子通道目录
  (`name (open|closed): description`)。未挂载的子通道无 meta 节点, 模型只能从 notice
  看到它存在 — 这是渐进式披露的信息闭环。
- 目录来源 = `build.virtual_children` 回调声明的集合。

**Step 2 — mindflow channel 重构** (概要, 待 step1 完成后细化)

- 父命令面收敛为 `status` (`always_observe` 自省); `set-priority` / `set-signal-bar` /
  `set-impulse-bar` / `pull` 下移为 virtual child, 进 gate 目录、默认关闭。
- 所有 nucleus channel 作为 gated 目录项。
- 三处重叠渲染合并进基本不变的 `notice`。
- 删 `enable_red_dot` 与 context 红点块; context 其余**不动** (上下文治理策略已变,
  context message 有痛, 不碰)。

### 待人类架构师确认 (1 点)

「states 呈现自动拼装进 notice」的落地方式。现状: `ChannelMeta.states` 由 runtime 填充,
但 prompt 层是**独立 `<states>` 块** (`core/ctml/v1_0/prompts.py` `states_message()`),
被 `tests/.../ctml/v1_0/test_prompts.py` 两条断言锁定为独立 section。

- **A (倾向)**: runtime 把 states 目录 + `Current state: X` 追加进 notice; `meta.states` /
  `current_state` 字段保留 (数据协议不变), 移除 prompts 的独立 `<states>` 渲染以免重复,
  相应改 2 条 prompt 断言。
- **B**: 只做 gate 目录进 notice, states 块完全不动 (纯增量)。

### 边界

- gate 只作用于**声明式**虚拟子通道; 运行时 `add_virtual_channel()` 注入的通道仍直接挂载
  (那是命令的显式副作用, 非目录项)。
- `available()` (谓词驱动整 channel 可见) 与 gate (模型驱动的子通道披露) 不同轴, 不互相实现。
- sustain children (`import_channels`) 不参与 gate, 始终挂载。
- 构建期 flag 与 gate 是不同轴: 前者决定「机制是否进目录」, 后者决定「是否打开」。

### 测试与验证

- `tests/ghoshell_moss/default/core/channels/test_state_channel.py` 追加: `gate=False` 全挂载
  (回归); `gate=True` 初始零挂载 + notice 目录; mount/unmount 生效与命令自动注册;
  gate 且目录为空时命令不可用。
- 若采纳 states 方案的 A, 更新 `test_prompts.py` 两条断言。
- 回归: `pytest tests/ghoshell_moss/default/core/mindflow/ -q` 确认 step1 未扰动静默行为。

### Step 1 完成 (2026-09-13)

gate 元机制已落地, 三项决策已定:

- **states 不删** — 只做 gate 目录进 notice (上文「待确认」取 B, 纯增量)。
- **命令名**: `mount_child` / `unmount_child` (宾语用 child, 不用 channel)。
- **gate 关闭则 notice 无附加内容** — 目录只在 gate 开启时拼装。

实现落点:

- `blueprint/states_channel.py`: `StatefulChannel.gate() -> bool` 默认 `False`;
  `new_prime_channel(..., gate=False)` 透传。放在 `StatefulChannel` 而非 `PrimeChannel`,
  避免 MRO 上遮蔽 `BaseStateChannel` 的实现。
- `core/py_channel.py`: `BaseStateChannel` 存 `_gate`; `StatefulChannelRuntimeImpl` 持
  `_opened_children` (默认空), `virtual_sub_channels()` 在 gate 开启时只保留已打开项,
  `is_dynamic()` 计入 gate 目录; 自动绑 `mount_child`/`unmount_child` (走 `PyCommand`,
  仿 `switch_state`); `_get_notice()` 在 gate 开启时追加目录
  (`- name (open|closed): description`)。

测试: `test_state_channel.py` 追加 6 条 gate 用例, `state_channel` 56 passed。
mindflow/prompts/blueprint 回归通过 (listener 2 条失败是 voice-input-state-machine
在途改动, 与本步无关)。

**Step 2 待开始**: mindflow 控制面下移为 gated virtual children。

### Step 2 进度 (2026-09-13)

gate flag 已贯通到 mindflow 构造面:

- `build_mindflow_channel(..., gate=False)` → `new_prime_channel(..., gate=gate)`。
- `AbsMindflow.__init__(..., gate=False)` 存 `_gate`; `as_channel()` 传 `gate=self._gate`。
- `BaseMindflow.__init__(..., gate=False)` / `new_default_mindflow(..., gate=False)` 透传。

默认 `gate=False` (行为不变); 调用方 `new_default_mindflow(gate=True)` / `BaseMindflow(gate=True)`
即可让 nucleus channel 走渐进式披露。

测试: `test_mindflow_channel.py` 追加 2 条 — gated mindflow 把 listener nucleus channel 收进
notice 目录并可 `mount_child` 挂载; 默认 gate=False 直接挂载。

**待做**: 注意力治理命令 (set-priority / set-signal-bar / set-impulse-bar / pull) 下移为 gated
virtual child; notice 三处重叠合并; 删 enable_red_dot + context 红点块。

### Step 2 拆解完成 (2026-09-13)

注意力治理面已从顶层下移为 gated 虚拟子通道 ``attention``:

- 顶层只剩常驻读面: ``status`` (always_observe 自省) + ``pull`` (enable_pull 门控, 默认关)。
- ``set-priority`` / ``set-signal-bar`` / ``set-impulse-bar`` 移入 ``attention`` 子通道
  (`_build_attention_child`), 由 ``enable_priority`` / ``enable_bar`` 决定注册哪些;
  两者都关则不注册该子通道。
- ``attention`` + 各 running nucleus 的 ``as_channel()`` 一起进 ``virtual_children`` 目录,
  gate 开启时默认关闭、由 mount_child 披露。
- 删 ``enable_red_dot`` 与 context 红点块 (context 其余不动)。
- 命令寻址: CTML 从 ``<mindflow:set-*/>`` 变为 ``<mindflow.attention:set-*/>``。

测试: ``test_mindflow_channel.py`` 改/加覆盖拆解 + gate; ``test_mindflow_channel_ctml.py``
更新寻址到 ``mindflow.attention``。mindflow 360 / channels+ctml+blueprint 379 全绿。

**待做**: notice 三处重叠合并 (notice 现在既列 nuclei, gate 目录也列 gated children)。
### Step 2 回退修正 (2026-09-14): 治理面回父 channel, 不折叠

上一轮把注意力治理下移为 gated ``attention`` 子通道做砸了, 全回退:

- **名字错位**: ``attention`` 是 mindflow 自身持有的状态 (``blueprint/mindflow.py`` 的调度
  单元, Impulse 创建 → 思考/执行结束退出). 一个治理子通道不该占这个词.
- **折叠即弃用**: 注意力治理 (尤其"运行时提升当前注意力") 是 mindflow channel 的**一等能力**.
  一旦折叠进 gate, 模型不会主动 mount, 能力等于不存在. gate 只该折叠"按需展开的细节"
  (各 nucleus 的子通道), 不该折叠 mindflow 自身的控制面.
- **常驻面收敛回父 channel**: ``set-priority`` / ``set-signal-bar`` / ``set-impulse-bar``
  回到父 channel 常驻; 顶层还有 ``status``(当前 attention 自省) + ``nuclei`` + ``pull``(默认关).
- **nucleus 讯息折叠进 ``nuclei`` 方法**: 之前 notice / status / context 三处各列一遍 nuclei,
  收敛成一个 ``nuclei`` 命令.
- **治理状态是温数据, 走 notice**: 状态级变更 (当前 attention / 水位) 不进每帧
  ``context_messages``(热面), 走 notice 随 refresh 差分投递. ``context_messages`` 整个去掉.
- **gate 默认 true**: ``build_mindflow_channel`` / ``AbsMindflow`` / ``BaseMindflow`` /
  ``new_default_mindflow`` 默认 `gate=True`, 但 gate 现在只折叠 nucleus 子通道.

CTML 寻址回 ``<mindflow:set-*/>``. 测试 ``test_mindflow_channel.py`` / ctml / shell_integration
相应改写; mindflow 361 / channels+blueprint 391 全绿.

### Step 2 追加修正 (2026-09-14): 删 pull + bar getter 上接口

- **删 ``pull``**: 原 ``pull`` 命令手搓 `nucleus.attended()` + `attention.absorb_impulse()`,
  把仲裁期运行时方法塞进控制面, 属于在 channel 里独立创作一套感知语义. 这种重要逻辑
  不能散落在 channel, 先删, 后续由人类在 mindflow 接口上重做 pull/poll.
- **bar getter 上接口**: ``signal_priority_bar()`` / ``impulse_priority_bar()`` 原本只挂在
  实现 ``AbsMindflow``, 是纯 hack. 已补进 blueprint ``Mindflow`` ABC (默认 ``BACKGROUND``,
  与 setter 同款 "反身性 channel 准备" 面).

### v3 收口 (2026-09-21): claim 上接口 + 命令面去 instruction/notice/context

把上一轮删掉的 pull 用正确形态重做, 并收敛命令面到「无 instruction / 无 notice / 无 context」。

- **``claim_impulse`` 上 ``Mindflow`` ABC** (权威契约), ``_mindflow.py`` 实现:
  ``peek`` → ``attended`` (物化 stub → full) → ``_pending_frame_impulses`` + ``need_observe``.
  不 ``absorb_impulse`` (不强化当前 attention); 缓冲在 mindflow 层, attention abort 不清它,
  折进下一个 attention 首帧. 返回 ``None`` 当 nucleus 未知 / 未运行 / 无可 claim.
- **命令面收敛** (``build_mindflow_channel``):
  - 删 ``instruction`` (「最好的 instruction 就是源代码」)、删 ``notice``、删 ``status`` /
    ``active attention`` (注意力自解释, 再报一遍是 100% 冗余)、删 ``context``。
  - 常驻命令: ``nuclei`` (拓扑, 不带 message 内容) / ``peek`` (n 个持有单元 + 状态摘要 +
    head 预览, 不带全文) / ``claim`` (消费 → 下一轮读) / ``specification`` (返回 blueprint
    源码模块路径) / ``set-priority`` ``set-signal-bar`` ``set-impulse-bar`` (注意力治理,
    一等能力常驻父 channel)。
  - ``set-priority`` 门控 = ``enable_priority AND mindflow.attention() is not None`` (attention
    活跃才可见); ``set-signal-bar`` / ``set-impulse-bar`` 由 ``enable_bar`` flag 门控。
  - gate 只折叠各 running nucleus 的子通道 (默认开), mindflow 自身控制面不折叠。
- **``InputSignalNucleus`` 补 ``NAME`` 常量** (此前是唯一缺 ``NAME`` 的 nucleus), 并修
  ``InputNucleusMeta.factory`` 的 ``name=self._target_signal`` (signal 名 'input') →
  ``name=self._name`` (nucleus 名) — 原实现会让 factory 产出的 nucleus 命名成 'input',
  ``nuclei()['input_signal_nucleus']`` 查不到。
- **测试**: ``test_claim_impulse.py`` 新增 (mindflow 层, 走标准 ``thinking_loop`` 消费, 用
  ``InputSignalNucleus`` 「输后保留」语义, 不 hack 接口); ``test_mindflow_channel.py`` 收敛
  命令面断言; epoch facade 测试改钉「命令面第零帧交付、不进 recap」(原「facade 不泄漏进
  echo」断言绑死旧静态 instruction 面, 已作废)。mindflow 389 全绿。
