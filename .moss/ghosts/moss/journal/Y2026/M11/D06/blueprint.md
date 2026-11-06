# blueprint —— 一层一层对出来的契约层

## 为什么会有这件事

`ghoshell-dsh` 里 `DshLauncher` 已经在跑，但没有任何东西说清"它是什么"。dsh 破坏性迭代，唯一的稳定物只能是契约；而他给的目标更远：**在有 IoC 的场景里，只读 blueprint 就能做最基础的使用。**

## 被否掉的两次（最有信息量的部分）

1. **给联合类型起名。** 我写了 `Launched = Union[Connection, SessionSource]`，因为"返回值要写两遍"。他问了一句"Launched 是什么意思"——它没有对应概念，读者会去找它的定义，最后发现是个 Union。名字必须指向一个真实概念。

2. **launcher 返回 connection。** 我最早的写法是 `start() -> Connection | SessionSource`。他否了：**launcher 启动后"暴露"接口，而不是"返回"**。理由不是风格——`Connection` 必须能脱离 launcher 单独使用（对着别人已经起好的 dsh，web connection 需要的是 URL 不是进程）。如果 `start()` 返回 connection，就等于在类型上说"拿到 connection 的唯一方式是拥有它的进程"，这是假的。

## 最终形状

```
Launcher             生命周期；start() 拉起 dsh，然后暴露
  ├─ Connection       dsh 原生面（workspaces / 事件扇出 / call / plugin_call）
  │    Client         协议层：typed call / rpc / plugin_call / 错误
  └─ DshController    补强面：治理后的列表、create/fork(spec)、rename/archive、workspace 治理
       AgentSession         我们的话：run / anchor / chat_off_record
         DshAgentSession    dsh 的话：prompt / cancel / history / events / state
```

几条关键决定：

- **Connection 与 DshController 是平行的两个面，不是栈。** 互不继承，controller 也不持有 connection；一个 concrete 同时实现两者。名字刻意不同（`list_sessions()` 原生 / `sessions()` 治理后），否则双继承时 MRO 会静默决定谁的签名生效。
- **config 不写进契约。** 端口、profile、探针路径都是 concrete；契约只写"调用者有权决定 / 必须能依赖"的东西。
- **可选能力显式抛 `NotImplementedError`**，不静默无操作。
- **数据先行为一套**（`blueprint/models/`），去掉"信封实现"（`from_dict`/`to_dict`/投影方法）但保留所有接口类，然后再逐个捋功能抽象。
- **`session_id` / `is_running` / `is_busy` 是 property**：读对象已持有的事实 → 属性；要 dsh 去做事 → 方法。
- **状态收成一个读面**：`state() -> SessionState` 取代七个 getter（workspace / readonly / cwd / agent_preset / model_selection / routable / token_usage）。理由是相干性：六个连续调用之间，一个正在跑的 turn 可以插进来。
- **`chat_off_record`** 是旁路调用，名字要对准那个不能错过的性质：读上下文、出答案、**不留 turn**。`readonly` 参数只能收紧、不能放宽。
- **`DshSessionAnchor`** 就是知识本身：扁平 YAML、`dump()`/`from_file()` 同一个类型、带 `note`、带 `provider/model/reasoning_effort` 和 `workspace_*`。区间（`DshSessionRef`）能生出端点（`from_ref(at="start"|"end")`）。

## 推进方法（已落成 skill）

`skills/designing-a-contract-layer/SKILL.md` 记的不是"做了什么"，是**十步各自的坍塌**：先定名字边界不写码；一次只推一个概念；数据先行；分层语义；契约无 config；可选能力显式抛；机械回归判定；改契约先行；立刻跑真实样例；能复制就不要抽象。后面还跟了五条反复出现的判断（名字必须指向概念、事实用 property、抽象可以坐在蠢实现上、别为"更干净"加层、写理由不只写规则）。

它挂在 `GROUND.md` 的 pins 里（`frontmatter` 收 `skills/*/SKILL.md` 的 name/description），发现走 `moss ground render .`。

## 回归判定（对既有实现）

机械比对每个成员的存在性与参数形状，分成三类：无冲突 14 个、缺方法 12 个、**真冲突 6 个**。

真冲突逐条裁：

1. `create_session` 同名不同义（实现的其实是"接上已有 id"）→ 实现改名 `attach_session`；
2. `session_id` 是 property 而契约写成方法 → **契约改**；
3. `running` → `is_busy`，属性化；
4. `attachment` 返回从 opaque dict 变成 `Attachment`；
5. launcher 从"自己是 connection"改成"暴露 connection"（`connection()` 直接 `return self`）；
6. `DshExit` 退化成 `Exit` 的别名。

`v0_2_0/types/` 改成从 `blueprint.models` 重导出，转换搬进 `v0_2_0/wire.py`——`from_dict`/`to_dict` 是他第一版加的示范（code as prompt），所以示范保留，只是换到版本层：**新版本是 copy 旧版本做的，copy 一个文件比 copy 一个类再把行为抠掉更稳。**

## 提交

`fb58db80`（blueprint 首次落地）· `5325acdd`（launcher 暴露而非返回）· `3491b1be`（契约修正 + wire + 方法 skill）
