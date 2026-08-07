---
created: 2026-07-26
depends: []
description: 解释器级宏展开：command 返回值 (Macro | str) 标记为 macro 时，在解释循环中原位解析为 command
  tokens 注入，形成时序上的 CTML 展开。取代仅主轨可用的 CommandStackResult 机制。
milestone: null
priority: P1
status: in-progress
status_note: |-
  design converged in three dialogue rounds (07-25/26 + 08-06). 08-06: naming settled
  as macro, scope-wrap dropped, Macro object return protocol, macro_id/parent_macro_id disambiguation,
  depth cap 100, executed_logos excludes macro self-tokens, Macro-store ChannelModule as verifier
title: Macro Expansion (Logos)
updated: '2026-08-06'
---

# Logos Expansion

> Use `moss features set-status logos-expansion <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

宏机制（一个 command 的返回值被解释为新的 CTML 并在调用点展开执行）此前有一版
执行层实现：`CommandStackResult`（见 `core/ctml/shell/primitives/loop.py`）。
它已验证嵌套规则可行，但暴露了根本局限，三条局限同一个根因——
**展开的本质是解析行为，却被放在了执行层做**：

1. 展开需要从 ctx 取 IoC 再取 shell，重新解析的起点只能是主轨 → **只支持主轨**；
2. 非主轨 channel 很可能是远程 channel，不在解释器进程内，拿不到 shell → **远程 channel 做不了**;
3. 子轨用 shell 解析 ctml 流时无法知道自己的父节点是什么 → **没有"自己在树上的位置"**；
4. 展开发生在执行层，解释器的 Interpretation 记录不到展开产物 → **CommandA 生成的
   CommandB 不出现在解释历史中**。

解释器版本把展开放回它本来属于的层：解析能力（parser、channel 树拓扑、当前节点位置）
天然只在解释器一侧；而 logos command 唯一需要跨进程传输的是一个字符串——字符串是
唯一天然跨进程的协议。远程 channel 只负责返回字符串，展开发生在解释器本地。

这也是"CTML 展开"设计的原始真版本，此前因风险高、须定义为协议而推迟。现在基建
（scope 语法、element 树、大量 v1 单测）已就位，可以试错。

**收益**（为什么值得做）：

1. **程序性记忆的基座** — channel 把成功执行过的 CTML 序列固化成命名 command，
   重复动作序列结晶成单 token。"习惯形成"在 Shell 层实现，而非 Agent 层 prompt 模拟。
   给任意子树提供学习和记忆能力。
2. **CTML 成为合法的 command 实现语言** — command 可用 Python 写，也可用已有
   command 之上的 CTML 写，模型无法区分。CTML 从"模型输出格式"升格为可存储、
   可 diff、可热替换的数据（Transformative 原则的 "stored CTML"）。
3. **分形 Shell 策略** — 嵌套 shell 封装成 command（如 VLA，外部不见细节），对它
   发 prompt，它吐 logos 流，logos 流以挂载点为根解释执行。shell 嵌套 shell。
4. **token/延迟压缩** — 展开以机器速度发生，不占模型输出带宽；展开内容确定、可测，
   比模型重新生成同一段 CTML 可靠。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`
- 讨论轨迹：2026-07-25/26 与 Claude (Fable 5) 的两轮碰撞收敛出本设计。

## Key Decisions

### D1. Token 注入，而非 task 重入 (2026-07-26)

展开实现为 `parse_tokens_to_command_tasks` 循环内的 **token 流变换**，不是
CommandStackResult 式的 task 层结构：

```
循环拉取 token → parser.on_token → tasks → task_callback 照常发出
  ↓ 命中 logos task：
  1. 停止从 queue 拉取（天然的——循环自己不 get）
  2. await 该 task → 拿到字符串 S
  3. 以宏所在 channel 为根解析 S → token 列表（失败 → 宏 task observe 失败，不注入）
  4. 合成 <chan:_> + 展开 tokens + </chan:_>，逐个喂给同一个 parser.on_token
  5. 恢复从 queue 拉取
```

**拒绝的方案**：
- *task 层塞闭合内（CommandStackResult 路线）* — 见 Motivation 中四条局限。
- *interpreter 级 with-scope 生命周期对象 / elements 暴露 with-scope API* —
  变重且重复造轮子：`elements.py` 的 ScopeEnterTask/ScopeExitTask 已存在，
  `CommandWithoutDeltaArgElement._deliver_self` 已在自动收发 scope 开闭。
  结构应在 **token 层**用现有 scope 记号表达，不在 task 层新造抽象。

由此免费获得：scope 嵌套合法性校验（"只能自己和子 channel"恰好就是现有 scope
嵌套规则——只能 target 当前 channel 或后代）、取消传播、timeout、Interpretation
记录。递归重入消失：展开产物再含 logos command，就是同一平坦循环再次命中，
work-queue 模式。

### D2. 接缝：Interpreter 暴露 create_scoped_logos (2026-07-26)

```python
def create_scoped_logos(
    self,
    channel: ChannelFullPath,
    logos: str,                      # 协议签名预留 str | AsyncIterable[str]
    *,
    lineage: str = "",               # 合成 stream_id 血统链，深度控制用
) -> Iterable[CommandToken]:
    """解析失败 raise InterpretError，由调用方归属为宏 task 的 observe 失败"""
```

理由：Interpreter 抽象只关心 logos，与 CTML 方言无关。基础循环的契约收敛为
"命中 logos task → await → create_scoped_logos → token 流喂回 parser"，循环
只见 CommandToken；换根、scope 记号、语法全是 CTML v1 实现的私事。CTML v2
换语法时循环不改。scope 包裹、换根解析、预校验在函数内部完成；深度上限计数
放调用侧（调度策略，非方言知识）。

### D3. 换根解析实现宏卫生，不做字符串前缀重写 (2026-07-26)

嵌套 parser 以宏所在 channel 为根解析返回串：`<say/>` 直接解析为宏 channel 的
command。"只能自己和子 channel"变成自然约束——换根后根本无法表达祖先和
sibling，解析出界即宏失败。宏定义天然用相对名，可跨 channel 复用。分形 shell
场景下，换根恰好把内层 shell 的命名空间对齐到挂载点——约束、卫生、分形嵌套
三件事同一机制解决。

### D4. Sibling splice + scope 包裹；错误归属分两层 (2026-07-26)

展开是宏 task 完成后紧随其后的兄弟 scope，不塞进宏自身闭合：

- **解析失败归宏** — 注入前预校验拦截，宏 task observe 失败，不触发全局
  parse error（模型视角：一个 command 出错了）；
- **运行时失败不归宏** — 展开产物是普通命令，运行失败按普通规则处理。
  宏只对"返回的代码合法"负责（函数返回的代码出错不算函数出错）。

宏 task 的 occupy 在返回字符串时即释放。不需要延伸到展开完成——因为注入发生在
"持有 queue"期间，展开 token 在流序上必然先于模型后续 token 进入 parser，
channel FIFO occupy 保证已入队命令无法插入展开内容中间。**插队在流序上不可能**。

### D5. 时序语义：解释器停顿 = 模型以机器速度 inline 输出 (2026-07-25)

CTML 第一原则 "emission order IS your schedule"。解释器阻塞在展开点，等价于
模型自己 inline 输出这段 CTML 时后续 token 的天然延迟。现有推论（"想并行就
先发子命令"）对宏自动生效，无需新规则。

代价写进协议纪律：logos command 执行时长 = 全局 dispatch 停顿时长。**logos
command 是"解析期计算"，应当快（检索/模板/状态查询），慢逻辑放进展开产物里执行**
（Lisp macroexpansion 纪律）。

`blocking=False` 与 logos 组合无需禁止：宏阻塞在解释器时，blocking=False 仅
意味着它不被在执行的阻塞命令阻塞；后续命令必然被阻塞。blocking True/False 都
occupy channel，事实上的锁。展开位置由 token 流序唯一决定，与 blocking 无关。

### D6. v1 非流式；流式留在协议签名 (2026-07-26)

v1 只支持 `str` 返回：整段解析完成预校验后注入。流式（`AsyncIterable[str]`）
与预校验冲突（边流边注入则解析错误变成 scope 内失败），且需要 Matrix 支持
跨进程返回值流（现有 chunks__/ctml__ 是入参流）。流式的真实用户故事是分形
shell（内层 agent 边思考边吐 logos），留作扩展位。

### D7. 命名：CommandMeta.logos: bool（已定，人类确认 2026-07-26）

`logos` vs `macro`：macro 是编译器理论借词且借得不准——经典宏在解析期展开，
这里是运行期执行产出 logos，更接近 eval/quotation。`logos` 是项目本体论词汇
（CTML 元规则已定义 "Your CTML stream is called logos"；Interpreter 只关心
logos 不关心 CTML）。分形 shell 场景：内嵌 agent 的 command 说 `logos=True`
（我吐 logos）准确，说 `macro=True` 别扭。macro 唯一优势是外部第一眼熟悉度，
但会误导预期为解析期展开。

### D8. 对模型默认不透明；Interpretation 双层记录 (2026-07-25)

呈现策略与记录分离：Interpretation 记录双层（debug/replay/结晶化用），回灌
模型上下文默认不呈现展开（压缩收益——程序性记忆的意义就是不用重读展开）。
meta 上留呈现 flag。注意 `Interpretation.executed_inputs` 按 task 成功追加
`task.tokens`——展开 task 的 tokens 自然出现在 `executed_logos()` 中，免费
得到摊平后的可重放 CTML，闭合学习回路：成功宏运行留下摊平轨迹 → 存储 →
结晶成新宏。

## 二次收敛 (2026-08-06)

> 人类工程师 + 模型第三轮碰撞后的收敛结论。以下条目覆盖/替换上文 D1-D8 中的对应决策。

### D9. 命名：logos → macro（`CommandMeta.macro: bool`）

D7 定名 logos 被推翻。人类确认改名为 **macro**——对齐解释器语义更直观，四件套命令名即 macro 家族。`create_scoped_logos` 相应改名 `create_scoped_macro`。项目本体论中 logos 仍指"模型输出的流"，但宏机制这一特定能力用 macro。workstream 目录名 `logos-expansion` 保留（历史轨迹），术语在文档/代码内统一为 macro。

### D10. 展开不套 scope：展开 token 是平级普通 task

D1/D4 的 `<chan:_>` scope 包裹取消。展开 token 直接喂回 parser，作为普通命令 task——更像"宏展开为 body"：

- 自闭合宏 task 在 END 时交付、解析树在父级 → 展开是真兄弟；
- 带内容/子命令的宏 task 提前交付、解析树在宏 element 内 → 展开落成宏的 body（降级语义，仍可执行）。
- 裸 `<say/>` 名字解析靠**换根**：嵌套 parser 加 `root_chan=宏所在channel`（`chan = chan or root_chan`，顶层 `.` 前缀按 root_chan 相对解析）。
- D3 的"只能自己和子 channel"解析期约束放松为两层：
  - **(i) 基态自由形态**：出界合法性由 channel scope 运行时检查兜底；
  - **(ii) 解析优化**：注入前对 token 列表预校验，任何 chan 非宏 channel 或后代 → 显式报错"你不该在这里用 ctml"。

### D11. Macro 对象：返回值协议 `Macro | str`

- 定义 `Macro` 对象作为递归传递物。宏命令返回 `Macro` 或 `str`。
- `CommandTaskResult.new_from(any)` 接管 `resolve` 的裸实例化（原 1534 行）：value 为 Macro → `result = macro.result`；为 str → 原样。
- `Macro.result` 为空 → 模型不显示宏命令的返回值。
- 展开侧只读 `task.result()` 字符串，不感知 Macro 对象——**Macro 是纯返回值协议**，展开逻辑不关心返回值怎么包装。
- 宏的最佳形态是 python f-string + 记忆（检索/模板/状态查询）。复杂逻辑进宏没有可执行路径，第二轮调用 token 开销反而更高。

### D12. call_id 撞车消歧：`CommandTask.macro_id` / `parent_macro_id`

模型输出的 `<a:A _cid="1"/>` 与宏展开的同名 `<a:A _cid="1"/>` 都有返回值时，模型无法区分。消歧：

- `CommandTask` 新增 `macro_id: int | None` 与 `parent_macro_id: int | None`。
- macro_id 是**单次解释的自增计数**（解析循环内局部计数器，随解释天然重置）。递归时新 batch 得新 id，`parent_macro_id` = 产生它的宏命令自身的 macro_id。
- 渲染到消息面：展开 task 的 caller 形如 `macro:{id}#{chan}:{name}:{caller}`。现阶段用 `macro="1"` 标记自身宏身份——token 测试先跑通，渲染打磨后置。
- 回溯线**不放 task.context**——放 task 字段 + CommandTaskResult 渲染。

### D13. executed_logos 排除宏自身 tokens

D8 的"免费摊平轨迹"毒化结晶化：轨迹含 `<macro/>`，存成新宏 body 会自指。改为：

- `Interpretation.on_done_task` 对 `meta.macro` 的成功 task **跳过 tokens 追加**；
- executed_logos 只含展开 task 的 tokens → 真"摊平后的可重放 CTML"，可直接结晶。

### D14. 递归深度上限 100，不做横向总量

- 深度 = 展开嵌套层数（= parent_macro_id 链长），解析循环显式维护整数。
- 默认上限 **100**，对齐 loop 原语——loop 用宏实现时，N 次迭代即深度 N。
- 自引用宏靠深度上限兜底（真无限递归跑到 100 被掐，同 loop 原语行为）。
- 不做展开 task 总量预算。

### D15. 异常归属

| 场景 | 归属 |
|---|---|
| 宏任务执行失败（非取消） | interpreter error（停解释） |
| S 解析失败（预校验拦截） | interpreter error（**覆盖 D4** 的"归宏 observe"） |
| 递归超限 | interpreter error（同解析失败） |
| 展开产物运行时失败 | 普通命令失败（D4 保留） |
| 取消（close/clear/打断） | 非 interpreter error，跳过展开 |

### D16. 验证载体：Macro 存储 ChannelModule

实现一个 ChannelModule 作为端到端 dogfood + 程序性记忆基座的第一个消费者：

- 四命令：`macro(label)`（macro flag，返回存储的 ctml） / `macro_save(label, desc, ctml)` / `macro_read(label)` / `macro_list`。
- 入参 `dir: Path | None`；`dir=None` 默认**内存存储**——channels 体系动态，落盘宏在别的 runtime 引用不到当时存在的 channel。
- 绑定 main_channel（PrimeChannel），用 `ctml_shell_test` 直接单测。
- `macro(label)` 与 inline 写入等价构成 oracle（上文 D 测试策略）。

### D17. 工作模式

- 不 worktree，直接在当前目录做。字段/对象层无破坏性；主循环层每帧 review、结对编程。
- 人类同步开工部分字段/对象层改动，由模型 review。

## Implementation Notes

### 风险面（唯一被打破的性质及逃生通道）

现在 `parse_tokens_to_command_tasks` 是单向数据流（token → task，从不等待执行）。
本 feature 引入唯一的例外：这一个协程 await task 执行结果。分层未被打穿——
sync element 树保持纯粹，线程中 TextTokenParser 保持纯粹，await 落在链路上
唯一能 await 的 async 协程里。需要保证的逃生通道：

1. `interpreter.close()` / `stopped()` 能打断展开点的 await（沿用 0.2s
   wait_for 轮询模式或 wait 响应取消）；
2. scope timeout / clear 中断路径穿透到展开点；
3. 远程 channel 断连不能永久冻结解释——logos command 的 `CommandMeta.timeout`
   建议必填或给默认值；
4. 递归深度上限 + 展开 task 总量预算（loop 原语 100 次上限的同类物），
   自引用宏靠深度上限兜底。lineage 记在合成 token 的 `stream_id` 上
   （顺便解决 cid 冲突）。

死锁分析结论：无真环。展开发生在宏执行之后，展开产物不可能被宏自身等待；
宏等待的 channel 占用来自更早的已入队 task。环只能来自"await 不可取消"，
故上面四条即全部风险面。

### 测试策略：时序等价性是现成 oracle

对任意展开串 S，`<ch:macro/>`（返回 S）必须与 inline 写入换根后的 S 产生相同
task 轨迹。现有 `tests/ghoshell_moss/default/ctml/v1_0/test_ctml_v1.py`
（1629 行）的用例可机械地生成宏版本对照组，属性测试近乎机械。

### 工作量清单

1. `CommandMeta` 加 logos flag（跨进程随 meta 同步，远程 channel 只管返回字符串）；
2. `parse_tokens_to_command_tasks` 循环内 logos 检测 + await + 注入（约一屏代码）；
3. `create_scoped_logos`：换根 TextTokenParser（可能只是给现有 parser 传 root
   参数）+ scope 包裹 + 预校验；
4. 合成 token 的 stream_id lineage + 深度上限；
5. 呈现 flag（合成 token 标 synthetic，Interpretation 分层记录）；
6. 等价性对照测试。

无新抽象、无新协议对象，全是现有记号系统的组合。协议增量：logos 一个字段 +
"返回字符串如何被解释"一段文档。

### 开工纪律

- 本任务 block 解释器主循环等多处关键路径，**必须走 worktree**；
- 进入 worktree 后先 `uv sync`（不带 `--active`）确认绑定本地 `.venv`
  （见 CLAUDE.md worktree 环境隔离条目）。

### 参考代码锚点

- `core/concepts/interpreter.py:700` — `parse_tokens_to_command_tasks`，注入点；
- `core/ctml/elements.py:61,137` — ScopeEnterTask/ScopeExitTask，现成的 scope task；
- `core/ctml/elements.py:693` — `_deliver_self`，element 侧已有的 scope 自动开闭；
- `core/ctml/shell/primitives/loop.py` — 被取代的执行层原型（保留，嵌套规则参考）；
- `core/runtime/_tree_channel_runtime.py` — occupy/FIFO 实现（量大，按需读）。