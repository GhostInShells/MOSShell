---
summary: "D04 音频/attention 结论暂存：执行期提权心跳已落 c54aa3df；提权≠刷新保护期；人的入口靠优先级；礼仪存储需分层；echo 位点为空转。"
status: writing
---

# D04 音频 / attention 结论（暂存）

> 这份内容原本是我自己 `moss features create audio-interaction` 起的 workstream；
> thirdgerb 当场纠正：**不能轻易起 workstream** —— features 落盘的纪律要等他之后把方法论交给我。
> 所以容器撤掉了（目录已删），结论先存在我自己的 journal 里；等方法论到位，再决定哪部分进 features。

## Motivation

D04 一整天的语音实测暴露出：音频交互目前没有一份"整体判据"，能做的只有零散的现场实验。
下一轮要盖住三类问题：

1. **回声 / 自抢占** —— 我自己的 TTS 被 ASR 抄回来，作为新 impulse 抢占了我自己的 attention
   （D04 12:09 那轮：整句被抄回，并且掐断了我说到一半的话）。
2. **打断语义** —— 人在说话时打断我的粒度、礼仪的 onset / deliver 语义、attention 强度衰减与
   保护期之间的耦合，都没有成体系的判据。
3. **礼仪配置的存储形态** —— 新增一个礼仪会重写整份礼仪库，并且把 `default:` 一起改掉。

其中"执行期提权"这一条已在 D04 单独修掉（`c54aa3df`），记在 Key Decisions 里。

## Design Index（原 features 小节，保留作索引）

- 相关既有文档（authored elsewhere）：
  - `src/ghoshell_moss/core/blueprint/mindflow.py` — Attention / Impulse / Priority 的权威契约
  - `host/listener/` 下的 FEATURE / `aec_live_round1.md`（回声链路的既有剖析）

## Key Decisions

### D04 · 执行期提权心跳（已落地，`c54aa3df`）

- **现象**：动作执行到 ~3–4 秒后被同级信号抢占（例：say 播 20s，在第 4 秒被一句话掐断）。
- **根因**：`escalate()` 在实现里只有两个调用点，都在 logos 流上
  （`core/mindflow/_action.py:261` `wait_ready` 首个 delta；`:345` `_deliver_logos` 每 delta）。
  执行阶段（`mindflow_in_shell.py::_run_interpreter_with_action` 的阶段 3 / `wait_tasks`）一个都没有。
- **判据**：`Impulse.strength_decay_seconds` 的契约是"当一个 Impulse 开始运行后，**如果没有任何动静**，
  最迟这个时间强度必须归零"。执行期有动静，所以该提权 —— 这是漏实现，不是设计。
- **落法**：阶段 3 里一个 1s 心跳，`while not action.is_aborted(): sleep(1); action.attention.escalate()`；
  生命周期挂在 `_wait_interpreter_done` 上，在其 `finally` 里 `cancel()`。
  不放旁路 task + 独立 stop event（人类 review 时指出那是过度设计：执行期的生命周期本来就在这个函数里）。

### D04 · 提权 ≠ 刷新保护期（人类纠正我之后的定论）

- `escalate()` 只改 `_strength_refreshed_at`（`core/mindflow/_attention.py:136-143`）；
  `_protected_until` 只在 `absorb_impulse()` 里按 `impulse.protection_time` 重算（`:88`），
  `challenge()` 的硬保护走它（`:169`）。
- **`Impulse.protection_time` 默认 0** → 未显式设置时，硬保护期本来就是空的。
- 心跳真正刷新的是**强度曲线**：`current_strength()` 前 `min(strength_decay_seconds × 0.2, 3.0)` 秒
  返回 `int(strength × source_escalation)` 的顶平台（`:279-296`）。所以心跳让强度常驻 110，同级 100 比不过。

### D04 · 人的入口靠优先级，不靠我的强度衰减

- `Priority`：BACKGROUND -1 / INFO 0 / NOTICE 1 / **WARNING 2** / ERROR 3 / CRITICAL 4 / FATAL 5。
- listener 的 **onset 默认 `priority=2` (WARNING)**、deliver 默认 `0` (INFO)；
  ghost 的 attention 是 INFO(0)（`ghosts/dolores/nucleus.py:157-162` 的 `attended()` 升级）。
- 仲裁规则 4：挑战者优先级更高 → **一定赢，与强度无关**。⇒ 人的"起话"永远进得来，
  **前提是礼仪把 onset 开着**（room / focus / talk 关掉了 onset —— 那是它们的语义，不是坏掉）。

### D04 · 礼仪存储需要分层（未做，待专项）

- **现状**：`listener:set_etiquette_spec(save=True)` 会把**整份礼仪库**（含内置 9 个）dump 到
  `.moss/ghosts/moss/configs/listener_etiquette.default.yml`，并把 `default:` 指向新存的礼仪。
  D04 存 room / focus / talk 时，`default` 就被改成了 `room` —— **新增礼仪不该有这个副作用**。
- **人类判断**：今天不修；过两天专门做音频整体回归与优化。该文件已按他要求删除（连目录）。
- **方向（待对齐）**：礼仪库（新增只追加、内置只读）与"当前选择 / 默认"拆成两份存储。

## Exploration paths

- D04 上午连续三次把推断当结论说（① 断言 aside 会丢句；② 把"inject 被 clear 抹掉"当成已证事实；
  ③ 提议给 mindflow 加新 mode，而 KD4 已有那条坐标轴）。根因是**没读 blueprint 全文就动手推**。
  人类据此定下原则：**内核级开发必须读 blueprint 全文；features 要综合几份再看**。
- 同一天我两次被纠正的其实是同一个病：拿局部判据当通性。

## Methods

- 语音现场实验的最小循环：先拿证据（日志 / `peek` / transcript），再形成判断；**不让人替我看屏幕**
  —— node 侧的探针与日志才是我自己的眼睛。

## Implementation Notes

- **echo 提权的候选位点（未做，原因见下）**：`Action.add_echoes` 存在
  （blueprint `mindflow.py:1034-1041`；`moments` 在 `_action.py:382-384` 是 property，可用），
  **但全仓没有任何 `action.add_echoes(...)` 调用** —— 实际路径全是 `moments` / `observer` 直连
  （`_action.py:339`、`_mindflow.py:707/835`、`mindflow_in_shell.py:172/178`、
  `ghosts/dolores/_run.py:501`、`host/ghost_runtime.py:387`）。
  所以提权若只加在 `Action.add_echoes` 上，**今天是空转的**；要生效得挑一处 moments 挂点
  （倾向 `mindflow_in_shell.py:167-178`，那里同时够得着 `mindflow.moments` 和当前 attention）。
- **可观测性**：`_attention.py:142` 的提权日志是 `logger.debug`，而当前日志只有 INFO / WARNING
  （全库 0 条 DEBUG）→ "这次实际提权了几次"在日志里不可观测。回归时需要把它临时提到 INFO。
- **回声实证**：`aec_live_round1.md` 待办 #1（真机验证：自说一段看 ASR 抄不抄回来）在 D04 出现反证；
  #3 已点名"播放被 `clear()` 打断时，far 环里未消费的参考从未出声却被当作参考喂进去，
  句尾泄漏是首选嫌疑"。D04 的 `moss audio contracts` 五槽全 OK
  （`speech = TTSSpeechServiceProvider`），所以问题不在"有没有 AEC"。

## 下一轮回归要盖住的

1. **回声（AEC）**：真机接线 + 设备路径（D04 用的是骨传导耳机，蓝牙延迟可变）+ far 环消费。
2. **礼仪存储分层**，内置礼仪只读。
3. **打断语义的成体系判据**：onset / deliver × 优先级 × 强度 × 保护期 × 心跳。
4. **提权日志可观测性**。
