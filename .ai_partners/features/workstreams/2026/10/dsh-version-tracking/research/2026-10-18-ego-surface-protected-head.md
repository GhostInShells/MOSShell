# ego session 的 surface 头非法：0.2.0 读不出历史会话的根因（2026-10-18）

> 症状入口："升级到 0.2.0 之后历史对话基本拉不出来，还是版本不兼容"。
> 本文是可复跑的事实 + 根因 + 选项，不含决策。判据与取舍归人类。

## 0. 结论（一句话）

**不是版本迁移的锅，是 Dolores 造 session 的形状不合 dsh v4 的不变式**：
`ego/create` 先把 `<memory>` 当 `user/message` append 进 surface，而 `system/message`
要等第一个 step 才由 dsh append —— 于是 surface 的首节点不是 system message。
dsh v4 要求"第一个 surface 节点必须是 system/message 且受保护"（protected first
surface head），因此**任何一次冷读**（memento read / 旁路 note / UI 打开历史）都被拒。
升到 0.2.0 只是把这条校验从"没有"变成"有"。

## 1. 失败面：同一个不变式，两个不同的报错

| 路径 | 校验档位 | 报错 |
|---|---|---|
| v3 存档冷读（要迁移到 v4） | `validation: 'transformed'` | `SessionFormatUnsupportedMigrationError: Session migration from v3 to v4 refuses the transformed artifact: system/message requires a protected first surface head` |
| v4 存档冷读 | `validation: 'current'` | `SessionPersistenceCorruptionError: session "…": stored log is corrupt: SessionFormatError: system/message requires a protected first surface head` |

两者都是 dsh 的 `Relationships.foldSurface`（`dsh-session-format-v3-to-v4/lib/index.js:599`）：

```js
if (event.type === "system/message" && this.surface.length > 0 && this.protectedHead === void 0)
    throw new SessionFormatError("system/message requires a protected first surface head");
```

`protectedHead` 只在"system/message append 到一个空 surface"时被记下。我们让它落到第 11 个
surface 位置，所以判据永远不成立。

## 2. 证据：所有非空 ego session 都是同一个形状

ghost 用的 dsh home 是 `.moss/ghosts/deepseek/.dsh`（不是 `~/.dsh`），session 存在
`sessions/<proj>/<id>/session.vN.jsonl.zstd`。逐个解压统计首节点：

```
625beb61 v4 first=user/message system=seq10 surface=222
d188c58a v3 first=user/message system=seq10 surface=94
aca03764 v3 first=user/message system=seq10 surface=27
b22d1902 v4 first=user/message system=seq10 surface=19
fb637665 v4 first=user/message system=seq10 surface=9
7a41a48e v4 first=user/message system=seq10 surface=2
7607997d v4 first=user/message system=seq10 surface=226   ← 当前这个会话
...（v3 的 1cd4626f / 4531da64 / 194e33bc / 0a0e5f55 / 1846df24 / ed786558 同形）
75b1e161 v4 first=(none) surface=0                          ← 唯一可读：空会话
```

**规律：凡是跑过 turn 的 ego session，首节点都是 `user/message`，`system/message` 恒定在
seq 10。surface 为空（没跑过）的才可读。** 这不是个别存档坏了，是造 session 的代码如此。

事件原文（`1cd4626f`，v3）：

```
seq 5  user/message  surfaceOp=append  {"content":[{"type":"text","text":"<memory>\n<ground>…"}]}
seq 6  agent/inbox/spliced  (启动输入)
seq 7  turn/start   seq 8 inbox 撤回   seq 9 step/start
seq 10 system/message surfaceOp=append (MOSS meta instruction)
seq 11 user/message  ← <cognition_epoch> baseline
seq 12 user/message  ← <moment>
```

## 3. 为什么会这样（代码位置）

`ghosts/dolores/dsh_plugin/moss-dolores-ghost-plugin.ts` 的 `DOLORES_EGO_CREATE`：

```ts
// 4. 注入 ghost.memory 上下文 — messages → user/message (surfaceOp append).
for (const msg of messages) {
  handle.agent.session.append('user/message', createUserMessage({...}), { surfaceOp: 'append' })
}
```

而在 dsh 里，`system/message` 是 agent-loop 在第一个 step 用 `renderPrompt(assemble())`
现算现 append 的（日志里 seq 10 带着 `turn:1, step:1`）。所以 create 时刻 surface 里只能有
我们塞的 user message —— 顺序被结构决定，不是偶发。

**dsh 侧的对称性问题**：`session.append('user/message', {surfaceOp:'append'})` 这个公开 API
允许构造出它自己的校验器必然拒绝的日志。写进去不管，读出来报 corrupt。这条值得上游报，
但修不修不在我们手里。

## 4. 影响面

- memento 主线 11 条 commit 指向 8 个 session，其中 **10 条指向不可读的 session**（只有指向
  空会话 `75b1e161` 的那条能读）。→ `ghost.memento read/chat` 对历史基本全废。
- 旁路 note 的冷读同样走 `sessionPersistence.open`，因此**新 commit 的 note 也补不出来**；
  `commit_notes.jsonl` 里 3 条 `run failed: …corrupt: system/message requires a protected
  first surface head` 是同一根因。
- UI 打开历史会话同理（列表只读 header，所以列表看着正常）。
- 活 session 不受影响：live 读走 `agent.session.snapshotEvents()`，不过校验。所以 ghost 现在
  还能正常跑，只是"回忆"断了。

## 5. 前向修复（新会话不再带病）

**A. memory 不再当 surface 节点，改走 system prompt 组装。**
memory 进入 protected head，天然抗 compaction —— 顺带把 `dsh-version-tracking` 里记的
"protected head 不对称"（ghost 的协作意图在可压缩层）变成正解。代价：memory 在一个 session
内不可变，且与 epoch recap 的二次注入可能重复。

**B. memory 推迟到第一个 turn 的 inbox 输入里（推荐，改动最小）。**
create 只记下 memory，不 append；第一个 `thinking/enter` 时把 `<memory>` 作为该轮的第一条
user 输入送进去。日志里 seq 11+ 的 moment 就是这么落进 surface 的 —— 顺序会自然变成
`system → memory → moment`。语义与现在完全一致（memory 仍在可压缩层，compact 后由 epoch
recap 重发），只是晚一步落盘。

> 两条都要求**先改代码再开新会话**，否则每开一次就多一份读不出来的存档。

## 6. 历史恢复 —— 已裁：不做（2026-10-18）

> **这两个选项都不做。** 人类裁决：只用 dsh 官方约束，官方没有兼容读接口就放弃；session log
> 是三层权威（日记 → memento → session log）的最底层，本来就可丢弃。保留原文见下方，作判例留档。
> 实际落地是读侧抛错 + 旁路 skip（见 FEATURE 的判例节）。

**R1. 旁路容错读（只救内容，不做合法 session）。**
`loadSourceEvents` 在 `persistence.open` 抛 `SessionFormatUnsupportedError` /
`SessionPersistenceCorruptionError` 时，回落"自己解压 `session.vN.jsonl.zstd` + 直接映射成
`SessionEvent`"。Node 24 有 `zlib.zstdDecompress`，不需要新依赖。memento read/chat 立刻复活，
不碰 dsh 的校验。局限：拿不到 dsh 的 surface 语义（compact 后的 replace 要自己折叠或不折叠），
且不可 resume。

**R2. 一次性重写存档成合法 v4。**
把 `system/message` 提到 surface 首位并重排 seq / 修 `surfaceOp.startSeq/endSeq` 与
`request/header.headerSeq` 引用。能换来"UI 里能打开、能 resume"。风险：重排是语义动作
（system 提前到 memory 之前），必须逐条校验再落盘，且要留原文件。

## 7. 未验证 / 未解释

- **没有实跑** `persistence.open` 的原路径，结论来自"用同一套 format catalog 按两种校验档位
  离线重放"。复跑命令见 §8，可自证。
- `~/.dsh`（个人 home，不是 ghost home）另有一类失败，**与本文无关**：4 个 subagent 子会话被
  v0→v1 拒 `subagent/descriptor uses unsupported descriptor version 2`，其余 26 个能迁。
- 本次会话开头，我在 moment 里看到过一行裸文本（无 input 标签）就是这个报错串的原文
  （`…(raw log:` 之后被截断）。**注入路径没查出来**：不在 memento note 里，不在任何日志里，
  也不像 `<command>` 结果。存疑。已知的是它带着绝对路径 —— 现在的读侧报错会把
  `(raw log: …)` 剥掉，这条泄漏路径算是堵上了。

## 8. 复跑

```bash
# 1) 列出 ghost dsh home 里每个 session 的首 surface 节点
python3 - <<'PY'
import json,os,subprocess
S=".moss/ghosts/deepseek/.dsh/sessions"
for dp,_,fs in os.walk(S):
    zs=sorted(f for f in fs if f.endswith(".jsonl.zstd"))
    if not zs: continue
    rows=[json.loads(l) for l in subprocess.run(["zstd","-dc",os.path.join(dp,zs[-1])],
          capture_output=True).stdout.decode().splitlines() if l.strip()]
    surf=[e for e in rows[1:] if "surfaceOp" in e]
    print(os.path.basename(dp)[:38], "v%s"%rows[0].get("version"),
          "first=%s"%(surf[0]["type"] if surf else "(none)"), "surface=%d"%len(surf))
PY

# 2) 用装好的 0.2.0 catalog 按两种校验档位重放（read-only，不改盘）
#    见 /tmp/dshmig3.mjs：createSessionFormatCatalogWithChildren([]) →
#    createRestore(header,{recovery:'recoverable',validation:'current'|'transformed'})
node /tmp/dshmig3.mjs .moss/ghosts/deepseek/.dsh/sessions/--Users-...--/<id> current
```

## 9. 决策（2026-10-18，人类裁）：H1

**head 不变，可变量不进 head。**

- head = 会话不变量：MOSS meta instruction + identity/persona + preset 工具面。
- memory / epoch baseline / recap 一律留在可压缩层，以 `user/message` 出现，每次开新 epoch
  由 recap 槽重发（既有通道，不新增机制）。
- compact 只折叠 head 之后的节点，head 原地保留。

代价与理由（讨论记录）：H2（memory 进 head + head 自替换）也能过校验 —— 实测在 turn/step 内
发一条 `system/message{surfaceOp:{op:'replace',startSeq:head,endSeq:head}}` 是合法的，head 指针
会移到新 seq。但 `system/message ∈ STEP_EVENT_TYPES` ⇒ head 只能在一轮之内更新，而 compact 发生在
idle，压根够不着它；再说 protected head 的价值就在"它不会变"，一旦每轮都变，保护只剩"位置不变"。

### 由此产生的两条硬约束（实测）

1. **compact 的 replace 范围不能含 head。** 否则
   `surface replacement cannot shadow the protected system head`。
   —— 这就是这次改动的第二处（原范围 `seq < cut` 改成 `seq < cut && seq !== head`）。
2. **dsh 的格式里没有"清空"。** `surfaceOp` 只有 `"append"` 与 `{op:'replace',startSeq,endSeq}`；
   "全部清空再 append"只能表达成一次覆盖全范围的 replace，而那个范围不能含 head。
   所以 H1 下 compact 的形状是唯一确定的：`replace (head, cut] → 一条说明`，并带全量
   `sourceEventSeqs`。**"清空"与"replace"对模型等价，对日志不等价** —— 日志要留"谁被谁遮蔽"的
   血统，那是 memento 与负知识的真相源。

### 落地（本次改动）

`ghosts/dolores/dsh_plugin/moss-dolores-ghost-plugin.ts`：

- `ego/create` 不再把 memory 写成 surface 节点，改为暂存 `pendingMemory`；
- `thinking/enter` 把 memory 作为第一帧 moment 消息投递（pre-step 以 `enter` 投递，落盘恒在
  `system/message` 之后）；
- `compact` 与 `compactLossily` 的折叠范围排除 surface 头。
