# Utterance End Detection — plain-text single-token 0-9 probe

量一个快模型能否对一个 live-transcript 话语**只吐一个 0-9 数字**（无散文），并且
数字落在正确的 band（low 0-3 / mid 4-6 / high 7-9）。

## 与 benchmark 约定的刻意偏离

- **无 `result_type` / `models.py`** —— 结构化输出实际会吐近百 token（tool-call 解码）。
  纯字符串单 token 的成本是 n 输入 token + 2 输出 token。**约束本身才是被测信号**：
  模型是否严格只输出一个数字。
- 解析走 `run.py` 的 `parse()`，不是 pydantic 校验。
- 无 scorer —— 用 `band(score)`（0-3/4-6/7-9）对 `cases.jsonl` 里的 `band` 字段做命中判断。

## 响应契约

| 输出 | 判定 |
|------|------|
| 单个数字 0-9（可带首尾空白） | clean，得分 |
| 含数字但带散文 | 约束违反，数字 salvage 出来参与准确率 |
| 无数字 | garbled，不可用 |

`max_output_tokens=2` 强制模型只吐一个 token（数字本身 1 token，留 1 token 缓冲）。

## Case 形状

`cases.jsonl` 每行 `{label, class, context, input, band}`。`class` 是自文档标签，
runner 不消费。`context` 承载判停礼仪（如"用 over 表示说完"），拼进 `<context>`；
`input` 是待判别话语，拼进 `<input>`。20 例覆盖：

| class | 含义 | band |
|-------|------|------|
| incomplete-filler | 嗯/那个/其实 —— 还在措辞 | low |
| incomplete-dangling | 如果…的话 —— 悬空条件 | low |
| incomplete-hard-cut | 半句硬切 | low |
| incomplete-conjunction | 悬空连接词（因为/然后/because/if） | low |
| complete-statement | 完整陈述/祈使 | high |
| ack-seeking | 你明白吗/对吧 —— 要短促反馈 | high |
| answer-seeking | 你觉得呢/怎么改 —— 要立即回答 | high |
| ambiguous | 也许吧/可能可以 —— 中带 | mid |
| short-answer | 好的/可以 | high |
| explicit-signal | context 声明的判停礼仪命中 | high |
| complete-question | 英文可答问句 | high |

## 判别策略（instruction.txt 承载）

1. 明确多分类任务 + 目标 + 机制（输出不严格即出错）。
2. 分句判别基本策略（尾连接词/悬空条件 = 未完）。
3. **听觉礼仪**：ASR 谐音按义不按字；context 显式判停信号 = 正向；三类口语讯号
   （嗯/啊=思考中 low，你明白吗=要反馈 high，你觉得呢=要回答 high）。
4. 行为约束：直觉 > 思考，只输出一个整数。
5. prompt 结构：`<context>…</context><input>…</input>`，`</input>` 后只出分数。

## Run

```bash
.venv/bin/python run.py --n 1                # 默认 small_fast_model
.venv/bin/python run.py --n 3 --tag flash     # 换模型重跑对比
```

产物信号：constraint（clean 单数字占比）+ band accuracy（命中占比）+ avg latency
+ token 开销（input / cache_read / cache_write / output）。

## Findings (2026-09-15, deepseek-flash, n=1)

- constraint 30/30 clean —— plain-text 单 token 可靠。
- band accuracy 26/30（87%）；谐音 5/5 全过；歧义中带（也许吧/可能可以）过判为 high。
- **cache: cache_read 11648 / input 18691 = 62%** —— 固定 instruction（384 token）命中缓存，变量 prompt（~240）未缓存。
- **latency 根因 = per-call agent rebuild**：`funcs.call` 每次重建 Agent（~3s），`funcs.caller` 复用（中位 0.84s）。服务端抖动尖峰仍在（最大 12s+）。
- 下一步：去掉 `<input>`、保留 `<context>`、一个 clause 一个 content block（缓存随累积稳定上升）；输出约束收紧到 `max_output_tokens=1`。
