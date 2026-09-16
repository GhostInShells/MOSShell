# Listener Controller 骨架 — 判停逻辑装线层

> 2026-09-13 会话收敛。承接 FEATURE.md 的 listener 分层设计，把"判停逻辑"从
> recognition 层彻底上移，落成一个可被 ghost 通过 command 治理的聆听单元。
> 本文是骨架理解，不是实现规范。

## 定位

listener controller 是判停逻辑的装线层，四点：

1. 所有判停逻辑由它装线 listener 完成（recognition 层不再判停）。
2. 提供若干 `async def method`，每一个是一种**聆听礼仪**。
3. 有 `as_channel` 机制，把聆听礼仪暴露成 ghost 可控制的机制，可图形界面化。
4. 它控制发送 signal 的机制（上行 signal 的产生方式）。

## 两条信号通道（核心）

判停逻辑需要两类信号，性质不同，不能混为一条：

| 通道 | 事件 | 性质 | 用途 |
|------|------|------|------|
| 决策点 | clause | 低频，inline await（可阻塞） | once 立刻 commit / always 启动等待 / llm 校验启动打分 / 快捷响应启动 |
| 活动信号 | first/partial | 高频，轻量非阻塞（只刷时间戳/置标志） | always 的等待 reset / llm 打分的 cancel |

- **on_clause 是 inline await**（做法 1）：回调可阻塞，外部自行裁决要不要 create task。
  这保留了未来插 LLM 级 ASR 重写（clause 回调阻塞入历史再发送）的能力——现在不立起
  这个机制，以后也不会再有。约束：commit 要能打断 inline await 中的回调（回调包成
  task，commit cancel 它，receive loop 兜 CancelledError 继续收 tail）；异常要兜住。
- **活动信号绝不能 inline await**——高频事件阻塞一次就堵死收包。

## 三个定死的概念

- **FIRST = 每 segment 一次**。`contracts/asr.py` 写的 "Emitted at most once per
  stream" 是错的，要改成 per segment。sauc recognizer 的 `_advance_segment` 重置
  `_first_emitted` 反而是对的。
- **首包 = clause 之后第一个有语义的包**（text 从空重新变非空），不是 phase。前提是
  seedasr 1:1 发包、静音也发包（text 空）——识别层是唯一知道"静音 text 空"的地方，
  判停逻辑拿不到静音包细节。首包可能是 FIRST（clause 后切段了）或 PARTIAL（同一段内），
  是可推导的活动信号。
- **裁决载体 = command 下行**，不是 signal 上行。signal 是感知上行（Shell→Ghost），
  command 是控制下行（Ghost→Shell）。ghost 通过 command 控制 listener controller 的
  channel 换状态机，从而改变它上行 signal 的产生方式。

## 七种聆听礼仪

> 2026-09-17 订正：此前此处记「五种」是模型压缩错误，真实讨论是七种。

| # | 礼仪 | 行为 | 判停时机 | 备注 |
|---|------|------|----------|------|
| 1 | 主动听一次 | 模型发起，听一次 | 模型触发 | 半双工 |
| 2 | 打断式 turn-taking | 人说话就打断，说完才响应 | 人说完 | 基础模式 |
| 3 | 智能判停 (llm_judge) | 在 2 上延长 vad + llm func 判停 | 打分 > threshold commit | 长论述判停，已落地 |
| 4 | buffer + 人工点击 | 默认不发送，本地 buffer，界面呈现 | 人点击发送 | 缓冲系 |
| 5 | 模型改写 | 在 4 上加 ASR 重写 + corpus 配置 | 改写后发送 | 缓冲系 |
| 6 | 快捷响应 | 端侧多分类 → command 到 ghost | clause 判定多分类 | 只执行不思考 |
| 7 | 旁听模式 | 每分句都发送 | 每分句 | 模型只思考不说话 |

「关键字」不是独立礼仪，是附加参数（叠加到任意礼仪），命中判停关键字即显式终点。

分工：**#1 短句快，#2 基础打断式，#3 长论述准，#4/#5 buffer 系，#6 反射，#7 旁听**。
llm 校验的意义反转——不是短句判停（主动听一次已零延迟做到），而是超长论述判停（vad
在长论述里到处分句、无法判终点；打分判断"论述讲完了吗"）。

## commit / drop 语义

- `commit()` = 结束当前 segment，发尾包（负序号），结果进 signal。有效提交。
- `drop` = 丢弃当前 segment 后续包。区分 commit 的"有效"——误触/废话不提交，切段但不
  送 signal。若 `close()` 已覆盖"丢弃后续包"语义则不必新增。

## 裁决机制

- 默认 always。宏观切换（对话状态级，不是语句级）。
- 切换 = command 下达到 channel → 换状态机（spawn 新 state / cancel 旧 / 重新装线）。
- 切换 = cancel 在飞状态（等待计时器 / 打分 task / 快捷响应 task），复用 commit 打断机制。
- 兜底 = default + 图形界面人类手动 override。
