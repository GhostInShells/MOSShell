# 2026-10-13 — 听觉链路的静默失败与无感知/无自愈故障

> 触发: 人类架构师要求「从 interleaved voice 开始, 看音频模块有没有静默失败(不记录),
> 以及听觉循环异常导致崩溃但没有任何感知渠道和自愈渠道的问题」。
> 结论: **两类都有, 且互锁**。本文记录发现 + 修复方案 + 决策。索引见 [FEATURE.md](FEATURE.md)。

## 阅读范围 (全文读毕)

`host/voice/interleaved.py`、`host/listener/listener.py`、`host/listener/controller.py`、
`host/listener/volcengine_sauc/recognizer.py`、`host/audios/miniaudio_impl/miniaudio_capture.py`、
`host/audios/aec/webrtc_aec.py`、`contracts/asr.py`、`contracts/listener.py`、
`host/listener/etiquette.py`、`core/asr/null.py`。

---

## 一、静默失败 (无记录 / 无观测面)

### S1 — 帧队列满 → 无声丢帧 (纯静默, 连日志都没有)

`host/audios/miniaudio_impl/miniaudio_capture.py:337-340`

```python
def _on_chunk(self, chunk: AudioChunk) -> None:
    if self._queue is not None and not self._shutdown:
        with contextlib.suppress(Exception):
            self._queue.sync_q.put_nowait(chunk)
```

ASR 消费慢时 janus 有界队列(默认 `max_queue_frames=128`)见底, `put_nowait` 抛
`janus.SyncQueueFull` 被 `suppress(Exception)` 整个吞掉。实时流丢帧是**设计允许**的
(采集线程不能阻塞), 但**丢了没有任何记录**: 无日志、无计数、无 signal。没有任何面
能观测"输入在丢"。

### S2 — 设备锁争用 → 假装起动

`miniaudio_capture.py:144-149`

```python
if not self._locker.acquire(timeout=0):
    self._logger.warning("Audio capture device locked by another process (%s), skipping start", lock_key)
    self._locker = None
    self._started = True          # ← 没开设备, 却宣布已启动
    return
```

锁被另一 MOSS 进程持有时, 记一条 `warning` 就把 `_started` 置 True 返回, 从不打开设备。
下游 `HostListener.__aenter__`(`listener.py:192-197`)见 `is_running()==True` 就不再启动它,
`ListenerController.__aenter__`(`controller.py:207`)同理 —— 耳朵"活着", 但无设备、无音频。
`is_running()` 无法区分"在采集"与"无设备空转", 也**无信号面**让模型知道耳朵是死的。
(knock 能量检测读 `pull_latest()` 恒为 None → 门铃永不响, 连降级也失效。)

### S3 — AEC 每帧失败 → 全丢帧, 不自愈

`miniaudio_capture.py:130-135` 经 `263-264`

```python
out = aec.process(mono)          # 抛异常 → 逐帧落到 _capture_generator 的 except
...
except Exception:
    logger.exception("Error in capture callback")   # 记异常, 帧被丢, 循环继续
```

`aec.process()`(pywebrtc 原生态)抛异常时, 帧被丢、异常每帧刷一条。**不卸载 AEC(不自愈)**,
对模型是"设备在跑却永远没识别结果"。

---

## 二、听觉循环异常 → 崩溃, 无感知、无自愈 (核心)

### H1 — pump 任务死掉, 全链路无人监督

**链路** (`host/listener/listener.py`):

- `_pump`(`309-311`)唯一职责是 `async for result in self._recognition: self._dispatch_result(result)`。
- ASR 流致命错误 → `recognizer.py` `_run_session` 的 finally 往队列放 `None`:
  - 连续 connect 失败达 `max_connect_retries`(默认 **10**, 退避 0.5→8s, 约 47s) →
    `_fatal_error = RuntimeError(...)`(`recognizer.py:264-268`);
  - 或 session 任意异常(`274-276`)。
- `__anext__` 取出 `None` 后 `raise self._fatal_error`(`recognizer.py:240-245`)。
- `async for` 接住该 raise → **`_pump_task` 带异常结束**。
- 此后: `_running` 仍 True、`is_running()` 仍 True、`state` 仍在。controller 的
  `_run_etiquette` 仍停在 `await self._wait_for_idle(...)`(`controller.py:384`)。

**无感知渠道**:

- 设计渠道 `ASR.on_error(callback)`(`contracts/asr.py:366`, `recognizer.py:92-93`)**全仓零注册**
  (grep 确认: 只有 `NullASR.on_error` no-op 与定义本身)。`_error_callback` 永远 `None`,
  `_report_error`(`recognizer.py:549-555`)只 `logger.error`。ASR 自己侦测到的错误(server
  `code!=0`、connect 失败)**到不了任何消费方** —— 一条存在但空接的死契约。
- pump 异常的唯一天然出口是 `__aexit__` 的 `await self._pump_task`(`listener.py:261-262`),
  那里只 `suppress(asyncio.CancelledError)` → 真异常在此**重抛**(且发生在会话拆除时, 太晚,
  还会污染 teardown: 被 cancel 的旧会话退出时把 fatal error 重抛成"never retrieved")。
  运行中**没有任何东西 poll / 监督** `_pump_task`(无 done_callback, 无看门狗)。
- **信号面会撒谎**: 默认礼仪 `always` 带 `idle_timeout=300`(`etiquette.py:221-226`), pump
  死后无识别事件 → 300s 后 `_wait_for_idle` 触发, 发一条
  `"no speech for 300s — degrading to knock"`(`controller.py:892-908`)。真实原因是 **ASR 挂了**,
  信号却归因于"没人说话", 然后降级到 knock(不做 ASR)。这是**误导性感知**, 不是感知。

**无自愈渠道**:

- 无重启、无看门狗。会话只能靠外部再次 `run_etiquette` / `stop`(`controller.py:578-582`) 重建,
  而崩溃本身不触发任何重建。
- 连带: `_active_task`(`controller.py:329-332`)亦无 done_callback, `start_default_etiquette`
  的返回值在 `interleaved.py:141` 被直接丢弃 → 礼仪任务若异常结束也静默。

### 次要同病

- `recognizer.py:376-377` `_send_loop` 吞异常只 `_report_error`(无人听)后静默退出, 靠 receive
  侧 tail 超时兜底。
- `listener.py:162-187` `_sound_detection_loop` 只捕 `CancelledError`, fire-and-forget 无 await。
- S1 与 H1 互锁: ASR 死后 `_run_session` 退出, 无人再迭代 `_audio_chunks`, 采集端持续入队 →
  队列满 → **静默丢帧(S1)**。

---

## 三、修复方案

### 决策 1 — 把"耳朵故障"做成一等感知面 (对称 ASR.on_error)

`ASR.on_error` 已存在但零消费。修法:

- `contracts/listener.py` 的 `Listener` 加**非抽象默认 no-op** 的
  `on_error(cb) -> Discard`, 与既有 `on_sound_detected` 同形(不破坏 mock/Null 子类)。
- `HostListener` 实现它, 扇出两个来源: ①`asr.on_error`(构造/enter 时注册);
  ②session pump 崩溃。
- `HostListenerState` 构造接 `on_fault` 回调; `_pump` 捕异常后**不再让异常逃逸**
  (catch → 上报 → 正常 return), 从根上消除 teardown 重抛。
- `is_running()` 说真话: pump 崩溃后置 `_running = False`。

### 决策 2 — controller 消费故障: 发 signal + 有界自愈

- `ListenerController` 订阅 `listener.on_error` → 发一条 `listener:error` signal
  (source = `cell_name`, 与 `listener:expect-timeout` / `idle-timeout` 同形), 让模型**知道**
  耳朵挂了及原因。
- 自愈: 若当前有激活礼仪, 排期**重启同一礼仪**(延迟一个 tick, 避免在 pump 协程内自取消)。
  **有界**: 连续故障计数, 封顶后停手并改发 `listener:degraded`(需模型手动 re-activate);
  任一识别事件到达即复位计数。封顶值 = 常量 `_MAX_FAULT_RESTARTS`。

  **Why 有界**: ASR 自身 `max_connect_retries=10` 约 50s 才 fatal; 无界重启会变成每 ~50s
  一次的 signal 风暴。有界=给真正的瞬断自愈窗口, 恒久故障交回模型。

### 决策 3 — 静默失败逐个补观测/自愈

- **S1**: `_on_chunk` 显式捕 `janus.SyncQueueFull` → 计数 + 节流 warning(不刷屏), 暴露
  `dropped_frames()`(public-internal)。
- **S3**: `_apply_aec` 捕异常 → **原样放行 raw 帧(不丢)** + 连续失败计数, 达阈值
  `_AEC_MAX_FAILURES` 后卸载 AEC 并记 error → 降级但不再聋(自愈)。
- **S2**: 锁未获取时改记 `error` + 新增 public `is_degraded()`(无设备空转), `device_explain`
  反映实况。`is_running()` 的"生命周期已 enter"语义**不变**(避免破坏 HostListener 托管判定)。

---

## 四、验证

契约行为测试(复用 `tests/ghoshell_moss/host/listener/test_controller.py` 的 mock 形态):

- pump 崩溃 → `listener.on_error` 观察到 → controller 发 `listener:error` signal + 重启;
  连续故障达封顶 → `listener:degraded` 且不再重启。
- `HostListenerState` pump 崩溃后 `is_running()` 转 False; `__aexit__` 不重抛(teardown 干净)。
- capture 队列满 → 记 warning 且 `dropped_frames()` 增长。
- AEC 连续失败达阈值 → 卸载 + 帧不丢(raw 放行)。

> 状态: 本文记录 + 修复同 commit。修复完成待人类架构师确认后提交。
