"""DoloresRun — Dolores articulate 的单次 run 事务 (操作面骨架, 待装线).

本文件只画操作面 + 记录方案, 实现留待下一步. 与 ``_ego.py`` 的草稿阶段同源 —
所有函数体均为占位 (``...``). 核心未决问题见「核心问题」节.

── 定位 ──────────────────────────────────────────────────────────────
DoloresRun 是 pydantic-ai ``run()`` 风格的 run 对象: ``ego.run()`` 返回它,
articulate 用 ``async with`` + ``async for`` 消费:

    async with ego.run() as run:
        async for event in run.events():
            ...

``async with`` 是 transaction 边界 (aenter 开 / aexit 关), ``async for`` 消费
事件流. 迭代结束后 run 对象仍在, 数据面事后可读 (usage / messages / head /
seq 范围), 这是相对纯 async generator 的本质优势 — generator 退出即烧状态.

── 拓扑: 两路生产, 一路消费, 完全平行 ──────────────────────────────

    [MOSS] run
      生产① events:  session 消费 loop → on_session_event("*") → put event ──┐
      生产② done:    run_task → await plugin.run RPC → finally put 毒丸 ─────┼→ [janus queue] → events()
      强关闭:        close() → cancel run_task + shutdown queue              ┘

    [plugin] POST /moss-api/ghost/dolores/run
      open preStep lock → steer 请求 → await turn done → close → return

── 生命周期迁移到 plugin: done 的权威是 plugin, 不是 MOSS ───────────
「这一轮 run 何时结束」不该由 MOSS 猜 (idle / turn-end / host 帧都不可靠).
把 run 的整个生命周期迁到 plugin: 发起 run 请求和 done 信号合成一个 RPC.
run_task 的生命周期 = run 的生命周期, **task 退出必塞毒丸** (finally).

    _drive():  try: resp = await call("/run", payload)
               finally: put 毒丸           # 成功 / 异常 / 取消 都塞

── seq watermark (解决跨进程竞态) ──────────────────────────────────
done 和 events 走两条 TCP (HTTP response vs WS mux), 毒丸可能**追过**最后
几个 event 先入队, events() 提前停丢 logos. 解法: plugin 的 run 响应返回
**权威 seq 范围** (start_seq + end_seq), MOSS 侧不自行推算.

    _drive():  resp = await call("/run", ...)      # {start_seq, end_seq, ...}
               自记录 start_seq / end_seq
               等 self._last_seq() >= end_seq       # 追平最后事件, 带超时
               才 put 毒丸

seq 范围同时是 **Memento moment record 的边界**: ``(session_id, start_seq,
end_seq)``. 这正是 FEATURE 里「Memento 只存 (sessionId, seq) 指针, 物理存储
委托 dsh」的落地 — run 的边界 = moment 的边界.

── 强关闭 ───────────────────────────────────────────────────────────
``close()`` 一次性终止两路: cancel run_task (→ HTTP 断开 → plugin 侦测
disconnect → abort turn) + shutdown queue (→ events() 抛 AsyncQueueShutDown).
幂等. 毒丸 = 正常 done, shutdown = 强关闭兜底, 分工不同.

── 核心问题 (下一步展开): turn/start 监听 + perStep 阻塞 ────────────
本设计把「done 判定」压到 plugin, 但 plugin 侧 run RPC 的接口面还没定 —
核心就在这两个:

1. **turn/start 监听** — plugin 怎么知道本轮 turn 开始 (start_seq 的锚), 以及
   「turn 结束」在 dsh 原生面用哪个 API 判 (agent idle? final result? turn/end
   配合 reason?). 这是「拆回 mindflow」的落点: perStep 在 articulate 相位内,
   articulate flag 不成立时锁 perStep; when idle 给 articulator 开锁.
2. **perStep 阻塞** — run 进行中, 除开锁的 turn 外, 其它 perStep 是否一律阻塞?
   阻塞面与 ego 的 preStep lock 如何咬合?

这两个决定 run RPC 的 payload / 返回形状, 是装线前必须先定的.

── 数据面 (迭代结束后可读) ─────────────────────────────────────────
  usage        累计 TokenUsage (assistant/message 累加)
  events       全量持有 (本 run 窗口内所有 session event)
  messages()   截取返回结果的 message 结合对象 (按 head 截窗)
  head         surface 末端游标 (messages() 的截取边界)
  start_seq / end_seq  本 run 的权威 seq 范围 (来自 plugin 响应)
  moment_record()       Memento moment record = (session_id, start_seq, end_seq)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.session import DshSession
    from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent, TokenUsage

__all__ = ["DoloresRun"]

# 毒丸 sentinel: run_task 退出时入队, events() 读到即正常终止.
# 与 shutdown (AsyncQueueShutDown) 分工: 毒丸 = 正常 done, shutdown = 强关闭兜底.
_POISON = object()


class DoloresRun:
    """articulate 单次 run 事务. 详见模块 docstring.

    操作面:
      async with / close      transaction 边界 + 强关闭
      events()                事件流 (毒丸 / shutdown 终止)
      usage / messages / head / seq / moment_record  迭代后数据面
    """

    def __init__(
        self,
        session: DshSession,
        *,
        call: Callable[[str, dict], Awaitable[dict]],
    ) -> None:
        """构造无副作用 (不碰 httpx / session / queue).

        :param session: ego 持有的长命 session — 事件源 (catch-all 订阅).
        :param call: plugin RPC 通道 (launcher.call 绑定), 发 run 请求用.
            窄依赖: run 只认「发 plugin 请求」这个面, 不认整个 launcher.

        todo: 内部态 — janus queue / _events 列表 / _last_seq / run_task 句柄 /
        start_seq / end_seq / usage 累计. 均 aenter 才物化.
        """
        ...

    # ── transaction 边界 ──────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """开 transaction. 注册 catch-all 监听 + 启动 run_task, 返回 self.

        顺序 (先后有依赖, 防竞态):
          1. 物化 janus queue + 累积态 (_events / _last_seq / usage).
          2. session.on_session_event("*", self._on_event) 注册全量持有.
          3. run_task = create_task(self._drive()) — 发 run 请求 + 塞毒丸.
        测试钩子: 早期可用 session.prompt("hello") 替代 plugin run RPC 验证
        「事件全量入队 + 毒丸终止」, 之后再换成真 run 请求.
        """
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """关 transaction, 委托 close(). 无论正常 / 异常都走到强关闭."""
        ...

    async def close(self) -> None:
        """强关闭, 幂等 — 一次性终止两路.

          cancel run_task      → HTTP 断开 → plugin 侦测 disconnect → abort turn
          dispose 监听          → 停止事件入队
          shutdown queue        → events() 抛 AsyncQueueShutDown, get 不阻塞

        顺序: 先停生产 (dispose), 再停 done (cancel run_task), 最后 shutdown
        queue. 反向会让 run_task 的 finally 往已 shutdown 的队列放毒丸撞
        QueueShutDown.
        """
        ...

    # ── 事件流 ─────────────────────────────────────────────────────

    async def events(self) -> AsyncIterator[SessionEvent]:
        """从队列读原始 session event, 毒丸或 shutdown 终止.

        终止条件二选一:
          - 读到 _POISON       → 正常 done (run_task 已追平 end_seq).
          - AsyncQueueShutDown → 强关闭兜底.

        todo: 待定 — run 对象自身是否也做 __aiter__ (async for evt in run),
        还是只暴露 events() 让 articulate 外层包 loop.
        """
        ...

    # ── 数据面 (迭代结束后可读) ────────────────────────────────────

    @property
    def usage(self) -> TokenUsage:
        """本 run 窗口的累计 token 用量 (assistant/message usage 累加)."""
        ...

    def messages(self) -> Any:
        """截取返回结果的 message 结合对象 — 从 surface 事件重建, 按 head 截窗.

        todo: 返回类型待定 (重建的 Message 列表? 还是 MOSS message 结合对象).
        """
        ...

    @property
    def head(self) -> Any:
        """surface 末端游标 — messages() 的截取边界. 语义待定 (seq? 消息索引?)."""
        ...

    @property
    def start_seq(self) -> int:
        """本 run 的权威 start seq — 来自 plugin run 响应, 非 MOSS 自行推算."""
        ...

    @property
    def end_seq(self) -> int:
        """本 run 的权威 end seq — plugin 响应, 毒丸追平的锚点."""
        ...

    def moment_record(self) -> Any:
        """Memento moment record = (session_id, start_seq, end_seq).

        todo: 返回类型待定 — 直接对接 momento-mori 契约的 moment 标识面.
        """
        ...

    # ── 内部 (设计面) ──────────────────────────────────────────────

    async def _drive(self) -> None:
        """run_task 本体: 发 run 请求 → 追平 end_seq → 塞毒丸 (finally 保证).

        形状 (todo):
            try:
                resp = await self._call("/run", payload)   # {start_seq, end_seq}
                self._start_seq, self._end_seq = ...
                await self._wait_catchup(self._end_seq)    # 追平, 带超时
            finally:
                put _POISON                                # 成功/异常/取消都塞

        核心问题 (下一步): payload 形状 + plugin 侧 turn/start 监听 + perStep
        阻塞. 见模块 docstring「核心问题」节.
        """
        ...

    def _on_event(self, event: SessionEvent) -> None:
        """catch-all 回调: 全量持有 — 累积 _events + 推进 _last_seq + 入队.

        生产① (events 路). 与 run_task (生产② 毒丸路) 完全平行, 这就是跨进程
        竞态的根源 — 毒丸可能先到, 故需 _drive 追平 end_seq 再塞.
        """
        ...

    async def _wait_catchup(self, end_seq: int) -> None:
        """等累积事件追平 end_seq (带超时). 追平后才允许放毒丸.

        这是把「完成」重新锚回事件流的 seq watermark 机制.
        """
        ...
