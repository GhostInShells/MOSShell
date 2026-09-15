"""Avatar — 一个正在渲染的 Live2D 形象, 驱动框架给形象作者的事件面.

形象作者 (写 `avatars/<name>/channel.py` 的模型) 只见这一个对象: 它把"我要它做什么"
发成事件, 驱动负责送到页面.

设计约束 (见 NODE.md):
  - **command 即真相**: 驱动不向页面回读参数. `_state` 记录"被命令过"的值, 就是真相.
  - **事件化**: 出站是事件流 (参数 / 动作 / 表情 / 背板 / 复位), 不是状态同步.
    参数事件在 flush 时按 id 合并 —— 同一帧里后写的赢.
  - **无页面也能用**: 没有 WS 客户端时事件照常入队并丢弃, `_state` 仍在累积.
    页面一旦连上, `hello` 会带一份全量快照让它追上.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

from .cubism import ModelSpec
from .persona import IdleConfig, Persona

Frame = dict[str, Any]


class Avatar:
    """事件的产生与分发端点. 页面 (WS 客户端) 是它的消费者."""

    FLUSH_INTERVAL = 1 / 30  # 出站帧率: 参数合并到 30fps 一帧
    PARAM_DURATION = 0.3  # 参数命令的默认缓动时长 (秒) —— 同轨命令的时间感

    def __init__(
        self,
        name: str,
        spec: ModelSpec,
        *,
        model_url: str,
        canvas: tuple[int, int] = (600, 1000),
        backdrop: str | None = None,
        backdrop_names: tuple[str, ...] = (),
        persona: Persona | None = None,
        logger: logging.Logger,
    ) -> None:
        self.name = name
        self.spec = spec
        self.persona = persona
        self.model_url = model_url
        self.canvas = canvas
        self.backdrop = backdrop
        self.backdrop_names = backdrop_names
        self.logger = logger

        self._state: dict[str, float] = {}
        self._clients: set[Any] = set()
        self._pending_params: dict[str, float] = {}
        self._pending_events: list[Frame] = []
        self._flush_task: asyncio.Task | None = None
        self._view_scale: float = 1.0
        self._view_x: float = 0.0
        self._view_y: float = 0.0
        self._lip_sync_enabled: bool = True
        self._speaking: bool = False
        self._interactions: deque[str] = deque(maxlen=5)

        idle_cfg = persona.idle if persona else IdleConfig()
        self.idle_delay: float = idle_cfg.delay
        self._idle: tuple[str, int] | None = (
            (idle_cfg.loop_group, idle_cfg.loop_index) if idle_cfg.loop_group else None
        )
        self.blink: bool = idle_cfg.blink
        self.breath: bool = idle_cfg.breath

        # 时间轨迹: _foreground 是"有命令正在占时"的计数, _idle_active 是待机循环是否在页面跑.
        self._foreground: int = 0
        self._idle_active: bool = False

    # ------------------------------------------------------------------ 能力

    def param_ids(self) -> list[str]:
        return [p.id for p in self.spec.params]

    def has_param(self, param_id: str) -> bool:
        return any(p.id == param_id for p in self.spec.params)

    def lip_param(self) -> str | None:
        """唇形参数: 优先 model3 的 Groups.LipSync 声明, 否则回退到 mouth-open 参数."""
        if self.spec.lip_sync:
            return self.spec.lip_sync[0]
        for p in self.spec.params:
            ident = p.ident.lower()
            if "mouth" in ident and "open" in ident:
                return p.id
        return None

    # ------------------------------------------------------------ 事件面 (出站)

    def param(self, param_id: str, value: float, *, manual: bool = False) -> None:
        """把一个参数推到某个值. 未知参数只记日志, 不抛 —— 形象代码不该被拼写错误打断.

        ``manual=True`` 表示这是模型显式命令 (而非唇动等连续驱动): 若目标是唇形参数,
        则关掉自动唇动, 让模型输出优先, 消除双驱动副作用.
        """
        if not self.has_param(param_id):
            self.logger.warning("avatar %s: unknown param %r ignored", self.name, param_id)
            return
        if manual and param_id == self.lip_param():
            self._lip_sync_enabled = False
        value = float(value)
        self._state[param_id] = value
        self._pending_params[param_id] = value

    def params(self, values: Mapping[str, float]) -> None:
        for k, v in values.items():
            self.param(k, v)

    def motion(self, group: str, index: int = 0) -> None:
        """fire-and-forget 播一个动作 (不占时、不自动复原). 供 channel.py 作者做非阻塞动作.

        占时的前景动作请用 ``await avatar.play(...)`` —— 它播完会清动作。
        """
        if group not in self.spec.motions:
            self.logger.warning("avatar %s: unknown motion group %r ignored", self.name, group)
            return
        self._pending_events.append({"t": "motion", "g": group, "i": index})

    async def play(self, group: str, index: int = 0, *, hold: float = 0.0) -> None:
        """占时的前景动作: 播一个动作, 占满它的时长, 结束即复原 (finally 级).

        时长默认取 motion3.json 的 Meta.Duration (单圈); ``hold>0`` 主动覆盖延长。
        动作全部 Loop:True (实测 hiyori), 所以结束不是靠页面回报, 而是 driver 自己计时
        后发 ``clear_motion`` 停掉。
        """
        if group not in self.spec.motions:
            self.logger.warning("avatar %s: unknown motion group %r ignored", self.name, group)
            return
        self._foreground += 1
        self._idle_active = False
        self.stop_motion()
        self._pending_events.append({"t": "motion", "g": group, "i": index})
        duration = hold if hold > 0 else self.spec.motion_duration(group, index)
        try:
            await asyncio.sleep(duration)
        finally:
            self._foreground -= 1
            self.clear_motion()

    def stop_motion(self) -> None:
        """停掉所有动作, 参数不动."""
        self._pending_events.append({"t": "stop_motion"})

    def clear_motion(self) -> None:
        """停掉动作并把参数复位到模型默认, 再重下发被命令过的状态 (command 即真相)."""
        self._pending_events.append({"t": "clear_motion"})
        if self._state:
            self._pending_params.update(self._state)

    def expression(self, name: str) -> None:
        if name not in self.spec.expressions:
            self.logger.warning("avatar %s: unknown expression %r ignored", self.name, name)
            return
        self._pending_events.append({"t": "expression", "n": name})

    def reset(self) -> None:
        """表情与参数回到模型默认, 并恢复自动唇动."""
        self._state.clear()
        self._pending_params.clear()
        self._lip_sync_enabled = True
        self._pending_events.append({"t": "reset"})

    def set_backdrop(self, url: str | None) -> None:
        self.backdrop = url
        self._pending_events.append({"t": "backdrop", "url": url})

    # ------------------------------------------------------------ 视图 (缩放/平移)

    def view_state(self) -> Frame:
        """当前视图: scale 是缩放倍率, x/y 是相对视口中心的偏移 (-1..1)."""
        return {"scale": self._view_scale, "x": self._view_x, "y": self._view_y}

    def zoom(self, factor: float) -> None:
        """缩放形象. factor=1.0 为默认大小."""
        self._view_scale = float(factor)
        self._emit_view()

    def move(self, x: float, y: float) -> None:
        """平移形象. x/y 取值 -1..1, (0,0)=视口中心."""
        self._view_x = float(x)
        self._view_y = float(y)
        self._emit_view()

    def _emit_view(self) -> None:
        self._pending_events.append({"t": "view", **self.view_state()})

    # ------------------------------------------------------------------ 状态面

    def state(self) -> dict[str, float]:
        """被命令过的参数快照. 这是驱动的"真相", 不是页面回读."""
        return dict(self._state)

    def set_idle_loop(self, group: str | None, index: int = 0) -> None:
        """设定全身待机循环动作; 传 None 回到自动 (优先名字像 Idle 的组, 否则第一个)."""
        if group is None:
            self._idle = None
        else:
            if group not in self.spec.motions:
                self.logger.warning("avatar %s: unknown idle group %r ignored", self.name, group)
                return
            self._idle = (group, index)
        if self._idle_active:
            # 待机正在跑 → 立刻换到新 loop
            self.idle_loop()
        else:
            self._idle_active = False  # 让 idle manager 下一拍按新 spec 重启

    def idle_group(self) -> str | None:
        """挑一个动作组当 idle: 优先名字像 Idle 的, 否则第一个."""
        for name in self.spec.motions:
            if "idle" in name.lower():
                return name
        return next(iter(self.spec.motions), None)

    def idle_spec(self) -> tuple[str, int] | None:
        """当前待机动作 (含自动挑选的结果); 没有动作组时 None."""
        if self._idle is not None:
            return self._idle
        group = self.idle_group()
        return (group, 0) if group else None

    def idle_loop(self) -> None:
        """把待机循环交给页面跑 (动作 Loop:True, 页面侧无限循环)."""
        spec = self.idle_spec()
        if spec is None:
            return
        self._pending_events.append({"t": "idle", "g": spec[0], "i": spec[1]})

    async def run_idle_manager(self) -> None:
        """待机仲裁 (driver 持有时间): 空闲超过 idle_delay 才进待机, 前景活动/说话则让位.

        跑在 channel 的 ``build.running`` 生命周期里, 永远循环. 不依赖 ``build.idle`` 的
        取消语义 —— 那个只在本 channel 自身收到命令时才退出, 子 channel 命令不会 (已查证),
        用它做待机会有漏洞.
        """
        quiet_since = time.monotonic()
        while True:
            await asyncio.sleep(0.1)
            if self._foreground or self.speaking:
                quiet_since = time.monotonic()
                if self._idle_active:
                    self.stop_motion()
                    self._idle_active = False
                continue
            if self._idle_active or self.idle_spec() is None:
                continue
            if time.monotonic() - quiet_since >= self.idle_delay:
                self.idle_loop()
                self._idle_active = True

    # ---------------------------------------------------------------- 唇动

    @property
    def lip_sync_enabled(self) -> bool:
        return self._lip_sync_enabled

    def set_lip_sync(self, enabled: bool) -> None:
        self._lip_sync_enabled = bool(enabled)

    def set_speaking(self, on: bool) -> None:
        """说侧的连续状态 (仅内部, 不发帧): 说话期间待机动画让位.

        模型自带的动作曲线几乎都驱动嘴部参数 (实测 hiyori 的 10 个 motion 全部驱动
        ParamMouthOpenY/ParamMouthForm), 循环待机动作会和唇动抢同一个参数 —— 所以
        说话时必须让待机停下来. idle manager 读 ``speaking`` 决定让位.
        """
        self._speaking = bool(on)

    @property
    def speaking(self) -> bool:
        return self._speaking

    # ---------------------------------------------------------------- 人类交互

    def record_interaction(self, text: str) -> None:
        """记一条人类交互 (点击/拖拽), 供 notice tail 给模型感知."""
        self._interactions.append(text)

    def interactions(self) -> list[str]:
        return list(self._interactions)

    def tap_group(self) -> str | None:
        """找一个点击反馈动作组: 优先名字含 Tap 的, 否则 None."""
        for name in self.spec.motions:
            if "tap" in name.lower():
                return name
        return None

    def on_tap(self) -> str | None:
        """人类点击形象: 记录交互, 有 Tap 动作则播一个 (占时前景动作)."""
        self.record_interaction("点击")
        group = self.tap_group()
        if group:
            asyncio.create_task(self.play(group, 0))
        return group

    # ---------------------------------------------------------------- 客户端

    def attach(self, ws: Any) -> None:
        self._clients.add(ws)

    def detach(self, ws: Any) -> None:
        self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def hello_frame(self) -> Frame:
        """新页面连上时的全量快照 —— 让它追上已经发生的命令."""
        spec = self.idle_spec()
        return {
            "t": "hello",
            "name": self.name,
            "model": self.model_url,
            "canvas": list(self.canvas),
            "backdrop": self.backdrop,
            "params": self.state(),
            "view": self.view_state(),
            "idle": {"g": spec[0], "i": spec[1]} if spec else None,
            "idle_active": self._idle_active,
            "parts": {"blink": self.blink, "breath": self.breath},
        }

    # ---------------------------------------------------------------- 分发循环

    def start(self) -> None:
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self.FLUSH_INTERVAL)
            await self.flush()

    async def flush(self) -> None:
        if not self._pending_params and not self._pending_events:
            return
        frames: list[Frame] = []
        if self._pending_params:
            frames.append({"t": "params", "v": self._pending_params})
            self._pending_params = {}
        if self._pending_events:
            frames.extend(self._pending_events)
            self._pending_events = []
        await self._broadcast(frames)

    async def _broadcast(self, frames: list[Frame]) -> None:
        if not self._clients:
            return
        dead = []
        for ws in list(self._clients):
            try:
                for frame in frames:
                    await ws.send_json(frame)
            except Exception as e:  # 连接断开 / 写失败: 摘掉这个客户端, 不打断其它
                self.logger.debug("avatar %s: dropping ws client: %s", self.name, e)
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
