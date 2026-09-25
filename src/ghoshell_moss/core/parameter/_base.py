"""
AbsParameters — Parameters 的公共实现.

分工:
  - ``AbsParameters``        声明 / 订阅 / 收敛队列 / 生命周期; transport 交给子类.
  - ``ParametersBroadcaster`` 通讯协议封装 (队列 + zenoh), key 组 address-free.
  - ``TruthHostParameters``   真相节点: 启动即监听, 单点定序.
  - ``WorkerParameters``      非真相节点: 懒启动, 收 host 真值.

真相模型 — version 三值 (全系统的真值序):

    -1   仅要求广播一次 (require): host 不采纳, 回播当前真值一次
     0   首包: 仅在 host 尚无真值时被采纳 (铸为 >=1); 否则等同于 require
    >=1  真值声明: host 在版本不低时采纳, 否则回播当前真值一次

两条不变量:

1. **declaration 进 → truth 必出.** 每条进入 host 的声明, 采纳与否都引发一次真值
   广播; 采纳与否只决定广播的是谁的值. 因此 declare 本身就是 require, 没有独立的
   拉原语, 也没有"回谁"这回事 — 广播即回复.
2. **振荡闸口.** truth 到达只更新本地值, 绝不触发 declaration. declaration 只由两件
   事触发: 显式的 ``set()``, 与 host 化身变更 (host_alive).
"""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Callable, Type

from typing_extensions import Self

import janus

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.blueprint.parameter import (
    Parameters,
    ParameterModel,
    ParameterSchema,
    ParameterDeclaration,
    ParameterSubscriber,
    T_PARAM,
    ParameterMeta,
    ParameterData,
    REQUIRE_ONLY,
    FIRST_PACKET,
)

__all__ = [
    "BaseParameterSubscriber",
    "BaseParameterDeclaration",
    "AbsParameters",
    "ParametersBroadcaster",
    "TruthHostParameters",
    "WorkerParameters",
]

_Key = str
_Address = str
_Epoch = str
_Disposer = Callable[[], None]


# ======================================================================
# 读者 / 写者 handle
# ======================================================================


class BaseParameterSubscriber(ParameterSubscriber[T_PARAM]):
    """
    读者: 本地值与网络真值两分.

    ``value`` 是最新的本地值 (乐观, 可能未被 host 确认); ``get_truth`` 拿已确认的
    真值. 真值到达只更新本地并触发回调, 绝不触发 declaration (振荡闸口).
    """

    def __init__(
            self,
            meta: ParameterMeta,
            key: str,
            model: Type[T_PARAM] | T_PARAM,
            logger: LoggerItf | None = None,
    ):
        self._meta = meta
        self._key = key
        if isinstance(model, type):
            self._model: Type[T_PARAM] = model
            default: T_PARAM = model.default()
        else:
            self._model = type(model)
            default = model.model_copy()
        self._default_value: T_PARAM = default
        self._current_value: T_PARAM = default.model_copy()
        self._last_truth_data: ParameterData | None = None
        self._on_change_callbacks: set[Callable[[T_PARAM], None]] = set()
        self._on_truth_callbacks: set[Callable[[ParameterData], None]] = set()
        self._has_any_truth_event = ThreadSafeEvent()
        self._logger = logger or get_moss_logger()
        self._log_prefix = "[Parameter key=%s]" % key
        self._closed = False
        self._disposer: _Disposer | None = None

    # -- 读 --------------------------------------------------------------

    @property
    def key(self) -> str:
        return self._key

    @property
    def meta(self) -> ParameterMeta:
        return self._meta

    @property
    def value(self) -> T_PARAM:
        """最新本地值 — 乐观, 未必已被 host 确认."""
        return self._current_value

    def get_truth_data(self, use_default: bool = False) -> ParameterData | None:
        """已确认的真值原始数据. 无真值时: ``use_default`` 则返回首包形态."""
        if self._last_truth_data is not None:
            return self._last_truth_data
        if use_default:
            return self.make_data(FIRST_PACKET)
        return None

    def get_truth(self, use_default: bool = False) -> T_PARAM | None:
        """已确认的真值. 无真值且 ``use_default=False`` 时返回 None."""
        data = self.get_truth_data(use_default)
        if data is None:
            return None
        return self._model.model_validate(data.payload)

    def _data_from(self, value: T_PARAM) -> ParameterData:
        """用某个值构造一条 ParameterData, 盖上有效 key (可能被 declare/subscribe 覆盖)."""
        data = value.to_parameter_data(self._meta)
        data.key = self._key
        return data

    def make_data(self, version: int) -> ParameterData:
        """用当前值构造一条 ParameterData. version 由调用方按语义给 (见三值)."""
        data = self._data_from(self._current_value)
        data.version = version
        return data

    def parameter_schema(self) -> ParameterSchema:
        return self._default_value.to_parameter_schema()

    # -- 真值到达 --------------------------------------------------------

    def set_truth_data(self, data: ParameterData) -> None:
        """transport 回调 — 收到一条网络真值. 只更新本地, 绝不回发."""
        if self._closed:
            return
        if not self._accept_truth(data):
            return
        try:
            model = self._model.model_validate(data.payload)
        except Exception as e:
            self._logger.error("%s invalid truth payload %r: %s", self._log_prefix, data, e)
            return
        self._current_value = model
        self._last_truth_data = data
        self._has_any_truth_event.set()
        self._fire(self._on_change_callbacks, model)
        self._fire(self._on_truth_callbacks, data)

    def _accept_truth(self, data: ParameterData) -> bool:
        """
        接受闸口:
          - 无 last truth → 接受
          - epoch 变了 (host 换化身) → 接受 (版本跨 epoch 不可比)
          - epoch 相同 → 只在 version 更大时接受
        """
        if data.key != self._key:
            return False
        if data.version < 1:
            # 真值广播恒带 >=1; 其余值是声明, 不属于这条通道.
            return False
        last = self._last_truth_data
        if last is None:
            return True
        if data.epoch != last.epoch:
            return True
        return data.version > last.version

    async def wait_first_truth(self) -> T_PARAM:
        """
        阻塞等到第一条真值.

        host 不存在则遥遥无期 — 调用方以 ``asyncio.wait_for`` 负超时.
        若在拿到真值前被 ``close()`` (close 会 set 事件以防死锁), 醒来即抛 RuntimeError.
        """
        await self._has_any_truth_event.wait()
        if self._closed:
            raise RuntimeError("subscriber closed before any truth arrived")
        return self._current_value

    # -- 订阅回调 --------------------------------------------------------

    def on_change(self, callback: Callable[[T_PARAM], None]) -> _Disposer:
        """网络真值到达并通过闸口时触发 (本地乐观 set 不触发)."""
        if self._closed:
            raise RuntimeError("subscriber is closed")
        self._on_change_callbacks.add(callback)
        return lambda: self._on_change_callbacks.discard(callback)

    def on_truth(self, callback: Callable[[ParameterData], None]) -> _Disposer:
        if self._closed:
            raise RuntimeError("subscriber is closed")
        self._on_truth_callbacks.add(callback)
        return lambda: self._on_truth_callbacks.discard(callback)

    def _fire(self, callbacks, payload) -> None:
        for callback in list(callbacks):
            try:
                callback(payload)
            except Exception as e:
                self._logger.error("%s callback %r failed: %s", self._log_prefix, callback, e)

    # -- 生命周期 --------------------------------------------------------

    def set_disposer(self, disposer: _Disposer) -> None:
        self._disposer = disposer

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_change_callbacks.clear()
        self._on_truth_callbacks.clear()
        self._has_any_truth_event.set()
        if self._disposer is not None:
            disposer = self._disposer
            self._disposer = None
            disposer()


class BaseParameterDeclaration(BaseParameterSubscriber[T_PARAM], ParameterDeclaration[T_PARAM]):
    """
    写者 handle — declare 的产物.

    ``set`` 本地立即生效 (乐观) 并上报; **不触发 on_change** — on_change 只由真值
    到达触发, 否则同一次写会因为真值回声而通知两次.
    """

    def __init__(
            self,
            meta: ParameterMeta,
            key: str,
            model: T_PARAM,
            set_callback: Callable[[_Key, ParameterData], None],
            logger: LoggerItf | None = None,
    ):
        super().__init__(meta=meta, key=key, model=model, logger=logger)
        self._set_callback = set_callback
        self._last_declaration_data: ParameterData | None = None

    def set(self, value: T_PARAM) -> None:
        """本地立即生效并上报 (fire-and-forget). version = 已确认真值 +1; 无真值则首包."""
        self._current_value = value
        data = self._data_from(value)
        last = self._last_truth_data
        if last is not None:
            data.version = last.version + 1
            data.epoch = last.epoch
        else:
            data.version = FIRST_PACKET
        self._last_declaration_data = data
        self._set_callback(self._key, data)

    def reannounce_data(self) -> ParameterData:
        """
        host 化身变更时, 用当前值原样推回.

        继承已知最大版本, **不 +1** — 凭空 +1 会让本节点的值看起来比现存真值更新,
        把全序冲垮. 用 ``_current_value`` (总是最新) 而非旧的声明快照, 避免把
        已被真值覆盖的旧值再推出去.
        """
        data = self._data_from(self._current_value)
        version = FIRST_PACKET
        if self._last_truth_data is not None:
            version = self._last_truth_data.version
            data.epoch = self._last_truth_data.epoch
        if self._last_declaration_data is not None and self._last_declaration_data.version > version:
            version = self._last_declaration_data.version
        data.version = version
        return data


# ======================================================================
# parameters 服务
# ======================================================================


class ParametersBroadcaster(ABC):
    """
    parameters 的通讯协议封装 (队列 + zenoh), key 组 **address-free**.
    address 只活在 payload 的 ``ParameterMeta`` 里, 不参与寻址 (形如 ``TopicMeta.sender``).

        {param_ns}/host/truth/{key}          host 广播真值
        {param_ns}/worker/declaration/{key}  节点发布声明 (声明即 require)
        {param_ns}/host/liveness             host 上线 / 存活

    统一规则见模块 docstring: declaration 进 → truth 必出.
    """

    @abstractmethod
    async def __aenter__(self):
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    # -- 真值通道 --------------------------------------------------------

    @abstractmethod
    async def subscribe_host_truth(
            self, key: str, callback: Callable[[ParameterData], None],
    ) -> _Disposer:
        """监听 {param_ns}/host/truth/{key} — 网络中该 key 的唯一真相."""
        ...

    @abstractmethod
    async def publish_host_truth(self, parameter: ParameterData) -> None:
        """host 广播一条真值到 {param_ns}/host/truth/{key}."""
        ...

    # -- 声明通道 --------------------------------------------------------

    @abstractmethod
    async def publish_declaration(self, parameter: ParameterData) -> None:
        """节点发布声明到 {param_ns}/worker/declaration/{key} — 含 require (version=-1)."""
        ...

    @abstractmethod
    async def subscribe_declarations(
            self, callback: Callable[[ParameterData], None],
    ) -> _Disposer:
        """host 监听 {param_ns}/worker/declaration/** — 全网声明入口."""
        ...

    # -- host 化身 --------------------------------------------------------

    @abstractmethod
    def set_host(self, address: _Address, epoch: _Epoch) -> None:
        """host 声明自身化身 — 发布 {param_ns}/host/liveness."""
        ...

    @abstractmethod
    async def on_host_alive(self, callback: Callable[[_Address, _Epoch], None]) -> None:
        """监听 {param_ns}/host/liveness — host 上线 / 化身 (epoch) 变更通知."""
        ...


class AbsParameters(Parameters, ABC):
    """
    parameters 的公共实现: 声明 / 订阅 / 收敛队列 / 生命周期.

    队列做线性卸载 — ``_set_parameter`` 可能在 transport 线程被调, 它只入队, 不做 IO;
    真正的发送在 ``_publish_declaration_loop`` 里, 由 event loop 串行执行.
    """

    def __init__(self, meta: ParameterMeta, logger: LoggerItf | None = None):
        self._meta = meta
        self._logger = logger or get_moss_logger()
        self._declarations: dict[_Key, BaseParameterDeclaration] = {}
        self._subscribers: dict[_Key, BaseParameterSubscriber] = {}
        self._pub_queue: janus.Queue[tuple[_Key, ParameterData]] = janus.Queue()
        self._started = False

    # -- 声明 / 订阅 ------------------------------------------------------

    async def declare(
            self, model: T_PARAM, *, key: str | None = None,
    ) -> ParameterDeclaration[T_PARAM]:
        """
        声明本 parameter. ``model`` 携带默认值.

        单节点内同 key 幂等 — 重复 declare 返回同一 handle.
        """
        key = key or model.parameter_key()
        if key in self._declarations:
            return self._declarations[key]
        declaration = BaseParameterDeclaration(
            meta=self._meta,
            key=key,
            model=model,
            set_callback=self._set_parameter,
            logger=self._logger,
        )
        dispose_declaration = self._add_declaration(declaration)
        dispose_subscription = await self._add_subscriber(declaration)

        def _disposer() -> None:
            dispose_declaration()
            dispose_subscription()

        declaration.set_disposer(_disposer)
        return declaration

    async def subscribe(
            self, model: Type[T_PARAM], *, key: str | None = None,
    ) -> ParameterSubscriber[T_PARAM]:
        """订阅本 key 的网络真值. 单节点内同 key 幂等."""
        key = key or model.parameter_key()
        if key in self._subscribers:
            return self._subscribers[key]
        subscriber = BaseParameterSubscriber(
            meta=self._meta, key=key, model=model, logger=self._logger,
        )
        disposer = await self._add_subscriber(subscriber)
        subscriber.set_disposer(disposer)
        return subscriber

    def declared(self) -> list[ParameterSchema]:
        return [d.parameter_schema() for d in self._declarations.values()]

    # -- 注册表 ----------------------------------------------------------

    def _add_declaration(self, declaration: BaseParameterDeclaration) -> _Disposer:
        key = declaration.key
        exists = self._declarations.get(key)
        if exists is not None:
            exists.close()
        self._declarations[key] = declaration

        def _disposer() -> None:
            if self._declarations.get(key) is declaration:
                self._declarations.pop(key, None)

        return _disposer

    async def _add_subscriber(self, subscriber: BaseParameterSubscriber) -> _Disposer:
        key = subscriber.key
        exists = self._subscribers.get(key)
        if exists is not None:
            exists.close()
        self._subscribers[key] = subscriber

        def _remove() -> None:
            if self._subscribers.get(key) is subscriber:
                self._subscribers.pop(key, None)

        unsubscribe = await self._subscribe_parameter(key=key, subscriber=subscriber)

        def _disposer() -> None:
            unsubscribe()
            _remove()

        return _disposer

    # -- transport 钩子 (子类实现) ----------------------------------------

    def _set_parameter(self, key: _Key, data: ParameterData) -> None:
        """线性卸载: 只入队, 不做 IO. 调用方可能在 transport 线程."""
        try:
            self._pub_queue.sync_q.put_nowait((key, data))
        except janus.SyncQueueShutDown:
            pass

    @abstractmethod
    async def _publish_declaration(self, key: _Key, data: ParameterData) -> None:
        """把一条本地声明送上 transport (host: 自身写即真相; worker: 上报 host)."""
        ...

    @abstractmethod
    async def _subscribe_parameter(
            self, *, key: _Key, subscriber: BaseParameterSubscriber,
    ) -> _Disposer:
        """把 subscriber 接到 key 的真值通道 (host: 本地直连; worker: 订阅 host 广播 + 声明)."""
        ...

    # -- 生命周期 --------------------------------------------------------

    @abstractmethod
    def is_running(self) -> bool:
        ...

    async def _publish_declaration_loop(self) -> None:
        while self.is_running():
            try:
                key, data = await self._pub_queue.async_q.get()
                await self._publish_declaration(key, data)
            except asyncio.CancelledError:
                raise
            except janus.AsyncQueueShutDown:
                break
            except Exception:
                self._logger.exception("failed publishing declaration")


class TruthHostParameters(AbsParameters, ABC):
    """
    真相节点 (host) 的 parameters.

    - **启动即监听**: session enter 时发布 liveness 并订阅全网声明 (见 ``__aenter__``).
    - **单点定序**: version 由本节点铸. 采纳首包时铸新; 采纳继承版本时原样保留 —
      host 重启后不把全序清零.
    - **本地 set 即真相**: host 自身也是 declarer.
    """

    def __init__(
            self,
            address: _Address,
            broadcaster: ParametersBroadcaster,
            logger: LoggerItf | None = None,
    ):
        super().__init__(meta=ParameterMeta(address=address, host=True), logger=logger)
        self._address = address
        self._broadcaster = broadcaster
        # host 化身 = host 地址 (每实例唯一, uid 非持久化) — address 即 epoch,
        # 便于跨节点 debug 对齐.
        self._epoch: _Epoch = address
        self._truth: dict[_Key, ParameterData] = {}
        self._versions: dict[_Key, int] = {}
        self._declaration_queue: janus.Queue[ParameterData] = janus.Queue()
        self._publish_task: asyncio.Task | None = None
        self._absorb_task: asyncio.Task | None = None
        self._declarations_disposer: _Disposer | None = None
        self._closed = False

    def is_host(self) -> bool:
        return True

    def is_running(self) -> bool:
        return self._started and not self._closed

    async def declare(
            self, model: T_PARAM, *, key: str | None = None,
    ) -> ParameterDeclaration[T_PARAM]:
        """
        host 是正常的 Parameters, 且它的声明即真值.

        首次 declare 立即铸版广播 (只迭代版本号), 让声明值成为真值;
        重复 declare 幂等, 不再铸版.
        """
        key = key or model.parameter_key()
        is_new = key not in self._declarations
        declaration = await super().declare(model, key=key)
        if is_new:
            await self._publish_declaration(declaration.key, declaration.make_data(FIRST_PACKET))
        return declaration

    # -- 版本锚 ----------------------------------------------------------

    def _mint_version(self, key: _Key) -> int:
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        return version

    def _absorb_version(self, data: ParameterData) -> int | None:
        """
        采纳裁决. 返回应广播的版本; None = 不采纳 (调用方仍须回播一次当前真值).

        版本按 epoch 作用域: 跨 epoch (含空 epoch) 的声明, 其版本不可比 — 视为
        首包, 从头铸版 (本 epoch 已有真值则拒绝); 同 epoch 才做单调比较.
        version 三值见模块 docstring.
        """
        key = data.key
        current = self._versions.get(key, 0)
        if data.epoch != self._epoch:
            if current > 0:
                return None
            return self._mint_version(key)
        if data.version > FIRST_PACKET:
            if data.version < current:
                return None
            self._versions[key] = data.version
            return data.version
        if data.version == FIRST_PACKET:
            if current > 0:
                return None
            return self._mint_version(key)
        return None  # REQUIRE_ONLY — 纯 require, 不采纳.

    # -- 真相 ------------------------------------------------------------

    def _stamp(self, data: ParameterData, version: int) -> ParameterData:
        data.version = version
        data.epoch = self._epoch
        data.meta = ParameterMeta(address=self._address, host=True)
        return data

    async def _broadcast_truth(self, data: ParameterData) -> None:
        self._truth[data.key] = data
        subscriber = self._subscribers.get(data.key)
        if subscriber is not None:
            subscriber.set_truth_data(data)
        await self._broadcaster.publish_host_truth(data)

    async def _on_declaration(self, data: ParameterData) -> None:
        """declaration 进 → truth 必出. 采纳则广播它, 不采纳则广播当前真值一次."""
        version = self._absorb_version(data)
        if version is None:
            current = self._truth.get(data.key)
            if current is not None:
                await self._broadcaster.publish_host_truth(current)
            return
        await self._broadcast_truth(self._stamp(data, version))

    # -- AbsParameters 钩子 ----------------------------------------------

    async def _publish_declaration(self, key: _Key, data: ParameterData) -> None:
        """host 自身的写 = 真相, 铸新版本."""
        await self._broadcast_truth(self._stamp(data, self._mint_version(key)))

    async def _subscribe_parameter(
            self, *, key: _Key, subscriber: BaseParameterSubscriber,
    ) -> _Disposer:
        # host 的真值在本地, 不经 transport.
        def _noop() -> None:
            pass

        return _noop

    # -- 生命周期 --------------------------------------------------------

    def _when_declaration(self, data: ParameterData) -> None:
        """transport 线程回调 → 队列 (线性卸载)."""
        try:
            self._declaration_queue.sync_q.put_nowait(data)
        except janus.SyncQueueShutDown:
            pass

    async def _absorb_loop(self) -> None:
        while self.is_running():
            try:
                data = await self._declaration_queue.async_q.get()
                await self._on_declaration(data)
            except asyncio.CancelledError:
                raise
            except janus.AsyncQueueShutDown:
                break
            except Exception:
                self._logger.exception("failed absorbing declaration")

    async def __aenter__(self) -> Self:
        self._started = True
        await self._broadcaster.__aenter__()
        self._broadcaster.set_host(self._address, self._epoch)
        self._declarations_disposer = await self._broadcaster.subscribe_declarations(self._when_declaration)
        self._publish_task = asyncio.create_task(self._publish_declaration_loop())
        self._absorb_task = asyncio.create_task(self._absorb_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._closed:
            return
        self._closed = True
        for task in (self._publish_task, self._absorb_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._publish_task = None
        self._absorb_task = None
        if self._declarations_disposer is not None:
            self._declarations_disposer()
            self._declarations_disposer = None
        await self._broadcaster.__aexit__(exc_type, exc_val, exc_tb)


class WorkerParameters(AbsParameters, ABC):
    """
    非真相节点 (worker) 的 parameters.

    - **懒启动**: 不随 session 启动打开通讯; 首个 subscriber (declare 亦计) 出现时,
      才打开 broadcaster (队列 + zenoh) 并开始收真值. 没有参数的节点不付通讯成本.
    - **host 在线**: 真值来自 host 广播; 本地 set 先本地生效再上报, 由 host 定序后
      广播回. 此时本地值只是"预期", 不是真相.
    - **host 离线**: 无仲裁, 本地值即真相; 上报不外发 (无人定序).

    振荡闸口: truth 到达只更新本地, 绝不触发 declaration; 上报只由 ``set()`` 与 host
    化身变更触发, 后者以 (address, epoch) 幂等门锁住.
    """

    def __init__(
            self,
            address: _Address,
            broadcaster: ParametersBroadcaster,
            logger: LoggerItf | None = None,
    ):
        super().__init__(meta=ParameterMeta(address=address, host=False), logger=logger)
        self._address = address
        self._broadcaster = broadcaster
        self._broadcast_started = False
        self._publish_task: asyncio.Task | None = None
        self._closed = False
        # 当前认定的 host 化身. 空 = 尚未发现 host.
        self._host_address: _Address = ""
        self._host_epoch: _Epoch = ""

    def is_host(self) -> bool:
        return False

    def is_running(self) -> bool:
        return self._started and not self._closed

    # -- 懒启动 ----------------------------------------------------------

    async def _ensure_broadcast_started(self) -> None:
        """首个 subscriber 触发, 幂等."""
        if self._broadcast_started:
            return
        self._broadcast_started = True
        await self._broadcaster.__aenter__()
        await self._broadcaster.on_host_alive(self._when_host_alive)
        self._publish_task = asyncio.create_task(self._publish_declaration_loop())

    # -- AbsParameters 钩子 ----------------------------------------------

    async def _subscribe_parameter(
            self, *, key: _Key, subscriber: BaseParameterSubscriber,
    ) -> _Disposer:
        """
        订阅 host 真值, 并发一条声明 (declaration 进 → truth 必出).

        declare 以首包 (0) 提议自身值; 纯订阅者以 require (-1) 只要一次广播.
        """
        await self._ensure_broadcast_started()
        disposer = await self._broadcaster.subscribe_host_truth(key, subscriber.set_truth_data)
        version = FIRST_PACKET if isinstance(subscriber, ParameterDeclaration) else REQUIRE_ONLY
        await self._broadcaster.publish_declaration(subscriber.make_data(version))
        return disposer

    async def _publish_declaration(self, key: _Key, data: ParameterData) -> None:
        """上报本地写, 只交给 host 定序. 尚未发现 host 时本地即真相, 不外发."""
        if not self._host_address:
            return
        await self._broadcaster.publish_declaration(data)

    # -- host 生命周期 ----------------------------------------------------

    def _when_host_alive(self, address: _Address, epoch: _Epoch) -> None:
        """
        host 上线或换化身: 相位 B 收敛.

        推的是本地值, 不是刚被广播回来的真值 (振荡闸口); (address, epoch) 幂等门
        保证每次化身只推一轮. 继承版本, 不 +1 — 见 ``reannounce_data``.
        """
        if address == self._host_address and epoch == self._host_epoch:
            return
        self._host_address = address
        self._host_epoch = epoch
        for key, declaration in list(self._declarations.items()):
            self._set_parameter(key, declaration.reannounce_data())

    # -- 生命周期 --------------------------------------------------------

    async def __aenter__(self) -> Self:
        # 不因 session 启动而打开通讯 — 等首个 subscriber.
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._closed:
            return
        self._closed = True
        if self._publish_task is not None and not self._publish_task.done():
            self._publish_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._publish_task
            self._publish_task = None
        if self._broadcast_started:
            await self._broadcaster.__aexit__(exc_type, exc_val, exc_tb)
