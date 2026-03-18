from typing import TypeVar, Generic, Type, Callable, Coroutine, Literal, TypedDict, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_common.identifier import Identifier
from pydantic import BaseModel, Field, ValidationError
from ghoshell_common.helpers import uuid
from ghoshell_moss.core import TopicModel
import datetime
import time

"""
# Event 介绍

Ghost 思维框架通过 Event 来管理并行思维节点的通讯. 

Event 本质上分为几类: 
1. 来自躯体 shell 的输入, 通过 Channel 的 Topics 广播分发给 Ghost. 是可以通过 Channel 协议定义的. 
2. 其它 UI 设备的输入. 
3. 来自并行思维链路的信息交换. 

所有的 Event 在事件总线 EventBus 中流转, 由不同的节点来消费. 

# Event 在并行思考架构中的作用. 

在并行思考架构中, 要考虑的通讯需求通常有: 

1. actor: 请求 + 返回. 
2. queue: 有序消费. 具体消费逻辑可能有 worker 的概念. 但是队列本身不关心. 
3. parameters: 动态数据的共享, 读或写. 

可以认为一个 Ghost 运行的时候, 它能使用的: 
1. Actor
2. Event
3. Parameter 
都是协议化的. 这个架构理念会高度类似 ROS2 . 协议本身定义了拓扑, 但是由开发者去设计拓扑的实现. 

Event 机制主要解决其中的 queue 相关的逻辑. 常见消费逻辑有: 

0. concurrent: 设计 1~n 个 worker 并行消费. 
1. priority queue: 优先级消费, 不丢弃消息. 
2. latest / oldest: 超过 maxsize 外的消息就丢弃, 不过要决定丢弃最新的, 还是最老的.  
3. context buffer + Scheduler: 将消息加工后缓存为一个上下文, 由生命周期决定合适消费. 

进一步的还有 QoS 的各种设计. 这些只能在迭代中完善. 

# Event 不区分内外部. 

注意在这个实现中, 并没有在抽象层隔离掉来自外部世界的输入 (Event) 和思维状态中的流转交互 (MindTopic). 
这种隐患是: 恶意躯体组件 (Channel) 能够发送 Event 污染内部思考链路, 造成破坏. 

这么做的动机是降低链路开发成本, 不区分思维节点本身的性质. 参考 ROS2, 也并不在抽象上区分 Sensors/Body 等. 
 
现阶段的解决策略是:
1. 来自 Shell 的 Event, 都被记录为 `Shell/{name}` 作为 issuer 
2. 来自 Mind 的 Event, 都被记录为 `Mind/{mind_node_name}` 作为 issuer. 

# 消费者的实现

消费 Event 的节点, 理论上只要做三件事: 
1. 监听 Event, 并且按自己的逻辑策略 (queue, priority queue, latest, context buffers) 管理. 
2. 消费 Event 逻辑, 执行有副作用的操作, 副作用操作会跨进程共享. 
3. 发送新的 Event, 激活思维链路. 

所以 Event 的流转本身构成了思维的拓扑图形, 以及思维的状态过程. 
需要有一套监控机制, 可以观测思维拓扑, 以及思维状态 (Event 的发生和流转).

底层考虑 Zenoh 等框架, 用类似 ROS2 的方式完成监控. 

# 序列化约定

1. 进程内通信：直接传递Python对象
2. 进程间通信：使用JSON序列化（通过model_dump_json()）
3. 未来可能支持：MessagePack、Protocol Buffers

# 实现屏蔽

Event 设计目标是屏蔽底层具体的实现. 
而具体的实现, 目前规划通过 Zenoh 等多进程通讯来替代 (曾考虑过 Ray, 不过太重了).

* Event 需要是自解释的, 基于 code as prompt 原则, 用 pydantic BaseModel 做自解释. 
* Event 是可传输, 最好语言无关. 所以实际传输协议会用 json. Python 版框架提供默认实现. 

"""

EventName = str


class EventMeta(BaseModel):
    """
    Event 的元信息. 在传输和路由时均可使用.
    """
    id: str = Field(
        default_factory=uuid,
        description="全局的唯一 id",
    )
    issuer: str = Field(
        default="",
        description="发送者. "
    )
    issuer_id: str = Field(
        default="",
        description="发送者的唯一 id. "
    )
    event_type: str = Field(
        default='',
        description="事件的类型, 对应 event model"
    )
    priority: int = Field(
        default=0,
        description="事件的优先级. 但如果不按优先级消费就没有用.",
    )
    event_name: str = Field(
        default="",
        description="事件的分发目的地. 可能很多个 event name 对应同一个 event type.",
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
    )
    overdue: float = Field(
        default=0,
        description="事件的过期策略. > 0 时 用于判断一个事件是否过期. "
    )


class Event(BaseModel):
    """
    在 Ghost 事件总线中广播的数据对象.
    """

    meta: EventMeta = Field(
        default_factory=EventMeta,
        description="基础讯息",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="对应 EventModel 的数据结构定义. "
    )

    def is_overdue(self) -> bool:
        """
        过期判断.
        """
        if self.meta.overdue <= 0:
            return False
        elapsed = time.time() - self.meta.created_at.timestamp()
        return elapsed > self.meta.overdue


class GhostEventTopic(TopicModel):
    """
    支持 MOSS 里的 Channel 通过这个 Topic 与 Ghost 直接通讯.
    而不用通过其它链路.

    之所以 GhostEvent 和 MOSS Topic 非常雷同但异构, 一个基本原因是:
    Ghost 实现可以不依赖 MOSS. MOSS 可以不用在 Ghost 里.
    """

    ghost_event: Event = Field(
        description="将 Ghost Event 封装成 MOSS 协议的 Topic. "
    )

    @classmethod
    def topic_type(cls) -> str:
        return "ghost/event"

    @classmethod
    def default_topic_name(cls) -> str:
        return "ghost/event"

    @classmethod
    def from_ghost_event(cls, ghost_event: Event) -> Self:
        return cls(ghost_event=ghost_event)


class EventModel(BaseModel, ABC):
    """
    对事件强类型数据结构的建模.
    也是一种协议手段. 以 JSON Schema 作为基础协议.
    """
    meta: EventMeta = Field(
        default_factory=EventMeta,
        description="用于初始化, 或者还原 event 现场. "
    )

    @classmethod
    @abstractmethod
    def event_type(cls) -> str:
        """
        事件的类型描述, 全局唯一.
        预计用 `foo/bar` 的方式定义.
        """
        pass

    @classmethod
    def default_event_name(cls) -> str:
        """
        事件的默认地址, 预计用 `foo/bar` 来描述.
        约定优先于配置, 默认用 event type 作为 default event name.
        """
        return cls.event_type()

    @classmethod
    def from_event(cls, event: Event, throw: bool = False) -> Self | None:
        if event.meta.event_type != cls.event_type():
            return None
        try:
            meta = event.meta.model_copy()
            model = cls(meta=meta, **event.data)
            return model
        except ValidationError:
            if throw:
                raise
            return None

    def to_event(
            self,
            *,
            event_name: str | None = None,
            overdue: float | None = None,
            priority: int | None = None,
    ) -> Event:
        """
        生成一个
        """
        meta = self.meta.model_copy()
        if overdue is not None:
            meta.overdue = overdue
        if priority is not None:
            meta.priority = priority
        meta.event_type = self.event_type()
        meta.event_name = event_name or self.default_event_name()
        return Event(meta=meta, data=self.model_dump(exclude_none=True))


EVENT_MODEL = TypeVar('EVENT_MODEL', bound=EventModel)


class Publisher(Generic[EVENT_MODEL], ABC):
    """
    事件的发送者, 本质上是实现一个声明. 让进程级别的 EventBus 理解自己的发送模式.
    """

    @abstractmethod
    async def publish(self, event: EVENT_MODEL) -> None:
        """
        发布事件.
        """
        pass


SubscriberMode = Literal['queue', 'priority']
"""作为语法糖, 定义事件的监听模式, 提供内置的处理规则. 逐步迭代. """


class Subscriber(Generic[EVENT_MODEL], ABC):
    """
    事件的监听者. 提供一部分语法糖, 完成最基本的实现.
    更多的状态相关抽象, 等迭代时再增加.

    本质上 Subscriber 在监听广播, 但同时将广播的结果按模式做队列化, 交给 Handler 去运行.
    """

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否正在运行中.
        """
        pass

    @abstractmethod
    def mode(self) -> SubscriberMode:
        """
        返回当前 Subscriber 的模式.
        """
        pass

    @abstractmethod
    async def work(
            self,
            handler: Callable[[EVENT_MODEL], Coroutine[None, None, None]],
            *,
            raise_exception: bool = False,
    ) -> None:
        """
        可以用来在协程环境下创建一个 asyncio.Task, 持续性地消费 EVENT MODEL
        每个 Work 的调用都是串行阻塞的. 消费完一个以后, 才能消费另一个.
        究竟创建几个 Worker, 由开发者决定好了.

        :param handler: asyncio 的 handler.
        :param raise_exception: 如果为 True 的话, 当一个 handler 运行一次事件失败, 就会抛出, 并且停止这个 work.
        """
        pass


class EventBus(ABC):
    """
    Ghost 的事件总线. 用来管理所有的事件获取和分发.

    一个核心设计原则是, EventBus 是跨进程可用的. 每个进程实际上会实现一个独立的 Eventbus.
    每个独立的 Eventbus 的 Identifier 也不一样.

    如果每个进程中启动的 EventBus 将状态汇总到一起的话, 则可以构成一个以事件为边, 以 Identifier 为节点的拓扑图.

    Eventbus 广播与监听的基本原则是:
    1. 需要先声明 Publisher 和 Subscriber (这样才能保留状态).
    2. 进程内监听自身的广播, 通过内存通讯. 进程间通过进程间协议 (比如 Zenoh).

    Eventbus 的接口设计有个基本原则:
    1. subscriber & publisher 不是线程安全的. 而且在协程环境里运行.
    2. Eventbus 本身是线程安全的.
    """

    @abstractmethod
    def identifier(self) -> Identifier:
        """
        自解释模块. 本地发送的事件, issuer 的标记会来自 identifier.
        """
        pass

    @abstractmethod
    def publishing(self) -> list[EventName]:
        """
        当前可能发布的 EventName.
        可以用来构建一个图谱.
        """
        pass

    @abstractmethod
    def subscribing(self) -> list[EventName]:
        """
        当前正在监听的 Event.
        可以用来构建一个图谱.
        """
        pass

    @abstractmethod
    def new_subscriber(
            self,
            event_model: Type[EVENT_MODEL],
            *,
            event_name: str | None = None,
            mode: SubscriberMode = 'queue',
            maxsize: int = 0,
            keep: Literal['latest', 'oldest', 'priority'] = 'priority'
    ) -> Subscriber[EVENT_MODEL]:
        """
        创建 Subscriber.
        """
        pass

    @abstractmethod
    def new_publisher(
            self,
            event_model: Type[EventModel],
    ) -> Publisher[EVENT_MODEL]:
        """
        声明式创建一个 Publisher.
        目标仍然是更新 publishing, 从而可以用来构建运行时图谱.
        """
        pass
