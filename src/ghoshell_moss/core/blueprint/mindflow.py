"""Mindflow — perception, thought, and action arbitration for concurrent duplex state management."""
from typing import Callable, Iterable, AsyncIterator, Any, Awaitable

from typing_extensions import Self, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime, ValidationError
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message, ContextType
from ghoshell_moss.message import unique_id
from ghoshell_container import IoCContainer
from .moment import Results, Moment, Moments, Observer
import datetime
import dateutil
import time
import asyncio
import enum

# 持久化智能体认知行为 三循环:
# 1. 思考循环: 模型接受信息, 思考并输出.
# 2. 感知循环: 接受外部世界各种感知信号, 产生冲动.
# 3. 执行循环: 执行流式指令, 同时获取流式的反馈.
# 双工:
# 1. 感知 -> 思考: 思考输出的同时, 感知在输入, 都是流式的.
# 2. 思考 -> 执行: 思考产生 token 的同时, 流式解释器立刻执行, 并且同时产生指令结果.
# 3. 执行 -> 结果: 当执行行为在外部世界产生效果, 会反馈到感知链路.
#
# 在这种场景下, 涉及一个复杂的状态管理体系.
# 1. 观测: 来自三个循环的信息需要有序记录, 可以观测.
# 2. 时序: 三循环的执行逻辑要对齐. 避免思维奔逸 (拿到反馈前就继续行动) 和裂脑 (感知/思考/行为消费不同时间轴上的信息.)
# 3. 中断: 来自三方的信号可能触发中断, 如高优打断事件, 模型调度异常, 执行错误指令等.
# 4. 结束: 状态需要有序地结束.
#
# 在当前 Mindflow 的体系中, signal + impulse + nucleus 是对感知的隔离建模, 预期用可迭代的单元将它们分割出去.
# Thinking + Articulator + Action 是运行状态的管理调度体系.
# 好在实现底层用多进程模型隔离.

__all__ = [
    'Priority',
    'SignalName', 'Signal', 'SignalMeta', 'InputSignalMeta', 'SignalSchema',
    'Impulse',
    'Logos', 'Moment', 'Results',
    'Action', 'Thinking',
    'Nucleus', 'NucleusMeta',
    'Mindflow', 'MindflowHook',
    'Attention',
    # 几个关键的通讯信号, 用来快速终止一些循环.
    'ChallengeVerdict',
    'ThinkingEffort',
    'ChallengeMode',
    'ImpulsePrimitive',
    'Articulator',
    'ActionGate',
    'StatementExitedException', 'ActionExitedException', 'AttentionExitedException', 'ThinkExitedException',
]

SignalName = str


class Priority(int, enum.Enum):
    """
    为了避免优先级无限膨胀, 因此做策略约定.
    """
    BACKGROUND = -1  # 永远不能挑战成功任何当前注意力.
    INFO = 0  # 基础值.
    NOTICE = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5  # 约定的最高级别, 永远抢占成功.


class Signal(BaseModel):
    """
    端侧发送给智能体响应的信号. 可能有以下几个关键特征:
    1. 多源头, 比如视觉/听觉/触觉/故障/通讯/异步回调....
    2. Partial, 典型的例子是 ASR 的首包到尾包, 每个分句都是一个 Partial 包.
    3. 保鲜, 过期的信号会直接丢弃.
    4. 以 AI 可以理解的消息为优先.
    """

    __state__: Literal['created', 'pending', 'dispatched', 'ignored'] = 'created'
    """内部用于 debug 的参数"""

    name: SignalName = Field(
        description="the signal name, if not match any mind pulse, the signal will be ignore",
    )
    id: str = Field(
        default_factory=unique_id,
        description="unique identifier of the signal",
    )
    trace_id: str = Field(
        default='',
        description="the trace id of the signal. 通常系统自动标记, 不需要传值. ",
    )
    complete: bool = Field(
        default=True,
        description="whether the signal complete or partial."
                    "如果是 partial 包, 应该后续传递 complete = True 的尾包."
                    "但 partial 包仍然有存在意义, 比如打断, 占据注意力等. 举个例子, "
                    "一个高优的 ASR 首包打断了 AI 行为, 同时占据了注意力."
                    "抽象设计上不做粘包逻辑. 如果有粘包的需要, 需要结合 Nucleus 定义内部协议.",
    )
    max_hop: int = Field(
        default=1,
        description="maximum hop number, 为 0 不传播. 系统内部调度时会处理. 不应该修改它. Mindflow 内部使用这个字段. ",
    )
    issuer: str = Field(
        default="",
        description="the issuer of the signal, 不需要显示传递, 实际链路发布时会添加.",
    )
    priority: Priority = Field(
        default=Priority.INFO,
        description="信号的优先级, 越大优先级越高. 用于做抢占式调度. 来自边缘系统的输入本身应包含第一轮优先级"
    )
    strength: int = Field(
        default=100,
        description="信号的强度. 输入信号在 0~300 之间做设计, 常态位是 100. 通常直接用默认值即可."
                    "因为信号的衰减逻辑在 Attention 中设计, 所以在不耦合 attention 的情况下, 对信号强度的理解就按百分比处理."
                    "比如 100 * 1.2 表示加权 20%. ",
        ge=0,
        le=300,
    )
    description: str = Field(
        default='',
        description="short description of the signal."
                    "这个字段是可省略的. 它的作用是在极简的 Nucleus 实现中, 直接用它提示状态. "
                    "类似 IM 里红点展示的用户消息, 会保留一个缩略的一句话提示. ",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="被处理过的消息体.",
    )
    hint: str = Field(
        default='',
        description="the hint of how to handle the signal."
                    "hint 也是可选的实现. 默认为空即可. 它的作用是一种补丁. 当一个输入进来时, 模型很可能按预训练约定去理解."
                    "典型案例如 图片, 模型会默认认为这是在 IM 里提交的一张照片. 而不知道这是自己的 vision. "
                    "这时就可以用补丁; 为什么拆到 prompt 字段呢? "
                    "因为 prompt 对多轮对话而言是一定要丢弃的; 放入 messages 里, 会导致上下文里被 prompt 补丁淹没. ",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="meta data of the signal follow the protocol of the name."
                    "可扩展的强类型约定, 通过 SignalMeta 可以提供一个 JSON Schema 协议去定义细节. ",
    )
    stale_timeout: float = Field(
        default=0,
        description="the stale signal will be ignored. ",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
    )

    @classmethod
    def new(
            cls,
            name: SignalName,
            *messages: Message,
            priority: Priority = Priority.INFO,
            description: str = '',
            metadata: dict[str, Any] | None = None,
            strength: int = 100,
            stale_timeout: float = 0,
            complete: bool = True,
    ) -> Self:
        return cls(
            name=name,
            messages=list(messages),
            priority=priority,
            description=description,
            metadata=metadata or {},
            strength=strength,
            stale_timeout=stale_timeout,
            complete=complete,
        )

    def priority_strength(self) -> int:
        return self.priority * 1000 + self.strength

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def to_json(self, indent: int = 0) -> str:
        # 传输数据类型取最小信息.
        return self.model_dump_json(indent=indent, exclude_none=True, exclude_defaults=True, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        # 传输数据类型取最小信息.
        return self.model_dump(mode='json', exclude_none=True, exclude_defaults=True)

    def __repr__(self):
        return f"<Signal id={self.id} trace={self.trace_id} name={self.name}>"


class SignalSchema(BaseModel):
    name: str = Field(
        description="signal name"
    )
    description: str = Field(
        description="signal description"
    )
    default_priority: int = Field(
        description="signal default priority"
    )
    metadata_schema: dict[str, Any] = Field(
        description="json schema of the signal meta data"
    )


class SignalMeta(BaseModel, ABC):
    """
    定义一个 Signal 的补充协议 (围绕 metadata), 用于在环境中被发现, 从而可以做到自解释.
    所有字段应该都是支持序列化的, 否则会在传输时报错.
    同时 Pydantic BaseModel 定义的 Signal Meta 可以作为协议被发现, 提供 metadata 的 json schema 协议.

    **字段设计三尺度**

    1. **功能性** — 字段有 nucleus (未来或当下) 的判决用途. 无判决用途 =
       不该加. metadata 是 nucleus 的判决依据, 不是消息展示位.
    2. **易生产** — 生产侧 (channel / listener) 天然拿到, 不为了填字段
       而额外做工作 (查库、拼字符串、格式化). 生产成本高的字段几乎一定
       是设计错位.
    3. **未来语义** — 现在不用没关系, 但要能预见 nucleus 什么时候会用它
       做分档、去重、抢占. 说不出未来用途 = 不该加.

    **常见错位** : 把 "给 ghost 看的消息内容" 塞进 metadata —
    退出码、错误尾、诊断入口路径、状态截图 URL 之类. 这些属于**消息主体**,
    通过 to_signal(messages=..., description=...) 承载, 不进 metadata.
    metadata 只应有让 nucleus 做出正确判决的最小信息.
    """

    @classmethod
    def to_signal_schema(cls) -> SignalSchema:
        return SignalSchema(
            name=cls.signal_name(),
            description=cls.__doc__ or '',
            default_priority=cls.priority(),
            metadata_schema=cls.model_json_schema(),
        )

    @classmethod
    @abstractmethod
    def signal_name(cls) -> SignalName:
        """定义唯一的 signal 名称. """
        pass

    @classmethod
    def priority(cls) -> Priority:
        """默认的优先级"""
        return Priority.INFO

    @classmethod
    def match(cls, signal: Signal) -> bool:
        return signal.name == cls.signal_name()

    @classmethod
    def from_signal(cls, signal: Signal) -> Self | None:
        """
        快速做 signal metadata 的数据还原加工

        典型用法:
        >>> def match_signal(s: Signal):
        >>>     if input_signal := InputSignalMeta.from_signal(s):
        >>>        ...
        """
        if cls.signal_name() != signal.name:
            return None
        try:
            metadata = signal.metadata
            return cls.model_validate(metadata)
        except ValidationError:
            return None

    def to_signal(
            self,
            *messages: ContextType,
            description: str = '',
            stale_timeout: float = 0,
            priority: int | None = None,
            hint: str = '',
    ) -> Signal:
        """快速用 meta 定义一个 signal. 提示两者的使用机制. """
        name = self.signal_name()
        wrapped_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                wrapped_messages.append(msg)
            else:
                wrapped_messages.append(Message.new().with_content(msg))
        priority = self.priority() if priority is None else priority
        return Signal(
            name=name,
            messages=wrapped_messages,
            metadata=self.model_dump(exclude_defaults=True, exclude_none=True),
            description=description,
            stale_timeout=stale_timeout,
            priority=priority,
            hint=hint,
        )


class InputSignalMeta(SignalMeta):
    """
    系统最基础的 Input 讯号. 代表一个明确的输入.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'input'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


# Impulse 发送的控制原语, 结合系统约定实现 Impulse 决策的控制功能.
# 高阶实现中 Primitive 本身是可以做扩展定义的.
# mode 不经过大脑处理, 是系统级别的处理. 主要的原语都在决定 思考/行为的基本逻辑.

class ChallengeMode(str, enum.Enum):
    """Impulse 与当前 attention 仲裁后的处置模式. 用于声明 Impulse 仲裁后的处理方式. """

    # 默认: 抢占成功创建新 attention, 失败 suppress (需要压抑 impulse, 避免频繁挑战注意力).
    default = '',

    # 抢占成功只 inject messages, 不创建新 attention; 抢占失败仍 suppress.
    # 用例: 高优广播 (FATAL + silent) — 必送达但不接管思维.
    # 类似传统 Agent 的 inject messages.
    silent = 'silent'

    # 抢占失败时 buffer messages 而非 suppress; 抢占成功正常创建新 attention.
    # 用例: 消息绝不能丢, 但可以不响应的情况. 比如连续的语音输入 (NOTICE + notify)
    notify = 'notify'


# Impulse 对应的决策倾向. Impulse 是预处理的思维状态, 相当于一种条件反射产生的思维倾向.
# default == '', 表示思考侧自行决定使用那种响应.
# none 则是表示思维不用对此做响应. 仍然可以用于打断注意力, 提供信息等.
ThinkingEffort = Literal['none', 'flash', '', 'low', 'medium', 'high', 'max']


class Impulse(BaseModel):
    """
    the impulse that raise mindflow attention
    Impulse 可以是 Nucleus 加工后的产物, 也可以是 Signal 的原样复制 (极简情况下).
    它的核心目的是将原始信号转换成更明确的调度信号.
    """
    id: str = Field(
        default_factory=unique_id,
        description="the impulse id",
    )
    source: str = Field(
        default='',
        description="the nucleus source name",
    )

    priority: Priority | int = Field(
        default=Priority.NOTICE,
        description="the impulse priority",
    )

    complete: bool = Field(
        default=True,
        description="if the impulse is complete, or just occupy the attention until complete impulse from the same id."
                    "可以用来使用首包抢占注意力, 尾包响应的场景.",
    )
    description: str = Field(
        default='',
        description="the impulse short description. 这个描述可以理解为 IM 消息列表上的摘要. ",
    )
    dynamic_messages: list[Message] = Field(
        default_factory=list,
        description="the impulse perspective, 伴随决策携带."
                    "本帧实时字段: 走 ChallengeMode 的 buffer 路径 (silent/notify) 时该字段被丢弃.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="the messages of the impulse. if empty, no need to think",
    )

    # --- 高级特性 --- #

    hint: str = Field(
        default='',
        description="the temporary instruction for model handling this impulse."
                    "本帧实时字段: 仅在 Impulse 获得 attention 时通过 update_moment 落到 moment.hint."
                    "走 ChallengeMode 的 buffer 路径 (silent/notify) 时该字段被丢弃.",
    )
    mode: str | ChallengeMode = Field(
        default='',
        description="Impulse 作为一种预处理思维模式, 通过原语和 Runtime 的规则通讯."
                    "规则可以自行扩展, 系统提供基线. 规则优先级高于大脑思考, 属于条件反射. "
                    "见 ChallengeMode 的对称表理解 silent/notify 的偏离语义.",
    )
    logos: str = Field(
        default='',
        description="伴随 Impulse 发送的 Logos, 可以是条件反射, 强制指令, 首动作提速 (口头禅) 等等."
                    "当 Impulse 获得了注意力时, 应该伴随发送到 Articulator, 由 Articulator 决定是否直接发送给 Action."
                    "如果作为 '反射弧' 直接发送, 则它会先于 思考帧生成 logos, 就发送给 Action 侧."
                    "这样先于思考就会有 logos 发送. 大脑也应该感受到它 (或像人一样意识不到小动作), 取决于具体实现."
                    "本帧实时字段: 走 ChallengeMode 的 buffer 路径 (silent/notify) 时该字段被丢弃 (跨 attention 没意义).",
    )
    interrupt: bool = Field(
        default=False,
        description="高级系统特性, 会在思维决策前停止所有执行中的 logos."
                    "如果整个躯体体系有平滑过度逻辑 (Idle), stop first 看起来像停止 (呆了一下). "
                    "如果没有任何平滑过度逻辑, 会产生类似 Shock/Frozen 的 震惊效果."
                    "如果为 False, 实际上 Ghost 仍然可以走快速决策 -> 详细回复, 通过快速决策做机制."
    )
    thinking_effort: ThinkingEffort = Field(
        default='',
        description="思考的强度, 作为不同输入逻辑的 '建议' 处理模式."
                    "实际上执行 articulator 的智能体仍然有权决定自己的处理逻辑. ",
    )
    stale_timeout: float = Field(
        default=0.0,
        description="当一个 Impulse 无法占据到 Attention 时, 可以定义过期时间在它未被及时清理时也不会生效."
    )
    protection_time: float = Field(
        default=0.0,
        description="显性的相同优先级保护期, 当获得注意力后, impulse 在保护期内, 不可以被相同优先级打断, 与强度无关."
                    "用于防抖.",
    )
    strength: int = Field(
        default=100,
        description="the impulse 初始强度, 在 attention 中设计强度计算曲线用来解决相同优先级打断机制."
                    "用于相同级别任务的优先级仲裁. 其权重值要么就是系统整体严格约定, 要么就是有通用评级仲裁."
                    "否则不需要特殊约定. 取值范围数字越大越强."
                    "为 0 表示**绝不竞争**: 只能完成初始化. "
                    "不参与任何 mode 的 buffer/suppress 分支, 由 nucleus 自然清理."
                    "构造首包 (占住 attention 不竞争) + protection_time 后发尾包的'冷静期'语义.",
        ge=0,
        le=1000,
    )
    strength_decay_seconds: float = Field(
        default=20,
        description="Strength decay 约定时间. 以秒为单位. "
                    "语义是, 当一个 Impulse 开始运行后, 如果没有任何动静, 最迟在这个数字时强度必须归零."
                    "无论 Attention 的仲裁曲线如何规划, 都要遵循这个约定.",
    )

    # -- 系统内部字段 -- #
    source_idx: int = Field(
        default=0,
        description="the impulse generated order in the source",
    )

    trace_id: str = Field(
        default='',
        description="the impulse trace id, 向上溯源.",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="the creation time of the impulse",
    )

    @classmethod
    def from_signal(cls, signal: Signal, source: str, stale_timeout: float | None = None) -> Self:
        """
        一个简单的示例, 直接将 signal 转化成 impulse 不做任何处理.
        实际上 Impulse 并不见得来源于单一 Signal. 这种涉及只为了通讯使用.
        """
        stale_timeout = stale_timeout if stale_timeout is not None else signal.stale_timeout
        if stale_timeout > 0:
            stale_timeout = stale_timeout - (time.time() - signal.created_at.timestamp())
        # 从 signal 直接反射 impulse 的做法, signal 相当于原始协议. 所以剥离了 impulse 所有高阶机制.
        return Impulse(
            source=source,
            trace_id=signal.trace_id or signal.id,
            priority=signal.priority,
            strength=signal.strength,
            messages=signal.messages.copy(),
            description=signal.description,
            hint=signal.hint,
            complete=signal.complete,
            stale_timeout=stale_timeout,
        )

    def priority_strength(self) -> int:
        """结合优先级产生的权重值, 用于比较两个 impulse. """
        return self.priority * 1000 + self.strength

    def is_stale(self) -> bool:
        """是否过期. """
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def to_dict(self) -> dict:
        return self.model_dump(mode='json', exclude_defaults=True, exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(exclude_defaults=True, exclude_none=True, ensure_ascii=False, indent=indent)

    def update_moment(self, moment: Moment) -> None:
        """
        将 Impulse 的数据更新 Moment.
        """
        if self.dynamic_messages:
            # 用 source 源, 占据一个 perspective.
            moment.with_dynamic_context(self.source, self.dynamic_messages)
        moment.with_percepts(self.source, self.messages)
        moment.hint = self.hint
        if self.logos:
            moment.command_logos += self.logos

    def __repr__(self):
        return f"<Impulse id={self.id} trace={self.trace_id} source={self.source}>"


class Nucleus(ABC):
    """
    并行 感知/思考/决策 单元的统一抽象. 它接受输入信号, 返回动机, 属于 “单生产者-单消费者”的有界缓冲区
    在输入场景中, 它是输入信号的治理层, 用于将高频的输入信号治理/加工/降频/加权后, 转化为 Mindflow 可以处理的 Impulse.
    可以拥有各种实现机制, 比如:
    1. lru buffer, 将所有的信号合并
    2. summary, 将信号合并摘要
    3. priory queue, 结合 maxsize 做单一信号量.
    4. arbiter, 加入仲裁者模型做快速校验.
    5. sidecar, 旁路思考, 向主路广播...

    同样, 它可以作为 MultiTasks/Planner/Timer/Ticker/MultiAgent 等各种机制, 通过 signal 和 impulse 两个大一统抽象管理特别复杂的
    异步通讯逻辑, 与主交互脑通讯. 理想情况下它不应该包含调度逻辑, 而只作为通讯调度层.
    """

    @abstractmethod
    def name(self) -> str:
        """
        用于区分不同的 Nucleus 单元.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        所有的 Nucleus 都应该是自解释的, 而且这个自解释要足够高效, 能一句话自我描述.
        """
        pass

    @abstractmethod
    def status(self) -> str:
        """
        当前 Nucleus 的状态提示, 参考 IM 的消息红点, 要简短而精准.
        如果为空, 会被忽略.
        """
        pass

    @abstractmethod
    def signals(self) -> list[SignalName]:
        """
        声明监听的信号类型.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        排空讯号, 应该强制清空所有状态.
        用于做极限故障下的还原, 作为最基础的恢复手段.
        """
        pass

    @abstractmethod
    def add_signal(self, signal: Signal) -> None:
        """
        接受一个信号量, 在内部开始执行校验逻辑, 生成 impulse.
        没有背压, 应当尽可能快地入队或丢弃，不执行任何耗时或异步操作。内部应有独立的任务循环消费队列。
        """
        pass

    @abstractmethod
    def with_bus(self, signal_broadcast: Callable[[Signal], None], fire_impulse: Callable[[Impulse], None]) -> None:
        """
        注册总线, 可以广播信号, 或者发送 impulse.
        1. Nucleus 可以广播 signal 给其它监听者.
        2. Nucleus 产生了 Impulse, 可以回调通知, 比如回调 Mindflow.
        注意, Impulse 回调时不能 pop, 如果回调的 Impulse 无法抢占 attention, 应该会收到一个 suppress 信号.

        关于通讯, 目前设计上 Nucleus 和 Mindflow 的接口层在相同循环内.
        但实际上总线的调用可能在不同线程. 所以总线函数底层必须是线程安全的 (比如用 janus.Queue).

        **调用 fire_impulse 时应该持有其缓存, 直到 mindflow 返回判别结果为止 **
        """
        pass

    @abstractmethod
    def suppress(self, suppress_by: Impulse) -> None:
        """
        如果产生的 impulse 不能被接纳, Nucleus 应该收到一个 suppress 信号.
        接受这个信号后, 一段时间不要 fire Impulse.
        可以在内部实现加权/降权 逻辑.
        **所有的 Nucleus 都需要通过独立的 supress 单测**

        :param suppress_by: 被别的信号压制, 得到别的信号. 未来可以通过决策单元判断是否要加权.
        """
        pass

    @abstractmethod
    def attended(self, impulse: Impulse) -> None:
        """
        通知 Nucleus 它的 Impulse 抢占 Attention 成功.
        执行 attended 后, 应该无法 peek 到相同的 impulse.
        """
        pass

    def with_moments(self, moments: Moments) -> None:
        """获得 moments 容器, 可以用于有状态逻辑. """
        pass

    def create_attention(
            self,
            observer: Observer,
            impulse: Impulse,
    ) -> 'Attention | None':
        """如果 nucleus 自身有能力实现 attention, 会取代 mindflow 生成的 attention """
        pass

    def ignored(self, impulse: Impulse) -> None:
        """
        通知 Nucleus 这个 Impulse 被忽视了. 通常是因为过期等原因.
        """
        pass

    @abstractmethod
    def peek(self, no_stale: bool = True) -> Impulse | None:
        """
        查看一下最新的 Impulse, 方便做 ranking.
        attended 后应立刻清除;suppressed 后 impulse 仍保留、可被 peek
        (只是在一段时间内不再主动 fire), 供 mindflow 下一轮重新 rank 调度.
        """
        pass

    def as_channel(self) -> Channel | None:
        """
        如果 Nucleus 有能力返回反身性的控制 channel,
        则 mindflow 应该将它作为动态或者静态节点接入.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """是否启动成功, 正在运行. """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        启动 Nucleus 自身的生命周期, 包含异步逻辑, 或者启动子进程.
        可以在生命周期中创建 signal 消费的 asyncio task, 方便做正确的生命周期治理.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        退出生命周期.
        """
        pass


class NucleusMeta(ABC):
    """
    Nucleus 的元配置. 是可选的实现.

    如果使用它来生成 Nucleus 实例, 则可提前得到自解释协议.
    可以实例化后, 在运行时构建出 Nucleus 实例.
    用这种方法可以在运行环境未启动之前, 就反应出协议.
    """

    @abstractmethod
    def name(self) -> str:
        """
        用于区分不同的 Nucleus 单元.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        所有的 Nucleus 都应该是自解释的, 而且这个自解释要足够高效, 能一句话自我描述.
        """
        pass

    @abstractmethod
    def signals(self) -> Iterable[type[SignalMeta]]:
        """
        声明监听的信号类型.
        """
        pass

    @abstractmethod
    def factory(self, container: IoCContainer) -> Nucleus:
        pass


Logos = AsyncIterator[str]
"""
智能体输出用来驱动躯体/工具/交互/思考 等一切能力的讯息. 对应中文的 "道". 目前在项目里主要是 CTML. 它包含四重含义:
1. 它本身是语言, 在 MOSS 架构里包含了运行时控制的魔力 (CTML). 
2. 它是逻辑的编织, 要符合现实世界的规律 (时间第一公民, 时序拓扑, 结构化并行)
3. 它驱动了躯体/工具/思维 的运行轨迹
4. 它包含了智能体与现实世界交互的底层原则, 一个智能体通过它输出的 logos 来展示它自身的 logos. 

经过和 Gemini/Deepseek 的多轮讨论, 没有更好的词能够精准涵盖它所包含的 哲学/技术拓扑, 又屏蔽掉底层实现 (比如 CTML).

在 MOSS 架构中运行的智能体, 更像是 "魔法师". 它不是用精确到舵机电平的神经脉冲控制外部世界, 而是用符号流.
类似用魔法吟唱的方式驱动火球, 石头人 等. 
"""

ChallengeVerdict = Literal['preempted', 'suppressed', 'absorbed', 'initial', 'buffered', 'yielded']
"""Impulse challenge 的仲裁结果。
- preempted: 抢占成功，创建新 Attention
- suppressed: 被压制，原 nucleus 收到 suppress()
- absorbed: 同 ID 更新 complete，不抢占
- initial: 当前无 attention（首个 impulse）
- buffered: silent 抢占成功侧 / notify 抢占失败侧 → messages 进 mindflow buffer
- yielded: strength=0 绝不竞争 — 不分 defender/quiet, 不打任何 mode 分支,
  不建 attention, 由 nucleus 自然清理缓存 (Zen 静默心智模型预留)
"""


class AttentionStatement(ABC):
    """mindflow 生产的有状态运行单元."""

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @abstractmethod
    async def wait_abort(self) -> None:
        """等待到退出, 适合旁路观测."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """在旁路设置 statement 退出, 仅退出自身, 不涉及父逻辑. """
        ...

    @abstractmethod
    def abort(self, reason: str | Exception | None) -> None:
        """不仅退出自身, 也退出父级的 Statement"""
        ...

    @abstractmethod
    def abort_reason(self) -> str:
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """是否已经运行, 而且没有退出. """
        ...

    @abstractmethod
    def is_aborted(self) -> bool:
        """已经退出, 可能由自身或父 statement 触发."""
        ...


class Articulator(ABC):
    """在思考过程中发送指令给 Action, 每次创建一个 Articulator 就会生成一个新的 Action. """

    @abstractmethod
    def send_nowait(self, logos_delta: str) -> None:
        """
        发送单个 logos delta, 无背压.
        """
        ...

    @abstractmethod
    async def send(self, logos_delta: str) -> None:
        """发送一个 logos delta, 有背压"""
        ...

    @abstractmethod
    async def wait_compiled(self) -> None:
        """等待到 logos 编译完成."""
        ...

    @abstractmethod
    async def wait_action_done(self) -> None:
        """等待到 action 执行完毕. """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """启动 articulator. """
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """确保 commit 动作触发, 通知 action 输入完毕. 或者因为异常中断执行. """
        ...


class ActionGate(ABC):
    """action 闸门, 在 think 的生命周期下执行.

    articulator 在 commit (logos 写完) 时调用 ``approve`` 裁决完整 logos:
    返回 ``(approved, message)``. ``approved=False`` 会由 articulator abort 掉当前
    action (进而 abort attention), 不重新 loop。
    """

    @abstractmethod
    async def approve(self, logos: str) -> tuple[bool, str]:
        """裁决一段完整 logos. 返回 (approved, message). """
        ...


class Thinking(AttentionStatement, ABC):
    """
    推理决策单元, 将推理的结果发送给执行单元.
    需要实现线程安全.
    """

    @property
    @abstractmethod
    def attention(self) -> 'Attention':
        """
        当前思考帧持有的 attention.
        """
        ...

    @property
    @abstractmethod
    def moment(self) -> Moment:
        """
        当前思考帧拿到的上下文.
        """
        ...

    @abstractmethod
    def gate(self) -> ActionGate:
        """开启闸口, 发送的 logos 都必须由 ActionGate 验证通过. """
        ...

    @property
    @abstractmethod
    def observer(self) -> Observer:
        """持有 Moments 的唯一观测者. """
        ...

    @abstractmethod
    def observe(self) -> Moment:
        """立刻观测, 并产生一个新的 moment 帧, 并且被持有."""
        pass

    @abstractmethod
    def effort(self) -> ThinkingEffort:
        """
        思维强度, 为 None 的话不应该执行 articulate.
        """
        ...

    @abstractmethod
    def articulator(
            self,
            replan: bool = False,
            wait_action_done: bool = False,
    ) -> Articulator:
        """
        创建一个可以发布 logos 的 articulator.
        """
        ...

    @abstractmethod
    async def wait_until_done(self, *futures: asyncio.Future) -> None:
        """同步等待到所有 future 结束, 或者自身退出."""
        ...

    async def articulate(self, logos: Logos) -> None:
        """
        发送 Logos 流
        """
        async with self.articulator() as articulator:
            async for delta in logos:
                await articulator.send(delta)


class StatementExitedException(Exception):
    """不同级别的退出信号, 用来在调用栈中直接退出, 会被 statement 的 exit 捕获. """
    pass


class ActionExitedException(StatementExitedException):
    """标记 Action 已经停止了, 方便退出调用栈, 进入到 Action exit. """
    pass


class ThinkExitedException(StatementExitedException):
    """标记 Think 已经停止了. 方便退出调用栈, 进入到 think exit. """
    pass


class AttentionExitedException(StatementExitedException):
    """标记 Attention 必须 Abort. """
    pass


class Action(AttentionStatement, ABC):
    """
    控制 Logos 的执行循环.
    与 Articulator 成对生成.
    Articulator 实际上可以比 Action 更早结束.
    思维可以走在行动的前面, 观察行动的结果.
    """

    @property
    @abstractmethod
    def attention(self) -> 'Attention':
        """action 也持有 attention """
        ...

    @abstractmethod
    async def wait_ready(self) -> None:
        """
        等待 abort 或第一个有语义的帧. 适合在 received logos 前调用, 避免副作用.
        :raise ActionStopError: 抛出内部的异常, 用来快速走到 __aexit__ 退出.
        """
        ...

    @property
    @abstractmethod
    def replaned(self) -> bool:
        """要求先停止所有的行动. """
        ...

    @abstractmethod
    def set_compiled(self):
        """
        标记 logos 已经全部读取完, 并且已经完成了编译. 这样不等待 Action 运行结束, Think 可以继续执行.
        """
        ...

    @abstractmethod
    def logos(self) -> Logos:
        """
        返回本轮生成的执行文本.
        :returns: AsyncIterable[str]
        """
        pass

    @abstractmethod
    async def wait_until_done(self, *futures: asyncio.Future) -> None:
        """同步等待到所有 future 结束, 或者自身退出."""
        ...

    @property
    @abstractmethod
    def moments(self) -> Moments:
        """返回持有的 moments, 可以用于 add result. """
        ...

    @abstractmethod
    def abort_thinking(self) -> None:
        """
        Action 是 Thinking 派生出来的, 如果出现了行动不可执行异常, 应该要主动停止思考. 可以不释放注意力.
        """
        ...

    def add_result(self, *messages: Message | str, observe: bool = False) -> None:
        """
        提交 outcome, 标记是否要引发下一轮观察.
        如果在一个 Action 的生命周期中 Observe 被标记了, 或者发生了特殊的异常,
        Attention 会循环下一组调用.
        如果没有需要观察的 outcome, Attention 会自然结束.
        """
        self.moments.add_result(list(messages), need_observe=observe)


class Attention(AttentionStatement, ABC):
    """
    一种三循环全双工运行时的资源和状态调度单元.
    它通常是 Impulse 创建出来的实例, 一直到 思考/执行 都结束后退出.
    它负责在思考循环和行动循环的执行过程中, 仲裁其它 impulse 的挑战.

    Attention 会在三种情况下结束:
    1) 思考/行动 运行完, 没有任何可观测信息时, 自动停下来.
    2) 发生致命异常, 导致 attention abort.
    3) 新的 impulse 挑战成功, attention 终结, 持有它的 Think / Action 也会退出.
    """

    @abstractmethod
    def draw_from(self) -> Impulse:
        """
        创建 Attention 的 impulse.
        """
        pass

    @abstractmethod
    def absorb_impulse(self, impulse: Impulse) -> Impulse | None:
        """
        吸收一个 impulse, 更新当前优先级和强度.
        """
        ...

    @abstractmethod
    async def wait_ready(self) -> Impulse:
        """
        等待到第一个完整的 impulse.
        适合首包抢占注意力的场景.
        """
        ...

    @property
    def id(self) -> str:
        return self.draw_from().id

    @abstractmethod
    def is_protected(self) -> bool:
        """声明仍然在保护期内. """
        ...

    @abstractmethod
    def priority(self) -> Priority:
        """当前运行的优先级"""
        ...

    @abstractmethod
    def set_priority(self, priority: Priority | None) -> None:
        """设置当前注意力优先级, 用于防打断. """
        ...

    @abstractmethod
    def escalate(self) -> None:
        """保持强度"""
        ...

    @abstractmethod
    async def challenge(self, challenger: Impulse) -> Literal['win', 'lose', 'absorb']:
        """
        仲裁新的 impulse. 决定自身是否被中断. 调度发起者是 mindflow.
        最基础的仲裁逻辑:
        0. 启动可以设置保护期, 保护期内不被条件.
        1. 如果 id 和当前 Impulse 相同, complete 取代 incomplete 并解除 impulse 阻塞.
        2. 挑战的 impulse priory 低于当前 impulse 优先级, 返回 False, 目标 impulse 发起方接受 suppress 回调.
        3. 优先级相同, 应该基于同源提权, 异元降权的原理做强度比较.
        4. 如果挑战者优先级更高, 则挑战一定成功. 当前 Attention 应该 abort.
        5. 如果 priority 为 Fatal, 应该永远被打断.

        这是最简单的规则. Attention 更好的做法是有一个速度极快的仲裁者. 它要具备响应大量讯号挑战的极简算法.

        - Preempted(True):
            如果挑战成功, Mindflow 应该实例化新的 Attention 之后, abort 当前的 Attention.
        - Supress (False):
            挑战失败, Mindflow 应该 supress impulse 的源头.
        - BufferImpulse (None):
            这个 Impulse 被 Attention 吸收了, 当 Attention 没被中断时, 会将 Impulse 提供到下一轮 Observation.
            Buffer Impulse 提供连续观察思考的语义. 只有同源的 Impulse, 且级别为 Info 时会更新.

        attention 管理一个源响应的生命周期.
        在这个生命周期中, 如果想要抢占, 则应该走 Impulse 逻辑打断.
        想要观察, 则走 outcome.
        想要提供低优先级的补充信息, 走 INFO.

        OnChallenge 在系统内最核心要解决的问题, 是消除大多数情况下的仲裁风暴和无限抖动.
        这在早期工程复杂度简单的时候, 直接通过约定的设计范式解决.
        更复杂的情况下会引入高阶反身性仲裁, 那属于甜蜜的烦恼.
        """
        pass


_NucleusName = str


class MindflowHook:

    def name(self) -> str:
        return ''

    def description(self) -> str:
        return ''

    def on_impulse_challenged(
            self,
            challenger: Impulse,  # challenger — 发起挑战的 Impulse
            defender: Impulse | None,  # defender   — 当前占据注意力的 Impulse，None 表示无当前 attention
            verdict: ChallengeVerdict,  # verdict    — 仲裁结果
    ) -> None:
        """注册 challenge 旁路观察回调。
        每次 impulse challenge attention 后触发:
          observer(challenger, defender, verdict)
        传 None 清除回调。同时只保留一个。仅观察，无副作用。
        """
        pass

    def on_error(self, error: Exception) -> None:
        pass


class Mindflow(ABC):
    """
    三循环智能体的思维调度中枢.

    它解决的核心问题是, 如何 管理/描述/隔离 一个全双工三循环系统的运行逻辑.

    三循环: 1. 感知输入;  2. 思考循环. 3. 躯体运行.
    双工: 1. 躯体输出; 2. 感知输入. 两者并行.
    有复杂的中断逻辑: 0. 强制命令, 比如熔断, 急停. 1. 思考异常; 2. 执行异常; 3. 执行结束; 4. 输入更强的信号产生抢占调度.

    同时有很多个状态和讯号通讯, 而在一个时间片里只有一组行为拥有可运行资源 (Attention).

    Mindflow 的作用就是统筹所有的实现模块:
    1. nucleus: 感知单元, 接受原始信号量, 通过加工后返回有优先级效果的 Impulse. 解决并行感知后聚合/行为仲裁的问题.
    2. attention: 单一执行状态管理, 能同时接受多方的讯号, 维持一个可被抢占的运行时状态. 交换数据, 管理所有生命周期.
    3. think: 思维的单元.
    4. action: 行为的单元.
    5. moments: 可观测讯息的轨迹, 可以在多个 nucleus 中共享.
    """

    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def with_hook(self, hook: MindflowHook) -> Self:
        """注册 hook"""
        pass

    @abstractmethod
    def remove_hook(self, hook: str | MindflowHook) -> None:
        """移除注册的 hook"""
        pass

    @abstractmethod
    def nuclei(self) -> dict[_NucleusName, Nucleus]:
        """
        持有的并行感知, 思考, 裁决单元.
        这里的 nucleus 并不一定是个执行单元, 也可以仅仅是一个通讯单元或 Adapter.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """是否在运行中. """
        pass

    @abstractmethod
    async def wait_started(self) -> None:
        """等待启动完成."""
        pass

    @abstractmethod
    def wait_started_sync(self, timeout: float | None = None) -> bool:
        """同步等待到 mindflow 开始运行. """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        has no attention and impulse
        """
        pass

    @abstractmethod
    def when_idle(
            self, callback: Callable[[Moments], None] | Callable[[Moments], Awaitable[None]],
    ) -> Callable[[], None]:
        """注册闲时回调逻辑. return disposer"""
        ...

    @property
    @abstractmethod
    def moments(self) -> Moments:
        """持有 Moments 轨迹. """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        排空讯号, 应该强制清空所有状态.
        用于做极限故障下的还原, 作为最基础的恢复手段.
        """
        pass

    def as_channel(self) -> Channel | None:
        """
        如果一个 mindflow 能够提供一个 Channel, 如果提供的话应该是 Mindflow 持有的属性.
        这个 channel 应该要持有所有的 nuclei channel.
        合并到 Shell 中提供对思维本身的反身性控制.
        """
        return None

    def set_signal_priority_bar(self, priority: Priority) -> None:
        """
        设置 signal 的优先级门槛.
        定于门槛时, signal 会被直接丢弃.
        系统级别的注意力门槛.
        """
        # 函数为 控制台和反身性 channel 准备. 默认不实现.
        pass

    def set_impulse_priority_bar(self, priority: Priority) -> None:
        """
        设置 impulse 的门槛, 低于门槛的 impulse 没有挑战资格.
        """
        # 函数为 控制台和反身性 channel 准备. 默认不实现.
        pass

    @abstractmethod
    async def add_nucleus(self, nucleus: Nucleus, override: bool = False) -> Self:
        """
        动态注册新的感知单元. 在运行时添加, 添加时启动.
        :raise DuplicatedError
        """
        pass

    @abstractmethod
    def with_nucleus(self, nucleus: Nucleus, override: bool = False) -> Self:
        """
        静态注册新的感知单元. 必须在 mindflow 启动前注册.
        :raise DuplicatedError
        """
        pass

    @abstractmethod
    def add_impulse(self, impulse: Impulse) -> None:
        """
        接受一个 impulse, 并进入和当前 attention 的 challenge 仲裁.
        注意, 这里的 on_signal / on_impulse 作为总线提供给 Nucleus 时, 要防止信号成环无限传播.
        似乎没有系统机制可以百分之百预防.
        """
        pass

    @abstractmethod
    def add_signal(self, signal: Signal) -> None:
        """
        接受 signal 回调. 由于 Signal 的回调很可能和 Mindflow 不是在同一个线程或循环,
        所以内测需要卸载到当前循环, 并且考虑做好讯号闸门.
        Signal 的限频最好不在 Mindflow 侧做, 而应该通过发送者/环境中间件解决限频问题.
        """
        pass

    @abstractmethod
    def peek_impulses(self) -> Iterable[tuple[Nucleus, Impulse]]:
        """当前的所有的 impulses. """
        ...

    @abstractmethod
    def attention(self) -> Attention | None:
        """
        返回当前的 Attention.
        任何时候只会有一个活跃的 Attention.
        """
        pass

    @abstractmethod
    def set_impulse(self, impulse: Impulse) -> None:
        """
        直接添加一个 Impulse 到池中, 立刻生成新的 Attention.
        """
        pass

    @abstractmethod
    def pause(self, toggle: bool) -> None:
        """
        急停, 仍然接受 signal/impulse, 但不会分发, 而是直接丢弃. 只有 set_ 系统指令仍有意义.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        立刻关闭 Mindflow.
        """
        pass

    @abstractmethod
    async def wait_close(self) -> None:
        pass

    @abstractmethod
    def thinking_loop(self) -> AsyncIterator[Thinking]:
        """
        循环生成 Think 对象, 将它们发送到 thinking 循环中.
        """
        pass

    @abstractmethod
    def action_loop(self) -> AsyncIterator[Action]:
        """循环生成 action 对象, 来自 think 调用 articulator 生产. """
        ...

    async def run(
            self,
            *,
            put_think: Callable[[Thinking], None] | Callable[[Thinking], Awaitable[None]],
            put_action: Callable[[Action], None] | Callable[[Action], Awaitable[None]],
    ) -> None:
        """
        mindflow 运行逻辑, 调度生产 thinking 和 action 两个循环. 通过队列桥接到别的 task 中运行.
        实际上也可以就地创建两个 loop.
        """

        async def _run_thinking_loop():
            async for think in self.thinking_loop():
                v = put_think(think)
                if asyncio.iscoroutine(v) or asyncio.isfuture(v):
                    await v

        async def _run_action_loop():
            async for action in self.action_loop():
                v = put_action(action)
                if asyncio.iscoroutine(v) or asyncio.isfuture(v):
                    await v

        await asyncio.gather(_run_thinking_loop(), _run_action_loop())

    @abstractmethod
    async def __aenter__(self):
        """启动"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出"""
        pass


class ImpulsePrimitive:
    """Impulse 控制原语组合表.

    每个原语把一组协议字段拼成一个具名行为, 让上层 nucleus / 测试 / 应用
    用"读名字就懂行为"的方式构造 Impulse, 代替手抠 priority/mode/effort/logos 四件套.

    原语本身是 code-as-prompt: 内部一两行字段赋值即可, 关键价值在名字传递的意图.
    单测覆盖每个原语的组合语义.
    """

    @staticmethod
    def command_only(impulse: Impulse, command_logos: str) -> Impulse:
        """直接执行指令, 不思考.

        组合: ``logos = command_logos`` + ``thinking_effort = 'none'``.
        priority 由调用方控制.
        """
        impulse.logos = command_logos
        impulse.thinking_effort = 'none'
        return impulse

    @staticmethod
    def fatal_command(impulse: Impulse, command_logos: str) -> Impulse:
        """强制指令 — 必抢占 + 不思考.

        组合: ``command_only`` + ``priority = FATAL``.
        用例: 急停 / 强制状态切换 / 任何"无论 ghost 在做什么都必须执行"的反射弧.
        """
        impulse = ImpulsePrimitive.command_only(impulse, command_logos)
        impulse.priority = Priority.FATAL.value
        return impulse

    @staticmethod
    def broadcast(impulse: Impulse) -> Impulse:
        """高优广播 — 必送达但不接管 ghost 运行时.

        组合: ``priority = FATAL`` + ``mode = silent`` + ``thinking_effort = 'none'``.
        FATAL 保证抢占成功, silent 在抢占成功侧偏离 default — 不创建新 attention,
        只把 messages 灌进 mindflow buffer, 由下一个 attention 自然 drain 到 percepts.

        用例: 系统通告 / 紧急广播 — ghost 不需要立刻切换上下文, 只要下一帧看到.
        与 ``fatal_command`` 区别: broadcast 不带 logos, 只送消息;
        fatal_command 带 logos 走 attention 立即执行.

        对偶: ``interrupt`` — 同样 FATAL + effort=none, 但用 notify 模式接管
        attention 并通过 ``interrupt`` 字段触发 shell.stop_interpretation, 表达"中断"
        而非"补充".
        """
        impulse.thinking_effort = 'none'
        impulse.priority = Priority.FATAL.value
        impulse.mode = ChallengeMode.silent.value
        return impulse

    @staticmethod
    def interrupt(impulse: Impulse) -> Impulse:
        """中断动作 — 必送达且必接管, 但不思考不执行新 logos.

        组合: ``priority = FATAL`` + ``mode = notify`` + ``thinking_effort = 'none'`` +
        ``interrupt = True``.

        FATAL 保证抢占成功, notify 走 default 成功路径创建新 attention,
        effort='none' 让 ghost.articulate 提前返回, ``interrupt=True`` 让
        ``ghost_runtime._run_articulator`` 在新 attention 起步时调
        ``shell.stop_interpretation()`` 清干净旧 logos.

        本原语就是"打断" 的本质形态: 新 attention 起来, 旧 logos 停, ghost
        不发表任何新意见 — 等下一个真正的 impulse 进来.

        用例: 急停 / 模型自我打断 / 状态机切换 / 用户喊"停".
        可携带 messages 解释中断原因, 由下一帧 percepts drain.

        对偶: ``broadcast`` — 同 FATAL + effort=none 但用 silent 不接管;
        interrupt 接管但立即放手.
        """
        impulse.thinking_effort = 'none'
        impulse.priority = Priority.FATAL.value
        impulse.mode = ChallengeMode.notify.value
        impulse.interrupt = True
        return impulse

    @staticmethod
    def notify(impulse: Impulse) -> Impulse:
        """不丢消息 — 保留 impulse 原 priority, 抢占失败也 buffer.

        组合: ``mode = notify``.
        priority 由调用方控制 (NOTICE 是典型用户消息).
        notify 在抢占失败侧偏离 default — 抢占成功正常创建新 attention,
        失败时 messages 进 buffer 而非 suppress.

        用例: 用户消息绝不能丢 — ghost 思考时说话, 不打断就留痕.
        """
        impulse.mode = ChallengeMode.notify.value
        return impulse

    @staticmethod
    def background_notice(impulse: Impulse) -> Impulse:
        """低优补充 — 永不抢占, 失败时留痕.

        组合: ``priority = BACKGROUND`` + ``mode = notify``.
        BACKGROUND 保证抢占失败, notify 让失败侧走 buffer 而非 suppress.

        用例: 日志 / 监控 / 环境信号 — 不重要到要打断 ghost, 但下一轮思考应该能看到.
        注意边界: quiet 系统 (无 attention) 下 notify 走 default 路径会创建新 attention,
        此时 BACKGROUND attention 会成为新焦点 (但低优, 任何信号都能抢占它).
        若需要 quiet 时也不创建 attention, 应使用 ``broadcast`` (但 priority 必须是 FATAL).
        """
        impulse.priority = Priority.BACKGROUND.value
        impulse.mode = ChallengeMode.notify.value
        return impulse
