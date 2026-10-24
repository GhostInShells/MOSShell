"""Etiquette — the interaction styles a model programs at runtime.

One etiquette configures one listening session. It has three independent layers,
and an interaction style is a point in their product space — not a code branch:

- **onset** (起话) — what to signal the instant speech begins.
- **stop** (判停) — how the turn end is decided.
- **deliver** (交付) — the single delivery knob: emit a signal or keep in buffer.

The examples below are out-of-box points in this space. Copy one, change a field,
and you have a new interaction style — no code. Register it on ``EtiquetteConfig``
to make it selectable by name.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.blueprint.mindflow import Priority

__all__ = [
    "OnsetSpec",
    "DeliverSpec",
    "DeliverMode",
    "ClassifierSpec",
    "DetectSpec",
    "StopSpec",
    "EtiquetteSpec",
    "EtiquetteConfig",
    "flatten_interaction",
    "deviated_keys",
    "apply_interaction_patch",
    "needs_session_restart",
    "once",
    "always",
    "aside",
    "notify",
    "scribe",
    "observer",
    "keyword_end",
    "scored",
    "murmur",
    "cue",
    "duplex",
    "knock",
]


#: Deliver mode — the loss-side behavior when the deliver signal cannot preempt.
#: Values align 1:1 with :class:`ghoshell_moss.core.blueprint.mindflow.ChallengeMode`:
#: - ``""`` — default: buffer suppressed on preempt failure
#: - ``"aside"`` — inject messages only, never take over attention
#: - ``"notify"`` — buffer instead of suppress on preempt failure (answer when idle)
#: - ``"next"`` — buffer + force next-frame observation (take the next turn)
DeliverMode = Literal["", "aside", "notify", "next"]


class OnsetSpec(BaseModel):
    """起话协议 — what happens the instant speech begins (barge-in)."""

    emit: bool = Field(
        default=True,
        description="emit a signal when speech begins",
    )
    interrupt: bool = Field(
        default=True,
        description="that signal stops the current behavior",
    )
    priority: Priority = Field(
        default=Priority.WARNING,
        description="preempt tier of the onset signal",
    )


class DeliverSpec(BaseModel):
    """交付协议 — the single delivery knob.

    ``emit`` 是唯一交付旋钮: True = 判停后把这一轮发成 listener signal 推给模型;
    False = 不发 signal, 强制走 buffer 留存, 模型经 ``get_transcript`` drain. 没有
    「既不发送也不留存」的组合 —— ``emit=False`` 必然留存 (留存不再是独立正交参数).

    ``interrupt`` / ``priority`` / ``mode`` 只在 ``emit=True`` 时有意义 (信号怎么发);
    ``history`` 只在 ``emit=False`` 时有意义 (buffer 环形容量).
    """

    emit: bool = Field(
        default=True,
        description="delivery knob: True = emit a listener signal (push); "
                    "False = keep in buffer for model pull (forces retention)",
    )
    history: int = Field(
        default=8,
        description="buffer ring capacity — effective only when emit=False",
    )
    interrupt: bool = Field(
        default=False,
        description="the signal stops the current behavior (emit=True only)",
    )
    priority: Priority = Field(
        default=Priority.INFO,
        description="preempt tier of the deliver signal (emit=True only)",
    )
    mode: DeliverMode = Field(
        default="notify",
        description="loss-side behavior when the deliver signal cannot preempt "
                    "(emit=True only; aligns with mindflow ChallengeMode): "
                    "'notify' (buffer — answer when idle) / "
                    "'aside' (inject without taking over) / "
                    "'next' (buffer + take next turn) / "
                    "'' (default — suppress on preempt failure)",
    )
    low_confidence: float | None = Field(
        default=None,
        description="词级置信度阈值: 置信度 < 该值的字会在 deliver signal 的 low_conf "
                    "属性里列出来 (供模型自查转写质量); None = 不列 (默认)。emit=True only",
    )
    barge_in_while_speaking: bool = Field(
        default=False,
        description="出声即让路: ghost 正在出声 (speech player 在播) 时, 交付信号带一个一次性提权 "
                    "(见 barge_in_boost), 在仲裁里赢下当前注意力 (emit=True only). 思考中不受它影响 —— "
                    "那时仍按 interrupt=False 走 mode 的排队/留存. 一个开关谈清「什么可以被抢」: "
                    "出声可以被抢, 思考不该被抢.",
    )
    per_clause: bool = Field(
        default=False,
        description="分句交付: 云端每判出一句就各发一包 (只带增量, 永不重发前文), 段的收尾另发一包 "
                    "(vad=segment, 只带没发过的那一截). 关 = 老行为: 只在段结束发一包, 带整段全文. "
                    "开启时 stop.silence 必须 > 0 —— 否则每句都提交一次, 段被切成碎片.",
    )
    barge_in_boost: Priority = Field(
        default=Priority.WARNING,
        description="barge_in_while_speaking 打开时, 出声期间交付信号提权到的那一档. 默认 WARNING: "
                    "高于 INFO/NOTICE 档的动作注意力, 又低于 ERROR 及以上的故障档. "
                    "提权只赢一局 —— 抢到注意力后 nucleus 会把它降回运行档.",
    )


class ClassifierSpec(BaseModel):
    """A programmable streaming classifier mounted on the turn-end slot.

    The slot decides only that a classifier runs here; ``instruction`` decides
    what it judges and how it scores. ``context`` carries volatile hints the
    model wants the classifier to consider (e.g. an explicit end signal the
    user just announced). ``threshold`` is the score at which the turn commits
    early.

    Cache shape: ``instruction`` is stable and rides the caller-level cache;
    ``context`` is the first user message, so the accumulated clause prefix
    hits the prompt cache within a segment.
    """

    instruction: str = Field(
        default="",
        description="the classifier's own instruction — what to judge and how to score",
    )
    context: str = Field(
        default="",
        description="volatile hints for the classifier (e.g. explicit end signals the "
                    "user announced, task background); referenced from instruction via "
                    "the ``<context>`` block. Empty = no context block sent.",
    )
    threshold: int = Field(
        default=7,
        description="score >= threshold commits the turn early",
    )
    delay: float = Field(
        default=0.3,
        description="debounce before scoring; a clause superseded within it costs no call",
    )


class DetectSpec(BaseModel):
    """Audio detection — a composable slot that signals on sound, not a commit.

    Symmetric to ``classifier``: ``classifier`` is a programmable LLM stop-judge
    (``instruction``), ``detect`` is a programmable audio detector (``handler``).
    The default handler is a simple energy threshold on the capture-precomputed
    ``meta.rms_db``; ``handler`` is a future sandbox extension point.
    """

    enabled: bool = Field(
        default=True,
        description="enable sound detection",
    )
    threshold_db: float = Field(
        default=-50.0,
        description="energy threshold (rms_db); frames below it are treated as silent",
    )
    handler: str | None = Field(
        default=None,
        description="optional sandbox audio-detection function; None = simple energy threshold",
    )
    # No priority knob: a knock is emitted at BACKGROUND — losable by promise,
    # and only the tier that can never preempt keeps that promise (D22).
    cooldown: float = Field(
        default=10.0,
        description="seconds to suppress repeat knocks after a detection",
    )


class StopSpec(BaseModel):
    """判停 — how the turn end is decided. Four composable commit paths:

    - ``silence``: commit N seconds after the last clause (0 = commit on the first clause).
    - ``keywords``: an explicit endpoint — a hit commits immediately.
    - ``classifier``: a score-based early commit; None = not mounted.
    - ``detect``: an audio detector that signals on sound (knock) instead of committing; None = not mounted.
    """

    silence: float = Field(
        default=1.5,
        description="commit after N seconds of quiet; 0 = commit on the first clause",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="explicit endpoint keywords — a hit commits immediately",
    )
    classifier: ClassifierSpec | None = Field(
        default=None,
        description="score-based early commit; None = not mounted",
    )
    detect: DetectSpec | None = Field(
        default=None,
        description="audio detector that signals on sound (knock) instead of committing; None = not mounted",
    )


class EtiquetteSpec(BaseModel):
    """一种礼仪 — name + description + the three layers."""

    name: str = Field(
        description="etiquette name; referenced by EtiquetteConfig.default",
    )
    description: str = Field(
        default="",
        description="one-line self-description, for model perception",
    )
    onset: OnsetSpec = Field(default_factory=OnsetSpec)
    stop: StopSpec = Field(default_factory=StopSpec)
    deliver: DeliverSpec = Field(default_factory=DeliverSpec)
    idle_timeout: float = Field(
        default=300.0,
        description="seconds without any recognition event before the session ends "
                    "itself (emit a notify signal to the model, then stop); "
                    "0 = never end on idle (stay on indefinitely)",
    )


# ── 开箱礼仪 — 每个是一个坐标; 复制一个改字段就是新礼仪, 注册到 Config 即可选用 ──


# 一次: 说一句立刻交付 (silence=0 → 首个 clause 即端点); 不起话, 不听中途.
once = EtiquetteSpec(
    name="once",
    description="one utterance — commit on the first clause",
    onset=OnsetSpec(emit=False),
    stop=StopSpec(silence=0.0),
)


# 默认: 打断式自由对话. 起话即打断 (barge-in), 静默 1.5s 交付; 抢不到注意力则 buffer, 闲了再答.
always = EtiquetteSpec(
    name="always",
    description="interruptible turn-taking — commit after 1.5s of quiet",
    onset=OnsetSpec(emit=True, interrupt=True, priority=Priority.WARNING),
    stop=StopSpec(silence=1.5),
    deliver=DeliverSpec(mode="notify"),
)


# 旁听: 起话发信号但绝不打断; 你干你的, 我不接管 (aside).
aside = EtiquetteSpec(
    name="aside",
    description="listen without interrupting — annotate, never take over",
    onset=OnsetSpec(emit=True, interrupt=False, priority=Priority.NOTICE),
    stop=StopSpec(silence=1.5),
    deliver=DeliverSpec(interrupt=False, mode="aside"),
)


# 闲时回应: 起话不打断, 你说的都收着; 我忙就继续忙, 闲了才回应 (低优 + notify).
notify = EtiquetteSpec(
    name="notify",
    description="keep working — answer only when idle",
    onset=OnsetSpec(emit=True, interrupt=False, priority=Priority.INFO),
    stop=StopSpec(silence=2.0),
    deliver=DeliverSpec(interrupt=False, priority=Priority.INFO, mode="notify"),
)


# 书记员/翻译官: 大段论述, 我一直听一直做事, 但不打断手头; 下轮一定轮到我 (next).
scribe = EtiquetteSpec(
    name="scribe",
    description="long dictation — keep working, take the next turn",
    onset=OnsetSpec(emit=True, interrupt=False, priority=Priority.INFO),
    stop=StopSpec(silence=2.5),
    deliver=DeliverSpec(interrupt=False, mode="next"),
)


# 只录不答: 不发任何信号, 只把听到的留存, 事后可拉读 (转写 / 会议记录).
observer = EtiquetteSpec(
    name="observer",
    description="transcribe only — never signal, keep recent heard text",
    onset=OnsetSpec(emit=False),
    stop=StopSpec(silence=2.0),
    deliver=DeliverSpec(emit=False, history=16),
)


# 对讲机: 显式端点, 说完喊 over 才交付, 不靠静默.
keyword_end = EtiquetteSpec(
    name="keyword_end",
    description="walkie-talkie — commit only on an explicit end keyword",
    stop=StopSpec(silence=30.0, keywords=["over", "完毕"]),
)


# 低语旁听: 持续听, 每句一交付, 绝不打断模型 — 忙时被动进 percepts (notify), 闲了才唤醒.
# 逐句接话: 同为每句一交付, 但用 next — 忙时强制当前 attention 多看一帧 (插队), 不打断身体.
# 已知陷阱: notify/next 只保证"忙时不抢占"; 一旦某条 clause 最终抢占成功 (attention 衰减到阈值以下),
# preempt 仍带 abort reason "interrupted" 触发 shell.clear() 停身体. "preempt 中性化"是独立 mindflow 待办
# (preempt / error / end / cancel 四相区分), 落地前这两条在"模型正忙"时不清身体, 但"衰减后被抢占"仍会停身体.
murmur = EtiquetteSpec(
    name="murmur",
    description="keep working — every clause reaches you without interrupting; respond or ignore",
    onset=OnsetSpec(emit=False, interrupt=False),
    stop=StopSpec(silence=0.0),
    deliver=DeliverSpec(interrupt=False, mode="notify"),
)


# 逐句接话: 每句自成一段, 每句强制下一轮轮到你 (next), 绝不打断"思考".
# 这一档是唯一打开"出声即让路"的 —— 我一开口, 他说的话就立刻掐断我的语音, 但不动我的动作.
# 它的语义是"被点名才说话": 我说的话是应答, 他一开口应答就该让路.
# 名字**不**写进机制 (这是系统默认, 不是给我一个人做的): 规则写在 description 里,
# 由 listener 每轮把它带进 instruction; 具体叫什么名字, 以后由 ghost 自己往说明里加.
cue = EtiquetteSpec(
    name="cue",
    description=(
        "被点名才开口: 没人叫你, 就别说. 你一出声, 我说到一半的话会被立刻掐断 "
        "(所以开口前先掂量这句话值不值得被打断). "
        "Speak only when addressed — your voice is the one that yields here."
    ),
    onset=OnsetSpec(emit=False, interrupt=False),
    stop=StopSpec(silence=0.0),
    deliver=DeliverSpec(
        interrupt=False, mode="next", per_clause=True, low_confidence=0.8,
        barge_in_while_speaking=True,
    ),
)


# 全双工: 每句一包 + 段尾一包 paused=true 收口. 尾包**不是**冗余包: 4s 静音期间那句还在长,
# 最后定稿的那一截、以及二遍识别才出来的文本, 都只能落在尾包里 —— 它既回答"他可能停了",
# 也可能带着前面没发过的新内容. 静音取 4s: 一句说完的正常停顿 (换气、想一下) 不该把同一段
# 切成两段, 而 4s 也够我在他真停下时及时接上.
# 其余照 cue: 不打断、走 next —— 什么时候开口是模型的判断, 不是礼仪的.
# **这一档不掐语音**: 出声即让路 (barge_in_while_speaking) 只在 cue 上开. 基线的物理性质是
# 半双工但不抢: 我说到一半他开口, 我把这句说完再让; 谁要立刻拿话筒, 用 cue 或 always.
# low_confidence=0.8: 火山词级置信度里, 0.8 以下基本就是听岔了 —— 默认开着, 让模型
# 自己看到哪几个字可疑 (属性只在有可疑字时出现, 正常句子的帧里不会多一个字节).
duplex = EtiquetteSpec(
    name="duplex",
    description="full duplex — every clause reaches you; a 4s quiet tail says he may have stopped. "
                "Your voice is never cut here: finish the sentence, then he takes the floor.",
    onset=OnsetSpec(emit=False, interrupt=False),
    stop=StopSpec(silence=4.0),
    deliver=DeliverSpec(
        interrupt=False, mode="next", per_clause=True, low_confidence=0.8,
    ),
)


# 长论述判停: 挂一个可编程分类器判"这段说完没有", 打分到阈值提前交付, 静默兜底.
scored = EtiquetteSpec(
    name="scored",
    description="classifier-decided endpoint over a silence fallback",
    stop=StopSpec(
        silence=3.0,
        classifier=ClassifierSpec(
            instruction=(
                "Rate how complete the speaker's thought is, from one utterance of a live "
                "speech transcript, as a single integer 0-9.\n"
                "0-3 = clearly unfinished — cut mid-phrase, ends on a trailing conjunction "
                "or an open condition (because…, if…, 如果…, 因为…), or is only fillers.\n"
                "4-6 = uncertain — could honestly stop here or continue.\n"
                "7-9 = clearly finished — a complete statement, an answerable question, a "
                "greeting, or a closed short answer (yes / no / okay).\n"
                "Judge by meaning: ASR renders homophones and near-sounds (谐音); never trust "
                "surface spelling. If <context> declares an explicit end signal, hearing it is "
                "strong evidence of completion. Output ONLY the digit, no punctuation, no prose."
            ),
            threshold=7,
            delay=0.3,
        ),
    ),
)


# 门铃: 空闲降级后的轻量能量检测 — 不做 ASR, 检测到声音发 knock signal, 让模型决定是否 re-engage.
knock = EtiquetteSpec(
    name="knock",
    description="idle gate — detect sound, knock, let the model decide whether to re-engage",
    stop=StopSpec(detect=DetectSpec()),
    idle_timeout=0.0,
)


class EtiquetteConfig(ConfigType):
    """对话礼仪配置 — the defined etiquettes plus the active one.

    开箱只发五套: ``once`` / ``always`` / ``aside`` / ``cue`` / ``duplex`` ——
    每一套都说得出"这套在什么时候用"的区别. 其余的 (notify / scribe / observer /
    keyword_end / scored / murmur) 只是这三个维度的某种取值, 不再各占一个名字:
    要就直接 ``tune`` 现调, 调好了 ``save`` 成自己的. ``knock`` 不是礼仪, 是空闲
    降级后的轻量门铃, 由 listener 自己持有.
    """

    etiquettes: list[EtiquetteSpec] = Field(
        default_factory=lambda: [
            once.model_copy(deep=True),
            always.model_copy(deep=True),
            aside.model_copy(deep=True),
            cue.model_copy(deep=True),
            duplex.model_copy(deep=True),
        ],
        description="the defined etiquettes (name + description + the three layers)",
    )
    default: str = Field(
        default="duplex",
        description="the active etiquette name",
    )

    # ── 辅助接口 (供 controller / command 使用) ──

    def get(self, name: str) -> EtiquetteSpec | None:
        """按 name 查找礼仪, 无则 None."""
        for spec in self.etiquettes:
            if spec.name == name:
                return spec
        return None

    def upsert(self, spec: EtiquetteSpec) -> None:
        """按 name 增/改一种礼仪 (同名覆盖, 异名追加)."""
        for i, existing in enumerate(self.etiquettes):
            if existing.name == spec.name:
                self.etiquettes[i] = spec
                return
        self.etiquettes.append(spec)

    def activate(self, name: str) -> None:
        """激活一种礼仪 (设 default). 未定义则抛错."""
        if self.get(name) is None:
            raise ValueError(f"etiquette {name!r} not defined")
        self.default = name

    def active(self) -> EtiquetteSpec | None:
        """当前激活的礼仪 spec, 无 default 则 None."""
        return self.get(self.default)

    @classmethod
    def conf_name(cls) -> str:
        return "listener_etiquette"


# ── voiceflow: the live state as three flat dimensions ──
#
# 一套礼仪是一个点, 三个维度是坐标系. 这两个函数把点摊平成可读的键值, 让模型看得见
# 「现在是什么样」「跟出身比哪一格动了」. 摊平是给**判断**用的投影, 不是规范真值 ——
# 真值永远是 EtiquetteSpec 本身; classifier / detect 这种嵌套对象在投影里退化成
# 一个开关 (有没有挂 / 开没开), tune 也只认这个开关, 要改细节请整份 spec 覆盖.

#: 摊平之后的键 → 改动它是否需要重开会话. A 判停是会话启动时被 copyt 进 StopJudge 的,
#: 现场改对象到不了正在跑的那一版; B 交付 / C 打断 每条信号现读 ``_active_etiquette``,
#: 所以改完立刻生效.
_RESTART_KEYS = frozenset({"stop.silence", "stop.keywords", "stop.classifier", "stop.detect"})


def flatten_interaction(spec: "EtiquetteSpec") -> dict[str, object]:
    """把一套礼仪摊平成三个维度的键值 (只含可读可调的标量)."""
    stop, deliver, onset = spec.stop, spec.deliver, spec.onset
    return {
        "stop.silence": stop.silence,
        "stop.keywords": list(stop.keywords),
        "stop.classifier": stop.classifier.threshold if stop.classifier is not None else None,
        "stop.detect": stop.detect.enabled if stop.detect is not None else None,
        "deliver.emit": deliver.emit,
        "deliver.per_clause": deliver.per_clause,
        "deliver.mode": deliver.mode,
        "deliver.priority": int(deliver.priority),
        "deliver.history": deliver.history,
        "deliver.low_confidence": deliver.low_confidence,
        "deliver.interrupt": deliver.interrupt,
        "deliver.barge_in_while_speaking": deliver.barge_in_while_speaking,
        "onset.emit": onset.emit,
        "onset.interrupt": onset.interrupt,
        "onset.priority": int(onset.priority),
    }


def deviated_keys(origin: "EtiquetteSpec | None", current: "EtiquetteSpec") -> list[str]:
    """跟出身比, 哪几格被改过 (origin 为空 = 没有出身, 全部算偏离)."""
    now = flatten_interaction(current)
    if origin is None:
        return sorted(now)
    was = flatten_interaction(origin)
    return sorted(key for key, value in now.items() if was.get(key) != value)


def apply_interaction_patch(spec: "EtiquetteSpec", patch: dict[str, object]) -> list[str]:
    """把一组键值就地写进 spec, 返回真正动到的键 (按名字排序).

    只认 ``flatten_interaction`` 摊出来的那几格; 不认的键当场抛 ValueError ——
    「写进去了但没生效」比直接报错更难查.
    """
    known = flatten_interaction(spec)
    unknown = sorted(set(patch) - set(known))
    if unknown:
        raise ValueError(f"unknown interaction keys: {unknown}; known: {sorted(known)}")
    touched: list[str] = []
    for key, value in patch.items():
        if key == "stop.silence":
            spec.stop.silence = float(value)
        elif key == "stop.keywords":
            spec.stop.keywords = [str(word) for word in (value or [])]
        elif key == "stop.classifier":
            # 只认开关: 有值 = 挂一套默认判官, None = 摘掉. 细调走 set_etiquette_spec.
            if value is None:
                spec.stop.classifier = None
            elif spec.stop.classifier is None:
                spec.stop.classifier = ClassifierSpec(threshold=int(value))
            else:
                spec.stop.classifier.threshold = int(value)
        elif key == "stop.detect":
            if value is None:
                spec.stop.detect = None
            elif spec.stop.detect is None:
                spec.stop.detect = DetectSpec(enabled=bool(value))
            else:
                spec.stop.detect.enabled = bool(value)
        elif key == "deliver.emit":
            spec.deliver.emit = bool(value)
        elif key == "deliver.per_clause":
            spec.deliver.per_clause = bool(value)
        elif key == "deliver.mode":
            spec.deliver.mode = value  # type: ignore[assignment]
        elif key == "deliver.priority":
            spec.deliver.priority = Priority(int(value))
        elif key == "deliver.history":
            spec.deliver.history = int(value)
        elif key == "deliver.low_confidence":
            spec.deliver.low_confidence = None if value is None else float(value)
        elif key == "deliver.interrupt":
            spec.deliver.interrupt = bool(value)
        elif key == "deliver.barge_in_while_speaking":
            spec.deliver.barge_in_while_speaking = bool(value)
        elif key == "onset.emit":
            spec.onset.emit = bool(value)
        elif key == "onset.interrupt":
            spec.onset.interrupt = bool(value)
        elif key == "onset.priority":
            spec.onset.priority = Priority(int(value))
        else:  # pragma: no cover — guarded by the unknown-key check above
            raise ValueError(f"unhandled interaction key: {key}")
        touched.append(key)
    return sorted(touched)


def needs_session_restart(keys: "list[str]") -> bool:
    """这些键里有没有必须重开会话才能生效的 (A 判停那一层)."""
    return any(key in _RESTART_KEYS for key in keys)
