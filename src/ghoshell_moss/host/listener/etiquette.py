"""Etiquette — the interaction styles a model programs at runtime.

One etiquette configures one listening session. It has four independent layers,
and an interaction style is a point in their product space — not a code branch:

- **onset** (起话) — what to signal the instant speech begins.
- **stop** (判停) — how the turn end is decided.
- **deliver** (交付) — what the committed turn sends.
- **retain** (留存) — whether heard segments stay pullable.

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
    "StopSpec",
    "RetainSpec",
    "EtiquetteSpec",
    "EtiquetteConfig",
    "once",
    "always",
    "aside",
    "notify",
    "scribe",
    "observer",
    "keyword_end",
    "scored",
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
    """交付协议 — what the committed turn sends."""

    emit: bool = Field(
        default=True,
        description="emit a signal when the turn is committed",
    )
    interrupt: bool = Field(
        default=False,
        description="that signal stops the current behavior",
    )
    priority: Priority = Field(
        default=Priority.INFO,
        description="preempt tier of the deliver signal",
    )
    mode: DeliverMode = Field(
        default="notify",
        description="loss-side behavior when the deliver signal cannot preempt "
                    "(aligns with mindflow ChallengeMode): "
                    "'notify' (buffer — answer when idle) / "
                    "'aside' (inject without taking over) / "
                    "'next' (buffer + take next turn) / "
                    "'' (default — suppress on preempt failure)",
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


class StopSpec(BaseModel):
    """判停 — how the turn end is decided. Three composable commit paths:

    - ``silence``: commit N seconds after the last clause (0 = commit on the first clause).
    - ``keywords``: an explicit endpoint — a hit commits immediately.
    - ``classifier``: a score-based early commit; None = not mounted.
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


class RetainSpec(BaseModel):
    """留存 — whether heard segments stay pullable after the fact."""

    enabled: bool = Field(
        default=False,
        description="on = keep recent segments readable via the pull slot; off = signal-only",
    )
    history: int = Field(
        default=8,
        description="recent segments kept in the ring",
    )


class EtiquetteSpec(BaseModel):
    """一种礼仪 — name + description + the four layers."""

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
    retain: RetainSpec = Field(default_factory=RetainSpec)


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
    deliver=DeliverSpec(emit=False),
    retain=RetainSpec(enabled=True, history=16),
)


# 对讲机: 显式端点, 说完喊 over 才交付, 不靠静默.
keyword_end = EtiquetteSpec(
    name="keyword_end",
    description="walkie-talkie — commit only on an explicit end keyword",
    stop=StopSpec(silence=30.0, keywords=["over", "完毕"]),
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


class EtiquetteConfig(ConfigType):
    """对话礼仪配置 — the defined etiquettes plus the active one."""

    etiquettes: list[EtiquetteSpec] = Field(
        default_factory=lambda: [
            once.model_copy(deep=True),
            always.model_copy(deep=True),
            aside.model_copy(deep=True),
            notify.model_copy(deep=True),
            scribe.model_copy(deep=True),
            observer.model_copy(deep=True),
            keyword_end.model_copy(deep=True),
            scored.model_copy(deep=True),
        ],
        description="the defined etiquettes (name + description + the four layers)",
    )
    default: str = Field(
        default="always",
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
