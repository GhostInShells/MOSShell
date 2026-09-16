"""Conversation etiquette — the config-driven listening state machine model.

An etiquette is "another always": a continuous listening session whose behavior
is defined by three orthogonal layers of config, not by a code branch:

- **first_packet** (首包协议) — what happens the moment speech begins (barge-in).
- **deliver** (尾包协议) — what the send signal carries once stop is decided.
- **stop** (判停) — when to commit (the send trigger). llm_judge is one instance
  of this layer, not a separate machine.

The config is independent of the listener organ (ASR params live elsewhere): it
belongs to ListenerController, is resolved via the config store, and can be
overridden at ghost/node level via ``with_config``. An array carries multiple
named etiquettes so the model can perceive the available interaction styles and
pull their details on demand (skill-like), rather than having them hard-coded.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.blueprint.mindflow import Priority

__all__ = [
    "FirstPacketSpec",
    "DeliverSpec",
    "StopSpec",
    "EtiquetteSpec",
    "EtiquetteConfig",
    "new_once_spec",
    "new_always_spec",
    "new_llm_judge_spec",
]


class FirstPacketSpec(BaseModel):
    """首包协议 — the barge-in behavior when speech first arrives.

    ``barge_in`` is the on/off switch for the first-packet interrupt signal;
    ``interrupt`` decides whether to also stop the current behavior (logos);
    ``priority`` is the preempt tier used to win attention.
    """

    barge_in: bool = Field(default=True, description="whether to emit a first-packet barge-in signal")
    interrupt: bool = Field(default=True, description="whether to stop current behavior before attending")
    priority: Priority = Field(default=Priority.WARNING, description="preempt tier of the barge-in signal")


class DeliverSpec(BaseModel):
    """尾包协议 — what the send signal carries once stop is decided.

    ``interrupt`` stops current behavior; ``priority`` is the preempt tier
    (default INFO — same tier as a running attention, so it does not interrupt
    but still responds when idle); ``mode`` is the loss-side semantics
    (notify / aside / '').
    """

    interrupt: bool = Field(default=False, description="whether to stop current behavior on deliver")
    priority: Priority = Field(default=Priority.INFO, description="preempt tier of the deliver signal")
    mode: str = Field(default="notify", description="loss-side semantics: notify / aside / ''(default)")


class StopSpec(BaseModel):
    """判停 — when to commit (the trigger of the deliver signal).

    Orthogonal params, not an enum:

    - ``segment_vad == 0`` → commit immediately on the first clause (once).
    - ``segment_vad > 0`` → commit after that many seconds of quiet.
    - ``judge`` → score clauses with an llm caller (llm_judge); commit early when
      score >= threshold, segment_vad as fallback.
    - ``keywords`` → explicit endpoint, commits immediately on a hit.
    """

    segment_vad: float = Field(default=1.5, description="silence fallback seconds; 0 = commit on first clause")
    judge: bool = Field(default=False, description="whether to llm-judge clause completion")
    judge_delay: float = Field(default=0.3, description="llm judge debounce seconds")
    threshold: int = Field(default=7, description="llm judge score threshold")
    keywords: list[str] = Field(default_factory=list, description="explicit endpoint keywords")


class EtiquetteSpec(BaseModel):
    """一种礼仪 — name + description + the three layers of config."""

    name: str = Field(description="etiquette name, referenced by EtiquetteConfig.default")
    description: str = Field(default="", description="one-line self-description for model perception")
    first_packet: FirstPacketSpec = Field(default_factory=FirstPacketSpec)
    deliver: DeliverSpec = Field(default_factory=DeliverSpec)
    stop: StopSpec = Field(default_factory=StopSpec)


class EtiquetteConfig(ConfigType):
    """对话礼仪配置 — an array of named etiquettes plus the default activation.

    Owned by ListenerController (not the listener organ). The array makes the
    available interaction styles perceivable; ``default`` names the active one.
    """

    etiquettes: list[EtiquetteSpec] = Field(
        default_factory=lambda: [new_once_spec(), new_always_spec(), new_llm_judge_spec()],
        description="the defined etiquettes (name + description + three layers)",
    )
    default: str = Field(default="always", description="the active etiquette name")

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


# ── 预组装基线 (once / always / llm_judge) ──


def new_once_spec() -> EtiquetteSpec:
    """主动听一次: 判停 = clause 立刻 commit (segment_vad=0)."""
    return EtiquetteSpec(
        name="once",
        description="hear one utterance — commit on the first clause",
        stop=StopSpec(segment_vad=0.0),
    )


def new_always_spec(segment_vad: float = 1.5) -> EtiquetteSpec:
    """打断式 turn-taking: 判停 = segment_vad 静默 commit. 启动默认."""
    return EtiquetteSpec(
        name="always",
        description="keep listening — commit after segment_vad seconds of quiet",
        stop=StopSpec(segment_vad=segment_vad),
    )


def new_llm_judge_spec(segment_vad: float = 3.0, threshold: int = 7) -> EtiquetteSpec:
    """智能判停: 判停 = llm 打分 >= threshold 提前 commit, segment_vad 兜底."""
    return EtiquetteSpec(
        name="llm_judge",
        description="llm-judged stop detection with segment_vad fallback",
        stop=StopSpec(segment_vad=segment_vad, judge=True, threshold=threshold),
    )
