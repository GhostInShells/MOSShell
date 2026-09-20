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
    "JudgeSpec",
    "StopSpec",
    "PerceiveSpec",
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


class JudgeSpec(BaseModel):
    """LLM 判停组件参数 — 出口位点上的一个降级实现 (openbox).

    它声明"挂不挂这个组件 + 它自己的参数", 不是"判停是什么". 将来专用端点
    模型进来时, 换掉的是这个组件, ``StopSpec`` 的槽位形状不变.
    """

    threshold: int = Field(default=7, description="llm judge score threshold")
    judge_delay: float = Field(default=0.3, description="llm judge debounce seconds")


class StopSpec(BaseModel):
    """判停 — 出口协议: 声明出口位点上组装了哪些判停组件.

    通用槽位参数 (任何实现都认):

    - ``segment_vad == 0`` → 首个 clause 立刻 commit (once).
    - ``segment_vad > 0`` → 该秒数静默后 commit (展期: 新 clause 前移 deadline).
    - ``keywords`` → 显式端点, 命中立刻 commit.

    可组装件 (声明式, None = 不挂):

    - ``judge`` → 挂一个 LLM 打分实现 (降级; 依赖外部注入的 caller).
      caller 缺席时该组件静默不生效, 礼仪退回纯 segment_vad/keywords.
    """

    segment_vad: float = Field(default=1.5, description="silence fallback seconds; 0 = commit on first clause")
    keywords: list[str] = Field(default_factory=list, description="explicit endpoint keywords")
    judge: JudgeSpec | None = Field(
        default=None,
        description="openbox fallback component — off by default; needs a caller injected at construction",
    )


class PerceiveSpec(BaseModel):
    """感知协议 — 是否把语音流保留成可拉读的槽位 (segment buffer).

    ``enabled`` 是开/关 (off = signal-only, 默认); ``history`` 是跨 session 保留的
    最近 n 轮 segment 环形容量. 与 first_packet/deliver/stop 平级: 前三层决定"何时
    判停 + 首尾包怎么发", 这一层决定"模型能否在 signal 之外拉读听到的内容".
    """

    enabled: bool = Field(default=False, description="on = retain + expose recent segments for pull; off = signal-only")
    history: int = Field(default=8, description="recent n segments retained in the ring buffer")


class EtiquetteSpec(BaseModel):
    """一种礼仪 — name + description + the four layers of config."""

    name: str = Field(description="etiquette name, referenced by EtiquetteConfig.default")
    description: str = Field(default="", description="one-line self-description for model perception")
    first_packet: FirstPacketSpec = Field(default_factory=FirstPacketSpec)
    deliver: DeliverSpec = Field(default_factory=DeliverSpec)
    stop: StopSpec = Field(default_factory=StopSpec)
    perceive: PerceiveSpec = Field(default_factory=PerceiveSpec)


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


def new_llm_judge_spec(
        segment_vad: float = 3.0,
        threshold: int = 7,
        judge_delay: float = 0.3,
) -> EtiquetteSpec:
    """降级判停: 出口位点挂 LLM 打分件, 打分 >= threshold 提前 commit, segment_vad 兜底.

    不是默认路径 —— 默认出口只有 segment_vad/keywords. 挂上本件要求环境配了模型
    (caller 能注入), 拿不到时该件静默缺席, 礼仪退回默认行为.
    """
    return EtiquetteSpec(
        name="llm_judge",
        description="openbox fallback — llm-scored stop detection over the segment_vad baseline",
        stop=StopSpec(
            segment_vad=segment_vad,
            judge=JudgeSpec(threshold=threshold, judge_delay=judge_delay),
        ),
    )
