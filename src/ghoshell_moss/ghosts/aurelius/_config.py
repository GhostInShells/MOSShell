"""Persistent policy for Aurelius's Memento memory."""

from pydantic import Field

from ghoshell_moss.contracts.configs import ConfigType

__all__ = ["MemoryConfig"]


class MemoryConfig(ConfigType):
    """Aurelius memory policy, stored as workspace ``configs/memory.yml``."""

    detail_n: int = Field(default=12, ge=1, description="模型上下文保留的最近完整 Moment 数量。")
    summary_m: int = Field(default=-1, ge=-1, description="进入模型上下文的早期 CommitNote 数；-1 表示全部。")
    auto_commit_every: int = Field(
        default=4, ge=0, description="达到该 staged Moment 数后创建 mechanical commit；0 关闭自动冻结。"
    )
    reflection_enabled: bool = Field(default=True, description="是否在 commit 后后台执行反思。")
    reflection_model_tag: str = Field(default="small_fast_model", description="反思器在 LLMConfig 中解析的模型 tag。")
    reflection_max_summary_chars: int = Field(default=360, ge=32, description="单条持久化反思 note 的最大字符数。")
    reflection_max_source_chars: int = Field(
        default=12000, ge=256, description="单次发送给反思模型的冻结可见原文最大字符数。"
    )
    reflection_startup_limit: int = Field(
        default=16, ge=0, description="单次启动最多追赶的待反思 commit 数；0 表示不追赶。"
    )
    curation_enabled: bool = Field(
        default=True, description="是否启用旁路 curation：小模型从冻结轨迹重写记忆笔记文件并 pin 进 Ground。"
    )
    curation_model_tag: str = Field(
        default="small_fast_model", description="curation 旁路在 LLMConfig 中解析的模型 tag。"
    )
    curation_index_sources: tuple[str, ...] = Field(
        default=("input_signal_nucleus", "input", "user"),
        description="mechanical 索引与 curation 摘录时视为用户输入的 Moment percept source。",
    )
    curation_max_source_chars: int = Field(
        default=12000, ge=256, description="单次发给 curation 模型的可见 commit log 最大字符数。"
    )
    curation_max_notes_chars: int = Field(
        default=4000, ge=256, description="持久化 facts.md 笔记正文的最大字符数。"
    )
    curation_notes_name: str = Field(
        default="facts.md", description="策展笔记文件名，写在 Ground 根下并被 pin 进当前帧。"
    )
    memory_discipline: str = Field(
        default=(
            "你的全部经历都保存在 Memento 轨迹里。回答涉及具体事实（代号、编号、地址、名称等）时，"
            "若当前上下文没有明确可见的依据，先用 memory_search 检索、再用 memory_show 展开冻结原文核对，"
            "并在回答中标注 commit id。无法核对时明确说“没有找到记忆证据”，不要猜测。"
        ),
        description="注入 instruction 的记忆纪律；替代旧的正则 verifier，把判断权交还模型。",
    )
    desktop_enabled: bool = Field(
        default=True, description="是否在 Aurelius 生命周期中自动打开项目 Ground 并注入当前帧。"
    )
    context_budget_enabled: bool = Field(
        default=True,
        description="是否按模型 token 预算主动收缩渲染窗口并在 provider 溢出时兜底重试。关闭则退回一次性组装。",
    )
    context_token_margin: int = Field(
        default=4096,
        ge=0,
        description="预算安全垫 (tokens)：从 context_window-max_output_tokens 再扣除，吸收估算误差。",
    )
    context_min_detail_n: int = Field(
        default=2,
        ge=1,
        description="主动收缩时明细窗口的下限；触底仍超预算则交给溢出兜底重试。",
    )
    context_fixed_overhead_tokens: int = Field(
        default=2048,
        ge=0,
        description="每次请求 system prompt + 当前输入 + ground 注入的固定开销估算，计入预算占用。",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "memory"
