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

    @classmethod
    def conf_name(cls) -> str:
        return "memory"
