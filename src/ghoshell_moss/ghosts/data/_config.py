"""Persistent policy for Data's Memento memory."""

from pydantic import Field

from ghoshell_moss.contracts.configs import ConfigType

__all__ = ["MemoryConfig"]


class MemoryConfig(ConfigType):
    """Data Ghost memory policy, stored as workspace ``configs/memory.yml``."""

    detail_n: int = Field(default=12, ge=1, description="Recent complete Moments kept in model context.")
    summary_m: int = Field(default=-1, ge=-1, description="Earlier commit summaries; -1 means all.")
    auto_commit_every: int = Field(
        default=4, ge=0, description="Stage count before a mechanical commit; 0 disables it."
    )
    reflection_enabled: bool = Field(default=True, description="Run post-commit reflection in a background task.")
    reflection_model_tag: str = Field(default="small_fast_model", description="LLMConfig tag used by the reflector.")
    reflection_max_summary_chars: int = Field(default=360, ge=32, description="Maximum persisted reflection length.")
    reflection_max_source_chars: int = Field(
        default=12000, ge=256, description="Maximum source transcript sent to reflection."
    )
    reflection_startup_limit: int = Field(
        default=16, ge=0, description="Maximum pending commits chased at startup; 0 means none."
    )

    @classmethod
    def conf_name(cls) -> str:
        return "memory"
