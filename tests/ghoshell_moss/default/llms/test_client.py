"""Tests for build_agent — pydantic-ai model construction with effort mapping."""

from ghoshell_moss.contracts.llms import ResolvedModel, ServiceConfig, ModelConfig
from ghoshell_moss.llms.pydantic_ai_adapter.client import build_agent


def _resolved(protocol: str, model: str) -> ResolvedModel:
    return ResolvedModel(
        service=ServiceConfig(name="s", base_url="http://x", api_key="k", protocol=protocol),
        model=ModelConfig(model=model),
    )


class TestBuildAgentEffort:
    """Protocol 契约: effort 映射到 anthropic_effort / openai_reasoning_effort."""

    def test_anthropic_default_disables_thinking(self):
        agent = build_agent(_resolved("anthropic", "claude-sonnet-4-6"))
        assert agent.model.settings == {"anthropic_thinking": {"type": "disabled"}}

    def test_anthropic_effort_replaces_disabled(self):
        agent = build_agent(_resolved("anthropic", "claude-sonnet-4-6"), effort="high")
        assert agent.model.settings == {"anthropic_effort": "high"}

    def test_anthropic_effort_none_keeps_disabled(self):
        """'none' 显式表达不启用 thinking — 保持 disabled 基线."""
        agent = build_agent(_resolved("anthropic", "claude-sonnet-4-6"), effort="none")
        assert agent.model.settings == {"anthropic_thinking": {"type": "disabled"}}

    def test_openai_default_no_settings(self):
        agent = build_agent(_resolved("openai", "gpt-5"))
        assert agent.model.settings is None

    def test_openai_effort(self):
        agent = build_agent(_resolved("openai", "gpt-5"), effort="medium")
        assert agent.model.settings == {"openai_reasoning_effort": "medium"}

    def test_temperature_max_tokens_go_to_agent_settings(self):
        agent = build_agent(
            _resolved("openai", "gpt-5"), temperature=0.3, max_output_tokens=512,
        )
        assert agent.model_settings["temperature"] == 0.3
        assert agent.model_settings["max_tokens"] == 512
