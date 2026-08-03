"""Build pydantic-ai model/agent from a resolved LLM config.

pydantic-ai is imported lazily — this module loads without the `ghost` extra;
the pydantic-ai import happens only when ``build_agent()`` actually runs.
"""

from __future__ import annotations

from ghoshell_moss.contracts.llms import ResolvedModel

__all__ = ["build_agent"]


def build_agent(
        resolved: ResolvedModel,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
) -> "Agent":
    """Build a pydantic-ai Agent from a resolved model.

    :param resolved: ``LLMConfig.get_model()`` 的结果. 其 ``service.base_url`` /
        ``api_key`` 必须是已 resolve 的真实值 — 仅内存使用, 调用方绝不打印.
    """
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings

    settings: dict = {}
    if temperature is not None:
        settings["temperature"] = temperature
    if max_output_tokens is not None:
        settings["max_tokens"] = max_output_tokens
    return Agent(
        model=_build_model(resolved),
        model_settings=ModelSettings(**settings) if settings else None,
    )


def _build_model(resolved: ResolvedModel):
    protocol = resolved.client_protocol
    service = resolved.service
    model_name = resolved.model.model

    if protocol == "anthropic":
        from anthropic.types.beta import BetaThinkingConfigDisabledParam
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            model_name=model_name,
            provider=AnthropicProvider(api_key=service.api_key, base_url=service.base_url),
            # disable extended thinking by default; enable via model param if needed
            settings=AnthropicModelSettings(
                anthropic_thinking=BetaThinkingConfigDisabledParam(type="disabled"),
            ),
        )

    if protocol == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=service.base_url, api_key=service.api_key),
        )

    raise ValueError(f"unsupported ClientProtocol: {protocol!r}")
