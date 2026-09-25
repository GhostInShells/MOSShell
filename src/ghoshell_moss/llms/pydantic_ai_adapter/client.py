"""Build pydantic-ai model/agent from a resolved LLM config.

pydantic-ai is imported lazily — this module loads without the `ghost` extra;
the pydantic-ai import happens only when ``build_agent()`` actually runs.
"""

from __future__ import annotations

from ghoshell_moss.contracts.llms import CallSettings, Effort, ResolvedModel

__all__ = ["build_agent"]


def build_agent(
        resolved: ResolvedModel,
        *,
        settings: CallSettings | None = None,
        effort: Effort | None = None,
) -> "Agent":
    """Build a pydantic-ai Agent from a resolved model.

    :param resolved: ``LLMConfig.get_model()`` 的结果. 其 ``service.base_url`` /
        ``api_key`` 必须是已 resolve 的真实值 — 仅内存使用, 调用方绝不打印.
    :param settings: 采样参数对象 (temperature / max_output_tokens).
    :param effort: thinking effort 刻度, 按协议映射 (anthropic_effort /
        openai_reasoning_effort). None 或 "none" 表示默认 (anthropic 保持
        extended thinking disabled).
    """
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings

    model_settings: dict = {}
    if settings is not None:
        if settings.temperature is not None:
            model_settings["temperature"] = settings.temperature
        if settings.max_output_tokens is not None:
            model_settings["max_tokens"] = settings.max_output_tokens
    return Agent(
        model=_build_model(resolved, effort=effort),
        model_settings=ModelSettings(**model_settings) if model_settings else None,
    )


def _build_model(resolved: ResolvedModel, effort: Effort | None = None):
    protocol = resolved.client_protocol
    service = resolved.service
    model_name = resolved.model.model

    if protocol == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
        from pydantic_ai.providers.anthropic import AnthropicProvider

        kwargs: dict = {}
        if effort and effort != "none":
            # anthropic_effort: low/medium/high/xhigh/max — 新式统一 effort。
            # 传了 effort 就不设 disabled thinking, 两者互斥。
            kwargs["anthropic_effort"] = effort
        else:
            from anthropic.types.beta import BetaThinkingConfigDisabledParam
            kwargs["anthropic_thinking"] = BetaThinkingConfigDisabledParam(type="disabled")

        return AnthropicModel(
            model_name=model_name,
            provider=AnthropicProvider(api_key=service.api_key, base_url=service.base_url),
            settings=AnthropicModelSettings(**kwargs),
        )

    if protocol == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
        from pydantic_ai.providers.openai import OpenAIProvider

        kwargs = {}
        if effort:
            # openai_reasoning_effort: none/minimal/low/medium/high/xhigh
            kwargs["openai_reasoning_effort"] = effort

        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=service.base_url, api_key=service.api_key),
            settings=OpenAIChatModelSettings(**kwargs) if kwargs else None,
        )

    raise ValueError(f"unsupported ClientProtocol: {protocol!r}")
