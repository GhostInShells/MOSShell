"""Data Ghost bootstrapper."""

from collections.abc import Callable
from pathlib import Path

from ghoshell_container import IoCContainer
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ghoshell_moss.contracts import ConfigStore, SystemPrompter
from ghoshell_moss.contracts.configs import get_conf, get_or_create_conf
from ghoshell_moss.contracts.llms import LLMConfig, ResolvedModel
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta, GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta

from ._config import MemoryConfig

__all__ = ["DataMeta"]


class DataMeta(GhostMeta):
    """Bootstrapper for the Memento-backed Data Ghost prototype."""

    def __init__(
        self,
        name: str = "data",
        description: str = "Data Ghost — persistent Memento-backed conversation.",
        *,
        soul_path: str | Path | None = None,
        soul_content: str | None = None,
        model: Model | None = None,
        llm_provider: str = "",
        llm_model: str = "",
        llm_tag: str | None = None,
        reflection_model: Model | None = None,
        memory_root: str | Path | None = None,
        memory_owner: str = "",
        memory_detail_n: int | None = None,
        memory_summary_m: int | None = None,
        auto_commit_every: int | None = None,
        reflection_enabled: bool | None = None,
        on_agent_build: Callable[[Agent[IoCContainer]], None] | None = None,
        nuclei_metas: list[NucleusMeta] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._soul_path = soul_path
        self._soul_content = soul_content
        self._model = model
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_tag = llm_tag
        self._reflection_model = reflection_model
        self._memory_root = Path(memory_root) if memory_root is not None else None
        self._memory_owner = memory_owner
        self._memory_detail_n = memory_detail_n
        self._memory_summary_m = memory_summary_m
        self._auto_commit_every = auto_commit_every
        self._reflection_enabled = reflection_enabled
        self._on_agent_build = on_agent_build
        self._nuclei_metas = nuclei_metas or []

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_metas(self) -> list[NucleusMeta]:
        return self._nuclei_metas

    @property
    def soul_content(self) -> str:
        return self._soul_content or ""

    def _load_soul(self, workspace: GhostWorkspace) -> None:
        if self._soul_content is not None:
            return
        path = Path(self._soul_path or "soul.md")
        if not path.is_absolute():
            path = workspace.home / path
        if path.exists():
            self._soul_content = path.read_text(encoding="utf-8")

    def build_instruction_from_ioc(self, container: IoCContainer) -> str:
        prompter = container.get(SystemPrompter)
        parts = [prompter.instruction()] if prompter is not None else []
        if self.soul_content:
            parts.append(self.soul_content)
        return "\n".join(parts)

    def build_instruction(self, context: RunContext[IoCContainer]) -> str:
        return self.build_instruction_from_ioc(context.deps)

    def _resolved_model(self, container: IoCContainer, *, tag: str | None = None) -> ResolvedModel:
        if container.bound(ConfigStore):
            config = get_conf(container, LLMConfig)
        else:
            config = LLMConfig().resolve()
        resolved = config.get_model(
            provider=self._llm_provider,
            model=self._llm_model,
        )
        selected_tag = tag if tag is not None else self._llm_tag
        if selected_tag:
            resolved = resolved.model_copy(update={"model": resolved.model.unwrap_tag(selected_tag)})
        return resolved

    def _memory_config(self, container: IoCContainer) -> MemoryConfig:
        if container.bound(ConfigStore):
            return get_or_create_conf(container, MemoryConfig())
        return MemoryConfig()

    @staticmethod
    def _build_configured_model(resolved: ResolvedModel) -> Model:
        values = {
            "api_key": resolved.service.api_key,
            "model": resolved.model.model,
        }
        missing = [key for key, value in values.items() if not value or value.startswith("$")]
        if missing:
            raise RuntimeError(f"LLM configuration is unresolved: {', '.join(missing)}")
        if resolved.client_protocol == "anthropic":
            provider = AnthropicProvider(
                api_key=resolved.service.api_key,
                base_url=resolved.service.base_url,
            )
            return AnthropicModel(resolved.model.model, provider=provider)
        provider = OpenAIProvider(
            api_key=resolved.service.api_key,
            base_url=resolved.service.base_url,
        )
        return OpenAIModel(resolved.model.model, provider=provider)

    def build_agent(self, container: IoCContainer) -> Agent[IoCContainer]:
        workspace = container.get(GhostWorkspace)
        if workspace is not None:
            self._load_soul(workspace)
        model = self._model or self._build_configured_model(self._resolved_model(container))
        agent = Agent[IoCContainer](
            name=self._name,
            description=self._description,
            instructions=self.build_instruction,
            model=model,
        )
        if self._on_agent_build is not None:
            self._on_agent_build(agent)
        return agent

    def build_reflection_agent(
        self,
        container: IoCContainer,
        config: MemoryConfig,
        *,
        enabled: bool,
    ) -> Agent[IoCContainer] | None:
        if not enabled:
            return None
        model = self._reflection_model or self._model
        if model is None:
            model = self._build_configured_model(
                self._resolved_model(container, tag=config.reflection_model_tag),
            )
        return Agent[IoCContainer](
            name=f"{self._name}-memory-reflector",
            description="Background Memento reflection worker.",
            instructions=(
                "Summarize only observable completed conversation. Never invent facts or reveal hidden reasoning."
            ),
            model=model,
        )

    def factory(self, container: IoCContainer) -> Ghost:
        from ._runtime import Data

        workspace = container.force_fetch(GhostWorkspace)
        config = self._memory_config(container)
        reflection_enabled = (
            self._reflection_enabled if self._reflection_enabled is not None else config.reflection_enabled
        )
        if self._memory_root is None:
            root = workspace.home / "memento"
        elif self._memory_root.is_absolute():
            root = self._memory_root
        else:
            root = workspace.home / self._memory_root
        detail_n = self._memory_detail_n if self._memory_detail_n is not None else config.detail_n
        summary_m = self._memory_summary_m if self._memory_summary_m is not None else config.summary_m
        auto_commit_every = (
            self._auto_commit_every if self._auto_commit_every is not None else config.auto_commit_every
        )
        return Data(
            meta=self,
            agent=self.build_agent(container),
            container=container,
            memory_root=root,
            memory_owner=self._memory_owner or self._name,
            memory_detail_n=detail_n,
            memory_summary_m=summary_m,
            auto_commit_every=auto_commit_every,
            reflection_agent=self.build_reflection_agent(container, config, enabled=reflection_enabled),
            reflection_max_summary_chars=config.reflection_max_summary_chars,
            reflection_max_source_chars=config.reflection_max_source_chars,
            reflection_startup_limit=config.reflection_startup_limit,
            reflection_enabled=reflection_enabled,
        )
