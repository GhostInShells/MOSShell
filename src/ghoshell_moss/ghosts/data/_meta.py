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
from ghoshell_moss.contracts.configs import get_conf
from ghoshell_moss.contracts.llms import LLMConfig, ResolvedModel
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta, GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta

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
        memory_root: str | Path | None = None,
        memory_owner: str = "",
        memory_detail_n: int = 12,
        memory_summary_m: int = -1,
        auto_commit_every: int = 4,
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
        self._memory_root = Path(memory_root) if memory_root is not None else None
        self._memory_owner = memory_owner
        self._memory_detail_n = memory_detail_n
        self._memory_summary_m = memory_summary_m
        self._auto_commit_every = auto_commit_every
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

    def _resolved_model(self, container: IoCContainer) -> ResolvedModel:
        if container.bound(ConfigStore):
            config = get_conf(container, LLMConfig)
        else:
            config = LLMConfig().resolve()
        return config.get_model(
            provider=self._llm_provider,
            model=self._llm_model,
            tag=self._llm_tag,
        )

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

    def factory(self, container: IoCContainer) -> Ghost:
        from ._runtime import Data

        workspace = container.force_fetch(GhostWorkspace)
        if self._memory_root is None:
            root = workspace.home / "memento"
        elif self._memory_root.is_absolute():
            root = self._memory_root
        else:
            root = workspace.home / self._memory_root
        return Data(
            meta=self,
            agent=self.build_agent(container),
            container=container,
            memory_root=root,
            memory_owner=self._memory_owner or self._name,
            memory_detail_n=self._memory_detail_n,
            memory_summary_m=self._memory_summary_m,
            auto_commit_every=self._auto_commit_every,
        )
