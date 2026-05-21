import os
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from anthropic.types.beta import BetaThinkingConfigDisabledParam
from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts.system_prompter import SystemPrompter
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.providers import Provider
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from ._runtime import Atom

__all__ = ["AtomMeta"]


class AtomMeta(GhostMeta):
    """Atom — 最小 Ghost 原型，作为后续所有 Ghost 的参照基线.

    Atom 演示了 Ghost ABC 的最小契约实现：
    - soul + model → build_agent() → factory() → Atom runtime
    - 单轮 articulate() 循环，纯内存历史
    """

    def __init__(
            self,
            name: str = "atom",
            description: str = (
                    "Atom is the minimal Ghost prototype — a reference baseline "
                    "for all Ghost implementations in MOSS. It demonstrates the "
                    "Ghost ABC contract with a single-turn articulate() loop."
            ),
            soul_path: str | Path | None = None,
            soul_content: str | None = None,
            model: Model | None = None,
            provider: Provider | None = None,
            on_agent_build: Callable[[Agent[IoCContainer]], None] | None = None,
            nuclei_metas: list[NucleusMeta] | None = None,
    ):
        self._name = name
        self._description = description
        self._soul_path = soul_path
        self._soul_content = soul_content
        self._model = model
        self._provider = provider
        self._on_agent_build = on_agent_build
        self._nuclei_metas = nuclei_metas or []

    # ── GhostMeta ABC ──────────────────────────────

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def nuclei_metas(self) -> list[NucleusMeta]:
        return self._nuclei_metas

    # ── soul ────────────────────────────────────────

    @property
    def soul_content(self) -> str:
        """已加载的 soul 内容（由 _load_soul 或构造时直接传入)."""
        return self._soul_content or ""

    def _load_soul(self, workspace_root: Path) -> None:
        """从 workspace/souls/ 加载 soul 文件. soul_content 非 None 时跳过."""
        if self._soul_content is not None:
            return

        path = self._soul_path
        if path is None:
            path = self._name
        if isinstance(path, Path):
            file = path
        else:
            file = workspace_root / "souls" / f"{path}.md"
        if file.exists():
            self._soul_content = file.read_text(encoding="utf-8")

    # ── agent ───────────────────────────────────────

    def _build_instruction(self, container: IoCContainer) -> str:
        system_prompter = container.get(SystemPrompter)
        if system_prompter is None:
            instructions = []
        else:
            instructions = [system_prompter.instruction()]
        instructions.append(self.soul_content)
        return "\n".join(instructions)

    def build_instruction(self, run_context: RunContext[IoCContainer]) -> str:
        return self._build_instruction(run_context.deps)

    def build_agent(self, container: IoCContainer) -> Agent[IoCContainer]:
        """创建 pydantic AI Agent. 不依赖 IoC 容器，可独立单测.

        model 为 None 时走 AnthropicModel + 环境变量.
        """
        if workspace := container.get(Workspace):
            self._load_soul(workspace.root_path())
        model = self._model
        if model is None:
            model_name = os.environ.get("ANTHROPIC_MODEL")
            if not model_name:
                raise RuntimeError(
                    "ANTHROPIC_MODEL env var not set. "
                    "Set it or pass model= explicitly."
                )
            model = AnthropicModel(
                model_name=model_name,
                provider=self._provider or AnthropicProvider(),
                # disable extended thinking by default; enable via model= param if needed
                settings=AnthropicModelSettings(
                    anthropic_thinking=BetaThinkingConfigDisabledParam(type="disabled"),
                ),
            )

        agent = Agent[IoCContainer](
            name=self._name,
            description=self._description,
            # allow  Callable[[Deps], str] as default
            instructions=self.build_instruction,
            model=model,
        )
        if self._on_agent_build is not None:
            self._on_agent_build(agent)
        return agent

    # ── factory ─────────────────────────────────────

    def factory(self, container: IoCContainer) -> "Atom":
        """完整工厂：加载 soul + build_agent + 实例化 Atom runtime."""
        from ._runtime import Atom

        agent = self.build_agent(container)
        return Atom(meta=self, agent=agent, container=container)
