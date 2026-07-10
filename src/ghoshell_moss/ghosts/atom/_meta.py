import os
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from anthropic.types.beta import BetaThinkingConfigDisabledParam
from ghoshell_container import IoCContainer
from ghoshell_moss.core.blueprint.ghost import GhostMeta, GhostWorkspace
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta
from ghoshell_moss.contracts import SystemPrompter
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

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
            provider: Any = None,
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

    def _load_soul(self, ghost_workspace: GhostWorkspace) -> None:
        """从 ghost_workspace.home 加载 soul 文件. soul_content 非 None 时跳过.

        ghost_workspace.home = matrix.ghost_home = workspace/ghosts/{ghost_name}/
        由 Host.run_ghost() 在 self.run() 之前通过 env.set_ghost_name() 确保 ghost_name 已设置.
        """
        if self._soul_content is not None:
            return

        file_path = None
        filename = ''
        if isinstance(self._soul_path, str):
            filename = self._soul_path
        elif isinstance(self._soul_path, Path):
            pass
        elif self._soul_path is None:
            filename = 'soul.md'

        if file_path is None and filename:
            file_path = ghost_workspace.home.joinpath(filename)

        if file_path and file_path.exists():
            self._soul_content = file_path.read_text(encoding="utf-8")

    # ── agent ───────────────────────────────────────

    def build_instruction_from_ioc(self, container: IoCContainer) -> str:
        moss_system_prompter = container.get(SystemPrompter)
        if moss_system_prompter is None:
            instructions = []
        else:
            # SystemPrompter.instruction() = CTML + project + mode + static 四层.
            # soul 不在这四层中——它由 Ghost 自行在末尾追加, 是 Ghost 的人格层.
            instructions = [moss_system_prompter.instruction()]
        instructions.append(self.soul_content)
        return "\n".join(instructions)

    def build_instruction(self, run_context: RunContext[IoCContainer]) -> str:
        return self.build_instruction_from_ioc(run_context.deps)

    def build_agent(self, container: IoCContainer) -> Agent[IoCContainer]:
        """创建 pydantic AI Agent. 不依赖 IoC 容器，可独立单测.

        model 为 None 时走 AnthropicModel + 环境变量.
        """
        ghost_workspace = container.get(GhostWorkspace)
        if ghost_workspace is not None:
            self._load_soul(ghost_workspace)
        model = self._model
        if model is None:
            # Atom 是 MOSS 的最小 Ghost 原型，默认 provider 保持历史兼容：
            # 没有显式配置 MOSS_LLM_PROVIDER 时仍走 Anthropic。OpenAI-compatible
            # provider 支持保留给 aEther/其它 mode 通过环境变量主动启用。
            llm_provider = os.environ.get("MOSS_LLM_PROVIDER", "anthropic").lower()
            if llm_provider == "anthropic":
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
                        timeout=120.0,
                    ),
                )
            else:
                # OpenAI 兼容 (DeepSeek / 通义 / Moonshot OpenAI 等)
                model_name = os.environ.get("OPENAI_MODEL")
                if not model_name:
                    raise RuntimeError(
                        "OPENAI_MODEL env var not set. "
                        "Set it or pass model= explicitly."
                    )
                base_url = os.environ.get("OPENAI_BASE_URL")
                api_key = os.environ.get("OPENAI_API_KEY")
                if not base_url or not api_key:
                    raise RuntimeError(
                        "OPENAI_BASE_URL / OPENAI_API_KEY env var not set."
                    )
                model_settings = OpenAIModelSettings(timeout=60.0)
                if os.environ.get("MOSS_OPENAI_DISABLE_THINKING") == "1":
                    # 部分 OpenAI-compatible 服务（例如某些 DeepSeek 兼容端点）
                    # 支持在 extra_body 中关闭 thinking，以换取语音场景低延迟。
                    # 这不是 OpenAI 兼容协议的通用能力，必须由 mode/env 显式启用。
                    model_settings["extra_body"] = {"thinking": {"type": "disabled"}}
                    model_settings["openai_continuous_usage_stats"] = False
                model = OpenAIModel(
                    model_name=model_name,
                    provider=OpenAIProvider(base_url=base_url, api_key=api_key),
                    settings=model_settings,
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
