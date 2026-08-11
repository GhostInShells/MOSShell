"""ProjectLLMFuncsProvider — project baseline default for LLMFuncs.

语义 = per-project singleton. 惰性 import pydantic-ai: fetch 时才构造
PydanticAIFuncs, 模块级 import 无副作用。无 ghost extra 时 fetch 报干净的
ImportError, 不拖垮 project 容器。

workspace 用户在 ProjectManifest.providers 里显式覆写即可覆盖 default。
"""

from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.llms import LLMFuncs

__all__ = ["ProjectLLMFuncsProvider"]


class ProjectLLMFuncsProvider(Provider[LLMFuncs]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[LLMFuncs]:
        return LLMFuncs

    def factory(self, con: IoCContainer) -> LLMFuncs:
        from ghoshell_moss.llms.pydantic_ai_adapter.funcs import PydanticAIFuncs
        return PydanticAIFuncs()
