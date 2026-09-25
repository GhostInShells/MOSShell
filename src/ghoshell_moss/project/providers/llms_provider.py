"""ProjectLLMFuncsProvider — project baseline default for LLMFuncs.

语义 = per-project singleton. 惰性 import pydantic-ai: fetch 时才构造
PydanticAIFuncs, 模块级 import 无副作用。无 ghost extra 时 fetch 报干净的
ImportError, 不拖垮 project 容器。

workspace 用户在 ProjectManifest.providers 里显式覆写即可覆盖 default。
"""

from typing import Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.llms import LLMConfig, LLMFuncs
from ghoshell_moss.contracts.logger import LoggerItf

__all__ = ["ProjectLLMFuncsProvider"]


class ProjectLLMFuncsProvider(Provider[LLMFuncs]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[LLMFuncs]:
        return LLMFuncs

    def factory(self, con: IoCContainer) -> LLMFuncs:
        from ghoshell_moss.llms.pydantic_ai_adapter.funcs import PydanticAIFuncs
        store = con.get(ConfigStore)
        config = store.get_or_create(LLMConfig()) if store is not None else LLMConfig()
        logger = con.get(LoggerItf)
        # container 供 convert() 的 converter 适配拉 IoC 依赖; 单例, 长期持有无碍。
        return PydanticAIFuncs(config=config, logger=logger, container=con)
