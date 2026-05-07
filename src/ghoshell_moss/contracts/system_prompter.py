from typing import Callable, Iterable
from abc import ABC, abstractmethod

__all__ = ['BaseSystemPrompter', 'SystemPrompter']


class SystemPrompter(ABC):
    """
    系统提示词组件.

    Moss 架构中运行的智能体, 其 Instruction 部分由若干组件构成.
    这些组件可分形, 或线性地组织出系统提示词. 它不做分级标题, 只做线性排序. 所以每个 prompter 应该都有一级标题.

    在 MOSS 架构中典型的例子是:
    - Moss Meta Instruction: 基于环境发现构建出来的 prompt. 分为
        - ctml version: 基于 ctml version 从环境中拼合的 prompt.
        - moss root instruction: 在 workspace 根目录定义的 MOSS.md 提供的 instruction. 整个环境复用.
        - moss mode instruction: 在某个特定模式下定义的 instruction. 只对模式生效.
    - Ghost instruction: 基于 Ghost 定义的 instruction.
        - soul
        - existence
        - purpose
        - alignment
    - Moss Static: 所有可运行组件的静态讯息.
    将这个模块拆分出来, 可以方便整个系统在运行时的不同位置组装 system prompt.
    环境中的 System Prompter 应该以 IoC 容器中注册的为基准. 通常就是 Matrix 所持有的.
    """

    @abstractmethod
    def instruction(self) -> str:
        pass

    @abstractmethod
    def is_dynamic(self) -> bool:
        pass

    @abstractmethod
    def with_prompter(self, key: str, prompter: Callable[[], str] | str) -> None:
        pass


class BaseSystemPrompter(SystemPrompter):
    """System Prompter 基础实现."""

    def __init__(
            self,
            *,
            own_instruction: str = '',
            slots: Iterable[str] | None = None,
            prompters: dict[str, Callable[[], str] | str] | None = None,
    ):
        self._own_instruction: str = own_instruction
        self._prompters: dict[str, str | Callable[[], str]] = prompters or {}
        self._slots: set[str] = set(slots) if slots is not None else set()
        self._dynamic: bool = False
        self._cached_instruction: str | None = None

    def is_dynamic(self) -> bool:
        return self._dynamic

    def instruction(self) -> str:
        if self._dynamic:
            return self._instruction()
        if self._cached_instruction is None:
            self._cached_instruction = self._instruction()
        return self._cached_instruction

    def _instruction(self) -> str:
        if self._own_instruction:
            values = [self._own_instruction]
        else:
            values = []
        if self._slots:
            prompters = []
            for key in self._slots:
                prompter = self._prompters.get(key, None)
                if prompter:
                    prompters.append(prompter)
        else:
            prompters = list(self._prompters.values())
        # 可能需要动态.
        for prompter in prompters:
            if isinstance(prompter, str):
                values.append(prompter)
            elif callable(prompter):
                values.append(str(prompter()))
        return "\n\n".join([v for v in values if v])

    def with_prompter(self, key: str, prompter: Callable[[], str] | str) -> None:
        if not isinstance(prompter, str):
            if not callable(prompter):
                raise TypeError(f"prompter must be string or func()->str, `{prompter}` given.")
            value = prompter()
            if not isinstance(value, str):
                raise TypeError(f"prompter must be string or func()->str, `{prompter}` returns invalid.")
            self._dynamic = True
        elif not prompter:
            # 为空直接忽略.
            return None

        if self._slots and key not in self._slots:
            raise KeyError(f"key {key} not in slots.")
        self._prompters[key] = prompter
        return None

    def __copy__(self):
        return BaseSystemPrompter(own_instruction=self._own_instruction, slots=self._slots, prompters=self._prompters)
