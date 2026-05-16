from typing import Callable, Iterable
from abc import ABC, abstractmethod

__all__ = ['SystemPrompter', 'BaseSystemPrompter', ]


class SystemPrompter(ABC):
    """系统提示词组件 — tree node.

    Moss 架构中运行的智能体, 其 Instruction 部分由若干组件构成.
    每个节点拥有自己的 instruction, 并可以包含子节点.
    children() 按插入顺序暴露树的层次结构, 使得外部可以遍历、在层间插入、替换.

    MOSS 架构中典型的层次:
    - ctml: CTML 语法版本 prompt
    - project: workspace 根目录 MOSS.md 声明的 project instruction
    - mode: 当前模式定义的 instruction
    - static: 所有可运行组件的静态信息 (lazy, 依赖 shell 启动)
    - ghost: Ghost 的存在/目的/对齐 (soul, existence, purpose, alignment)
    """

    @abstractmethod
    def instruction(self) -> str:
        """本节点完整 instruction — own + children 组装."""
        pass

    def description(self) -> str:
        """本节点的自解释描述. 构建时传入, flatten 时暴露."""
        return ""

    def children(self) -> dict[str, 'SystemPrompter']:
        """子节点映射, 按插入顺序. key 是语义名, value 是子 SystemPrompter."""
        return {}

    def child(self, key: str) -> 'SystemPrompter | None':
        """按 key 取子节点."""
        return self.children().get(key)

    def child_instruction(self, key: str) -> str:
        """按 key 取子节点的 instruction, 不存在返回空字符串."""
        c = self.child(key)
        return c.instruction() if c else ""

    def flatten(self, separator: str = "/") -> list[tuple[str, str]]:
        """DFS 展平树, 返回 [(路径, 描述), ...].

        每个节点贡献自己的 description, 用于自解释和结构校验.
        """
        result: list[tuple[str, str]] = []
        for key, child in self.children().items():
            child_desc = child.description()
            result.append((key, child_desc))
            for sub_path, sub_desc in child.flatten(separator):
                result.append((f"{key}{separator}{sub_path}", sub_desc))
        return result

    def linear(self, slots: list[str]) -> str:
        """按给定顺序拼装子节点 instruction, 跳过空.

        典型用法: ghost 定义自己的分层顺序:
            parts = [prompter.linear(["ctml", "project", "mode"]), soul, prompter.linear(["static"])]
        """
        parts = []
        for key in slots:
            if text := self.child_instruction(key):
                parts.append(text)
        return "\n\n".join(parts)

    @abstractmethod
    def is_dynamic(self) -> bool:
        """是否有动态内容 — 影响 instruction() 缓存策略."""
        pass

    @abstractmethod
    def with_prompter(self, key: str, prompter: 'SystemPrompter | Callable[[], str] | str') -> None:
        """添加子节点.

        接受 SystemPrompter (直接作为子节点),
        或 str / callable (自动包装为叶子节点).
        """
        pass


class BaseSystemPrompter(SystemPrompter):
    """SystemPrompter tree 的基础实现.

    每个节点有 own_instruction (自己的内容) 和 children (子树).
    instruction() = own_instruction + children 按序组装.
    """

    def __init__(
            self,
            *,
            own_instruction: str = '',
            description: str = '',
            slots: Iterable[str] | None = None,
            children: dict[str, SystemPrompter] | None = None,
    ):
        self._own_instruction: str = own_instruction
        self._description: str = description
        self._children: dict[str, SystemPrompter] = children or {}
        self._slots: set[str] = set(slots) if slots is not None else set()
        self._dynamic: bool = False
        self._cached_instruction: str | None = None

    # ── SystemPrompter ABC ────────────────────────

    def description(self) -> str:
        return self._description

    def children(self) -> dict[str, SystemPrompter]:
        return self._children

    def is_dynamic(self) -> bool:
        if self._dynamic:
            return True
        return any(child.is_dynamic() for child in self._children.values())

    def instruction(self) -> str:
        if self.is_dynamic():
            return self._instruction()
        if self._cached_instruction is None:
            self._cached_instruction = self._instruction()
        return self._cached_instruction

    def _instruction(self) -> str:
        values = []
        if self._own_instruction:
            values.append(self._own_instruction)
        for child in self._children.values():
            if text := child.instruction():
                values.append(text)
        return "\n\n".join(values)

    def with_prompter(self, key: str, prompter: 'SystemPrompter | Callable[[], str] | str') -> None:
        """添加子节点. 接受 SystemPrompter 或 str/callable (自动包装为叶子)."""
        if isinstance(prompter, SystemPrompter):
            child = prompter
        elif isinstance(prompter, str):
            if not prompter:
                return
            child = _Leaf(prompter)
        elif callable(prompter):
            child = _DynamicLeaf(prompter)
        else:
            raise TypeError(
                f"prompter must be SystemPrompter | str | callable, got {type(prompter)}"
            )

        if self._slots and key not in self._slots:
            raise KeyError(f"key {key!r} not in slots {self._slots!r}")

        self._children[key] = child
        self._cached_instruction = None

    def __copy__(self):
        copy = BaseSystemPrompter(
            own_instruction=self._own_instruction,
            description=self._description,
            slots=self._slots,
            children=dict(self._children),
        )
        copy._dynamic = self._dynamic
        return copy


# ── internal leaf nodes ────────────────────────────


class _Leaf(BaseSystemPrompter):
    """不可变叶子节点 — 固定 instruction 字符串."""

    def __init__(self, text: str):
        super().__init__(own_instruction=text)

    def with_prompter(self, key, prompter):
        raise TypeError("leaf node cannot have children")


class _DynamicLeaf(BaseSystemPrompter):
    """动态叶子节点 — instruction 每次调用时重新计算."""

    def __init__(self, fn: Callable[[], str]):
        super().__init__()
        self._fn = fn

    def instruction(self) -> str:
        return str(self._fn())

    def is_dynamic(self) -> bool:
        return True

    def with_prompter(self, key, prompter):
        raise TypeError("leaf node cannot have children")
