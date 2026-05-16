from ghoshell_moss.contracts.system_prompter import BaseSystemPrompter
from ghoshell_moss.core.blueprint.host import MossSystemPrompter

__all__ = ["MossSystemPrompterImpl"]


class MossSystemPrompterImpl(BaseSystemPrompter, MossSystemPrompter):
    """MOSS 约定的 SystemPrompter 默认实现.

    BaseSystemPrompter 提供 tree 存储 + instruction 组装.
    MossSystemPrompter 提供四个命名访问器 (ctml/project/mode/static).
    二者通过钻石继承组合, 注册为 SystemPrompter 和 MossSystemPrompter 两个 IoC key.
    """
    pass
