from ghoshell_moss.core.concepts.channel import Channel, MutableChannel


class WorkflowChannel(MutableChannel):
    """
    一种特殊的 Channel, 它有两种模式:
    1. router 模式: 暴露子 Channel 给人直接使用. 也包含它自身创建的 Command.
    2. developer 模式: 基于子 Channel 上下文, 可以进行开发, 创建新的 command. 并且将编译的结果保存到本地. 未来可复用.
    """
    pass
