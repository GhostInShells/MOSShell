from ghoshell_moss.core.concepts.command import PyCommand

__all__ = ['thinking', 'thinking_command']


async def thinking(chunks__):
    # thinking 原语默认挂载在 Shell 上, 而且不对外展示 (visible 为 False
    # 它用来兼容 当前 (2026年) 仍然有主流大模型仍然在用 thinking xml 做思考的流式标记.
    # 实际上大模型全部都应该用 CTML 的思路去实现流式全双工交互才对.
    # 这个函数用于兼容. 不用对模型展示.
    # 替换这个原语实现 (在 Shell 上重载), 可以定义自己的比如 topic 等.
    async for chunk in chunks__:
        pass


thinking_command = PyCommand(
    thinking,
    # 对模型不可见.
    visible=False,
    blocking=True,
)
