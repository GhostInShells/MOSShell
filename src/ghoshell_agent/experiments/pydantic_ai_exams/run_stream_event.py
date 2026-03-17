import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider


# 假设 Container 已定义
class Container:
    pass


# 注意：deepseek-reasoner 包含“思考过程”，目前部分 Provider 封装可能还在适配其 reasoning_content
model = AnthropicModel(
    'deepseek-reasoner',
    provider=AnthropicProvider()
)

agent = Agent(model, deps_type=Container)
container = Container()


async def run():
    # 1. 启动流式运行
    async for event in agent.run_stream_events("hello", deps=container):
        print("--- 开始接收流式输出 ---")

        # 模式 A: 获取纯文本增量 (最常用)
        # debounce_by=None 确保每个 token 立即输出，降低感知延时
        print(f"Content Block event: {event!r}")

        print("\n--- 流式结束 ---")


if __name__ == "__main__":
    asyncio.run(run())
