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
    async with agent.run_stream("hello", deps=container) as result:
        print("--- 开始接收流式输出 ---")

        # 模式 A: 获取纯文本增量 (最常用)
        # debounce_by=None 确保每个 token 立即输出，降低感知延时
        async for text_delta in result.stream_text(debounce_by=None):
            print(f"Content Block Delta: {text_delta!r}")

        print("\n--- 流式结束 ---")

        # 2. 检查最终结果和消耗
        print(f"Usage: {result.usage()}")
        # 注意：在流结束后才能访问最终的 result.data 或 result.response
        print(f"Final Response: {result.all_messages()}")


if __name__ == "__main__":
    asyncio.run(run())
