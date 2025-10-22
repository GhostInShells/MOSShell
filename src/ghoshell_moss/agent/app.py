from typing import Optional, List, Dict
from ghoshell_moss.agent.simple_agent import SimpleAgent, ModelConf
from ghoshell_moss.agent.console import ChatRenderer, ChatRenderOutput
from ghoshell_moss.concepts.shell import MOSSShell, Output
from ghoshell_moss.shell import new_shell
from ghoshell_container import IoCContainer, Container
from ghoshell_common.contracts import workspace_providers
import asyncio


class ConsoleAgentApp:

    def __init__(
            self,
            *,
            instruction: str,
            model: Optional[ModelConf] = None,
            container: Optional[IoCContainer] = None,
            shell: Optional[MOSSShell] = None,
            messages: Optional[List[Dict]] = None,
    ):
        if container is None:
            container = Container(name="ConsoleAgentApp")
            container.register(*workspace_providers())

        self.chat = ChatRenderer()
        self.output = ChatRenderOutput(self.chat)
        self.agent = SimpleAgent(
            instruction=instruction,
            messages=messages,
            model=model,
            container=container,
            shell=shell,
            output=self.output,
        )

    async def handle_user_input(self, text: str) -> None:
        try:
            await self.agent.response([
                {"role": "user", "content": text}
            ])
            self.agent.raise_error()
        except Exception as e:
            self.chat.print_exception(e)
            self.chat.app.exit()

    async def run(self):
        await self.agent.start()
        try:
            self.chat.set_input_callback(self.handle_user_input)
            await self.chat.run()
        except Exception as e:
            self.chat.print_exception(e)
        finally:
            self.chat.start_ai_response()
            self.chat.update_ai_response("done")
            await self.agent.close()


if __name__ == "__main__":
    app = ConsoleAgentApp(
        instruction="你是 JoJo",
    )
    asyncio.run(app.run())
