from typing import Optional, List, Dict, Any, ClassVar
from ghoshell_moss.concepts.shell import MOSSShell, Output
from ghoshell_moss.shell import new_shell
from ghoshell_moss.depends import check_agent
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer, Container
from ghoshell_moss.agent.console import ChatRenderer, ChatRenderOutput
from ghoshell_common.contracts import workspace_providers
from pydantic import BaseModel, Field

import os
import asyncio

if check_agent():
    import litellm


class ModelConf(BaseModel):
    default_env: ClassVar[Dict[str, None | str]] = {
        "base_url": None,
        "model": "gpt-3.5-turbo",
        "api_key": None,
        "custom_llm_provider": None,
    }

    base_url: Optional[str] = Field(
        default="$MOSS_LLM_BASE_URL",
        description="base url for chat completion",
    )
    model: str = Field(
        default="$MOSS_LLM_MODEL",
        description="llm model name that server provided",
    )
    api_key: Optional[str] = Field(
        default="$MOSS_LLM_API_KEY",
        description="api key",
    )
    custom_llm_provider: Optional[str] = Field(
        default="$MOSS_LLM_PROVIDER",
        description="custom LLM provider name",
    )
    temperature: float = Field(default=0.7, description="temperature")
    n: int = Field(default=1, description="number of iterations")
    max_tokens: int = Field(default=4000, description="max tokens")
    timeout: float = Field(default=30, description="timeout")
    request_timeout: float = Field(default=40, description="request timeout")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="kwargs")
    top_p: Optional[float] = Field(None, description="""
An alternative to sampling with temperature, called nucleus sampling, where the
model considers the results of the tokens with top_p probability mass. So 0.1
means only the tokens comprising the top 10% probability mass are considered.
""")

    def generate_litellm_params(self) -> Dict[str, Any]:
        params = self.model_dump(exclude_none=True, exclude={"kwargs"})
        params.update(self.kwargs)
        real_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                default_value = self.default_env.get(key, "")
                real_value = os.environ.get(value[1:], default_value)
                if real_value is not None:
                    real_params[key] = real_value
            else:
                real_params[key] = value
        return real_params


class SimpleAgent:

    def __init__(
            self,
            instruction: str,
            *,
            model: Optional[ModelConf] = None,
            container: Optional[IoCContainer] = None,
            shell: Optional[MOSSShell] = None,
            output: Optional[Output] = None,
            messages: Optional[List[Dict]] = None,
    ):
        self.container = Container(name="agent", parent=container)
        self.container.register(*workspace_providers())
        self.chat = ChatRenderer()
        shell = shell or new_shell(container=self.container, output=output)
        model = model or ModelConf()
        self.instruction = instruction
        self.shell = shell
        self.model = model
        self.messages: List[Dict] = messages or []
        self._inputs = []
        self._response_task: Optional[asyncio.Task] = None
        self._response_done = asyncio.Event()
        self._started = False
        self._closed_event = asyncio.Event()
        self._error: Optional[Exception] = None

    async def handle_user_input(self, text: str) -> None:
        try:
            self.chat.add_user_message(text)
            await self.response([
                {"role": "user", "content": text}
            ])
            self.raise_error()
        except Exception as e:
            self.chat.print_exception(e)
            self.chat.app.exit()

    async def wait_done(self):
        await self._closed_event.wait()
        if self._error is not None:
            raise RuntimeError(f"agent failed: {self._error}")

    @property
    def logger(self) -> LoggerItf:
        return self.container.force_fetch(LoggerItf)

    def raise_error(self):
        if self._error is not None:
            raise RuntimeError(self._error)

    async def response(self, inputs: List[Dict]) -> None:
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
            await self._response_task
        if self._error is not None:
            raise RuntimeError(f"agent failed: {self._error}")

        task = asyncio.create_task(self._response(inputs))
        self._response_task = task

    async def _response(self, inputs: List[Dict]) -> None:
        if len(inputs) == 0:
            return

        generated = ""
        execution_data = []
        try:
            self._response_done.clear()
            params = self.model.generate_litellm_params()
            messages = [
                {"role": "system", "content": self.instruction}
            ]
            # 增加历史.
            messages.extend(self.messages.copy())
            # 增加 inputs
            messages.extend(inputs)

            params['messages'] = messages
            params['stream'] = True
            response_stream = await litellm.acompletion(**params)
            # interpreter = self.shell.interpreter()
            # async with interpreter:
            self.chat.start_ai_response()
            reasoning = False
            async for chunk in response_stream:
                delta = chunk.choices[0].delta
                if "reasoning_content" in delta:
                    if not reasoning:
                        reasoning = True
                    self.chat.update_ai_response(delta.reasoning_content, is_gray=True)
                    continue
                elif reasoning:
                    self.chat.start_ai_response()
                    reasoning = False
                content = delta.content
                if not content:
                    continue
                self.chat.update_ai_response(content)
                generated += content
                # interpreter.feed(content)
                # interpreter.commit()
                # tasks = await interpreter.wait_execution_done()
                # for task in tasks.values():
                #     if task.success():
                #         result = task.result()
                #         if result is not None:
                #             execution_data.append(
                #                 f"{task.tokens}:\n```\n{task.result()}\n```"
                #             )
        except asyncio.CancelledError:
            pass
        finally:
            self._response_done.set()
            if generated:
                self.messages.extend(inputs)
                self.messages.append({"role": "assistant", "content": generated})
                execution = "\n\n".join(execution_data)
                self.messages.append({"role": "system", "content": "## executions:\n\n%s" % execution})

    async def run(self):
        async with self:
            self.chat.set_input_callback(self.handle_user_input)
            await self.chat.run()

    async def start(self):
        if self._started:
            return
        self._started = True
        self.container.bootstrap()
        await self.shell.start()

    async def close(self):
        if self._closed_event.is_set():
            return
        if self._response_task:
            self._response_task.cancel()
            await self._response_done.wait()
        await self.shell.close()
        self._closed_event.set()
        self.container.shutdown()

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
