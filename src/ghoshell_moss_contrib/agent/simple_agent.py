import asyncio
import json
import logging
import os
import time
from typing import Any, ClassVar, Optional

from ghoshell_common.contracts import LoggerItf, Workspace, workspace_providers
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_container import Container, IoCContainer
from pydantic import BaseModel, Field

from ghoshell_moss.core.concepts.shell import MOSSShell, Speech
from ghoshell_moss.core.shell import new_ctml_shell
from ghoshell_moss.message.adapters.openai_adapter import parse_messages_to_params
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_moss_contrib.agent.chat.console import ConsoleChat
from ghoshell_moss_contrib.agent.depends import check_agent

if check_agent():
    import litellm


class ModelConf(BaseModel):
    default_env: ClassVar[dict[str, None | str]] = {
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
    kwargs: dict[str, Any] = Field(default_factory=dict, description="kwargs")
    top_p: Optional[float] = Field(
        None,
        description="""
An alternative to sampling with temperature, called nucleus sampling, where the
model considers the results of the tokens with top_p probability mass. So 0.1
means only the tokens comprising the top 10% probability mass are considered.
""",
    )

    def generate_litellm_params(self) -> dict[str, Any]:
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
        talker: Optional[str] = None,
        model: Optional[ModelConf] = None,
        container: Optional[IoCContainer] = None,
        shell: Optional[MOSSShell] = None,
        speech: Optional[Speech] = None,
        chat: Optional[BaseChat] = None,
    ):
        self.container = Container(name="agent", parent=container)
        for provider in workspace_providers():
            if self.container.bound(provider.contract()):
                continue
            self.container.register(provider)

        self.chat: BaseChat = chat or ConsoleChat()
        self.talker = talker
        shell = shell or new_ctml_shell(container=self.container, speech=speech)
        model = model or ModelConf()
        self.instruction = instruction
        self.shell = shell
        if speech is not None:
            self.shell.with_speech(speech)
        self.model = model

        _ws = self.container.get(Workspace)
        self._message_filename = f"message_{int(time.time())}.json"
        if _ws:
            self._history_storage = _ws.runtime().sub_storage("agent_history")
        else:
            self._history_storage = MemoryStorage("agent_history")

        self._response_done = asyncio.Event()
        self._started = False
        self._closed_event = asyncio.Event()
        self._error: Optional[Exception] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: asyncio.Queue[list[dict] | None] | None = None
        self._logger: Optional[LoggerItf] = None
        self._main_loop_task: Optional[asyncio.Task] = None

        # 打断优化
        self._interrupt_requested = False
        self._response_cancellation_lock = asyncio.Lock()

    def interrupt(self):
        """优化中断方法"""
        self._interrupt_requested = True

        # 如果有循环，通知中断
        if self._loop:
            # 尝试取消当前响应任务
            asyncio.run_coroutine_threadsafe(self._cancel_current_response(), self._loop)

    async def _cancel_current_response(self):
        """取消当前响应"""
        async with self._response_cancellation_lock:
            if hasattr(self, "_current_responding_task") and self._current_responding_task:
                if not self._current_responding_task.done():
                    self.logger.info("Cancelling current response...")
                    self._current_responding_task.cancel()

                    # 等待一小段时间让任务处理取消
                    try:
                        await asyncio.wait_for(self._current_responding_task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    finally:
                        self._current_responding_task = None
                        self._interrupt_requested = False

    def handle_user_input(self, text: str) -> None:
        try:
            self.raise_error()
            if not text:
                self.logger.info("agent received an empty input")
                self._loop.call_soon_threadsafe(self._input_queue.put_nowait, None)
                return

            self._loop.call_soon_threadsafe(
                self._input_queue.put_nowait, [{"role": "user", "content": text, "name": self.talker}]
            )
        except Exception as e:
            self.chat.print_exception(e)

    async def _main_loop(self) -> None:
        """优化主循环"""
        responding = None
        while not self._closed_event.is_set():
            try:
                # 检查中断请求
                if self._interrupt_requested:
                    self.logger.info("Interrupt detected, skipping input")
                    await asyncio.sleep(0.1)  # 短暂等待
                    self._interrupt_requested = False
                    continue

                new_input = await self._input_queue.get()
                self.logger.info("Received new input: %s", new_input)

                # 如果有正在进行的响应，先取消
                if responding is not None and not responding.done():
                    self.logger.info("Cancelling previous response for new input")
                    responding.cancel()
                    try:
                        await responding
                    except asyncio.CancelledError:
                        self.logger.info("Previous response cancelled")
                    finally:
                        responding = None

                # 处理新输入
                if new_input is not None:
                    # 存储当前响应任务引用
                    responding = asyncio.create_task(self._response_loop(new_input))
                    self._current_responding_task = responding

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.chat.print_exception(e)
            finally:
                # 清理
                if responding is not None and responding.done():
                    responding = None

    async def wait_done(self):
        await self._closed_event.wait()
        if self._error is not None:
            raise RuntimeError(f"agent failed: {self._error}")

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self.container.get(LoggerItf) or logging.getLogger("SimpleAgent")
        return self._logger

    def raise_error(self):
        if self._error is not None:
            raise RuntimeError(self._error)

    async def _response_loop(self, inputs: list[dict]) -> None:
        try:
            if not inputs:
                return
            while inputs is not None and not self._interrupt_requested:
                inputs = await asyncio.create_task(self._single_response(inputs))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("Response loop failed")
            self.chat.print_exception(e)

    def _get_history(self) -> list[dict]:
        if not self._history_storage.exists(self._message_filename):
            return []
        history = self._history_storage.get(self._message_filename)
        return json.loads(history)

    def _put_history(self, messages: list[dict]) -> None:
        messages_str = json.dumps(messages, indent=4, ensure_ascii=False)
        self._history_storage.put(self._message_filename, messages_str.encode("utf-8"))

    async def _single_response(self, inputs: list[dict]) -> Optional[list[dict]]:
        """
        临时实现的回复模式.
        todo: 有挺多问题的, 核心问题是 shell interpreter 的生命周期没设计完.
            计划中除了支持全双工交互外, 还需要支持传统的 react 模式.
            这其中又要为上下文 token 裁剪设计一个简洁的办法. 目前 interpreter 还没有完工, 所以临时使用这种方式.
        """
        self.logger.info("Single response received, inputs=%s", inputs)
        generated = ""
        execution_results = ""

        history = self._get_history()
        try:
            self.chat.start_ai_response()
            self._response_done.clear()
            params = self.model.generate_litellm_params()
            async with self.shell.interpreter_in_ctx() as interpreter:
                reasoning = False

                moss_instruction = interpreter.moss_instruction()
                # 系统指令.
                messages = []
                if moss_instruction:
                    messages.append({"role": "system", "content": moss_instruction})
                # 注册 agent 的 instruction.
                messages.append({"role": "system", "content": self.instruction})

                # 增加历史.
                messages.extend(history)
                # 增加 context
                context = interpreter.context_messages()
                if len(context) > 0:
                    parsed = parse_messages_to_params(context)
                    messages.extend(parsed)
                # 增加 inputs
                if inputs:
                    messages.extend(inputs)
                params["messages"] = messages
                params["stream"] = True
                response_stream = await litellm.acompletion(**params)
                async for chunk in response_stream:
                    delta = chunk.choices[0].delta
                    self.logger.debug("delta: %s", delta)
                    if "reasoning_content" in delta:
                        if not reasoning:
                            reasoning = True
                        self.chat.update_ai_response(delta.reasoning_content, is_thinking=True)
                        continue
                    elif reasoning:
                        self.chat.start_ai_response()
                        reasoning = False
                    content = delta.content
                    if not content:
                        continue
                    self.chat.update_ai_response(content)
                    generated += content

                    interpreter.feed(content)
                interpreter.commit()
                results = await asyncio.create_task(interpreter.results())
                generated = interpreter.executed_tokens()
                if len(results) > 0:
                    execution_results = "\n---\n".join([f"{tokens}:\n{result}" for tokens, result in results.items()])
                    self.logger.info("execution_results=%s", results)
                    return []
                else:
                    return None
        finally:
            self._response_done.set()
            self.chat.finalize_ai_response()

            history.extend(inputs)
            if generated:
                history.append({"role": "assistant", "content": generated})
            if execution_results:
                history.append({"role": "system", "content": f"Commands Outputs:\n ```\n{execution_results}\n```"})
            if self._interrupt_requested:
                history.append({"role": "system", "content": "Attention: User interrupted your response last time."})
            self._put_history(history)

    async def run(self):
        async with self:
            self.chat.set_input_callback(self.handle_user_input)
            self.chat.set_interrupt_callback(self.interrupt)
            await self.chat.run()
        await self.wait_done()

    async def start(self):
        if self._started:
            self.logger.info("SimpleAgent already started")
            return
        self.logger.info("SimpleAgent starting")
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._input_queue = asyncio.Queue()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self.container.bootstrap()
        await self.shell.start()
        self.logger.info("SimpleAgent started")

    async def close(self):
        if self._closed_event.is_set():
            self.logger.info("SimpleAgent already closed")
            return
        self.logger.info("SimpleAgent closing")
        if self._main_loop_task is not None:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._main_loop_task = None

        await self.shell.close()
        self.container.shutdown()
        self._closed_event.set()
        self.logger.info("SimpleAgent closed")

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
