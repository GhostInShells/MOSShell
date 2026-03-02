import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterable
from typing import Any, ClassVar, Optional

from ghoshell_common.contracts import LoggerItf, Workspace, workspace_providers
from ghoshell_common.contracts.storage import MemoryStorage, Storage
from ghoshell_container import Container, IoCContainer
from pydantic import BaseModel, Field

from ghoshell_moss import Message
from ghoshell_moss.core import MOSSShell, Speech, new_ctml_shell
from ghoshell_moss.core.llm.shell_driver import ShellTurnRequest, ShellTurnResult, with_shell
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_moss_contrib.agent.chat.console import ConsoleChat
from ghoshell_moss_contrib.agent.shell_litellm import litellm_text_streamer


class ModelConf(BaseModel):
    """LiteLLM parameters with env expansion.

    Copied from `simple_agent.py` to avoid coupling to its implementation.
    """

    default_env: ClassVar[dict[str, None | str]] = {
        "base_url": None,
        "model": "gpt-3.5-turbo",
        "api_key": None,
        "custom_llm_provider": None,
    }

    base_url: Optional[str] = Field(default="$MOSS_LLM_BASE_URL")
    model: str = Field(default="$MOSS_LLM_MODEL")
    api_key: Optional[str] = Field(default="$MOSS_LLM_API_KEY")
    custom_llm_provider: Optional[str] = Field(default="$MOSS_LLM_PROVIDER")
    temperature: float = Field(default=0.7)
    n: int = Field(default=1)
    max_tokens: int = Field(default=4000)
    timeout: float = Field(default=30)
    request_timeout: float = Field(default=40)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    top_p: Optional[float] = None

    def generate_litellm_params(self) -> dict[str, Any]:
        params = self.model_dump(exclude_none=True, exclude={"kwargs"})
        params.update(self.kwargs)
        real_params: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                default_value = self.default_env.get(key, "")
                real_value = os.environ.get(value[1:], default_value)
                if real_value is not None:
                    real_params[key] = real_value
            else:
                real_params[key] = value
        return real_params


class SimpleAgentV2:
    """Turn-based agent built on `with_shell(...)`.

    Compared to `SimpleAgent`, the interpreter/LLM glue is delegated to
    `ghoshell_moss.core.llm.shell_driver`.
    """

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
        history_storage: Storage | None = None,
        react_max_turns: int = 5,
    ):
        self.container = Container(name="agent", parent=container)
        for provider in workspace_providers():
            if self.container.bound(provider.contract()):
                continue
            self.container.register(provider)

        self.chat: BaseChat = chat or ConsoleChat()
        self.talker = talker
        self.model = model or ModelConf()
        self.instruction = instruction

        self.shell = shell or new_ctml_shell(container=self.container, speech=speech)
        if speech is not None:
            self.shell.with_speech(speech)

        _ws = self.container.get(Workspace)
        self._message_filename = f"moss_message_{int(time.time())}.json"
        if history_storage is not None:
            self._history_storage = history_storage
        elif _ws:
            self._history_storage = _ws.runtime().sub_storage("agent_history")
        else:
            self._history_storage = MemoryStorage("agent_history")

        self._logger: Optional[LoggerItf] = None
        self._started = False
        self._closed_event = asyncio.Event()
        self._error: Optional[Exception] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: asyncio.Queue[str | None] | None = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._current_turn_task: Optional[asyncio.Task] = None
        self._response_cancellation_lock = asyncio.Lock()
        self._interrupt_requested = False

        if react_max_turns < 1:
            raise ValueError("react_max_turns must be >= 1")
        self._react_max_turns = react_max_turns

        # LiteLLM streamer + shell runner
        self._base_streamer = litellm_text_streamer()
        default_llm_kwargs = self.model.generate_litellm_params()
        self._runner = with_shell(
            self.shell,
            llm=self._ui_streamer,
            default_llm_kwargs=default_llm_kwargs,
        )

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self.container.get(LoggerItf) or logging.getLogger("SimpleAgentV2")
        return self._logger

    def raise_error(self):
        if self._error is not None:
            raise RuntimeError(self._error)

    def interrupt(self) -> None:
        """Interrupt the current running turn (best-effort)."""

        self._interrupt_requested = True
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._cancel_current_turn(), self._loop)

    async def _cancel_current_turn(self) -> None:
        async with self._response_cancellation_lock:
            if self._current_turn_task is None or self._current_turn_task.done():
                return
            self._current_turn_task.cancel()
            try:
                await asyncio.wait_for(self._current_turn_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            finally:
                self._current_turn_task = None

    def _load_history(self) -> list[Message]:
        if not self._history_storage.exists(self._message_filename):
            return []
        raw = self._history_storage.get(self._message_filename)
        if not raw:
            return []
        payload = json.loads(raw)
        if not isinstance(payload, list):
            return []
        return [Message(**item) for item in payload]

    def _save_history(self, messages: list[Message]) -> None:
        payload = [m.dump() for m in messages]
        self._history_storage.put(
            self._message_filename,
            json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
        )

    def _agent_system_messages(self) -> list[Message]:
        return [Message.new(role="system").with_content(self.instruction)]

    def _ui_streamer(self, messages: list[Message], /, **kwargs: Any) -> AsyncIterable[str]:
        stream = self._base_streamer(messages, **kwargs)

        async def _iter() -> AsyncIterable[str]:
            async for delta in stream:
                self.chat.update_ai_response(delta)
                yield delta

        return _iter()

    async def single_turn(self, user_text: str) -> ShellTurnResult:
        """Run one user turn, including ReAct follow-up turns.

        If the shell execution yields observation messages (i.e.
        `interpretation.observe == True`), we automatically run additional LLM turns
        so the model can read tool outputs and continue.
        """

        self.raise_error()
        self._interrupt_requested = False

        persisted = self._load_history()
        user_msg = Message.new(role="user", name=self.talker).with_content(user_text)
        last_result: ShellTurnResult | None = None

        for i in range(self._react_max_turns):
            inputs: list[Message]
            if i == 0:
                inputs = [user_msg]
            else:
                inputs = []

            history = self._agent_system_messages() + persisted
            request = ShellTurnRequest(inputs=inputs, history=history)

            self.chat.start_ai_response()
            try:
                last_result = await self._runner(request)
            finally:
                self.chat.finalize_ai_response()

            if inputs:
                persisted.extend(inputs)
            if last_result.assistant_text:
                persisted.append(Message.new(role="assistant").with_content(last_result.assistant_text))
            if last_result.interpretation.messages:
                persisted.extend(last_result.interpretation.messages)

            if self._interrupt_requested:
                break

            should_continue = bool(last_result.interpretation.observe or last_result.interpretation.messages)
            if not should_continue:
                break

        self._save_history(persisted)
        assert last_result is not None
        return last_result

    # --- interactive runner ---

    def handle_user_input(self, text: str) -> None:
        try:
            self.raise_error()
            if self._loop is None or self._input_queue is None:
                return
            self._loop.call_soon_threadsafe(self._input_queue.put_nowait, text)
        except Exception as e:
            self.chat.print_exception(e)

    async def _main_loop(self) -> None:
        assert self._input_queue is not None
        while not self._closed_event.is_set():
            try:
                text = await self._input_queue.get()
                if text is None:
                    continue

                if self._current_turn_task is not None and not self._current_turn_task.done():
                    await self._cancel_current_turn()

                self._current_turn_task = asyncio.create_task(self.single_turn(text))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Agent loop failed")
                self.chat.print_exception(e)

    async def run(self):
        async with self:
            self.chat.set_input_callback(self.handle_user_input)
            self.chat.set_interrupt_callback(self.interrupt)
            await self.chat.run()

    async def start(self):
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._input_queue = asyncio.Queue()
        self._main_loop_task = asyncio.create_task(self._main_loop())
        self.container.bootstrap()
        await self.shell.start()

    async def close(self):
        if self._closed_event.is_set():
            return
        if self._main_loop_task is not None:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
            self._main_loop_task = None
        await self.shell.close()
        self.container.shutdown()
        self._closed_event.set()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._error = exc_val
        await self.close()


# for testing
if __name__ == "__main__":
    import pathlib

    from ghoshell_moss_contrib.example_ws import get_example_speech, workspace_container

    CURRENT_DIR = pathlib.Path(__file__).parents[3]
    WORKSPACE_DIR = CURRENT_DIR.joinpath("examples", ".workspace").absolute()

    print("WORKSPACE_DIR: ", WORKSPACE_DIR)

    with workspace_container(WORKSPACE_DIR) as container:
        speech = get_example_speech(container)
        print("==> speech: ", speech)
        shell = new_ctml_shell(container=container, speech=speech)
        agent = SimpleAgentV2(
            instruction="你是 JoJo, 一个智能的AI小助手，请友善地回答我的问题",
            chat=ConsoleChat(),
            model=ModelConf(
                kwargs={
                    "thinking": {
                        "type": "disabled",
                    },
                },
            ),
            shell=shell,
        )

        asyncio.run(agent.run())
