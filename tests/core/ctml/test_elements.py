import asyncio
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from ghoshell_moss.core.concepts.command import BaseCommandTask, Command, CommandToken, PyCommand
from ghoshell_moss.core.concepts.interpreter import CommandTokenParserElement
from ghoshell_moss.core.ctml.elements import CommandTaskElementContext
from ghoshell_moss.core.ctml.token_parser import CTMLTokenParser
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.speech.mock import MockSpeech


@dataclass
class ElementTestSuite:
    ctx: CommandTaskElementContext
    parser: CTMLTokenParser
    root: CommandTokenParserElement
    queue: deque[BaseCommandTask | None]
    stop_event: ThreadSafeEvent

    def as_tuple(self):
        return self.ctx, self.parser, self.root, self.queue

    def stop(self):
        self.parser.close()
        self.stop_event.set()

    async def parse(self, content: Iterable[str], run: bool = True) -> None:
        with self.parser:
            for c in content:
                self.parser.feed(c)
        if run:
            gathered = []
            for task in self.queue:
                if task is not None:
                    gathered.append(task.run())
            await asyncio.gather(*gathered, return_exceptions=False)


def new_test_suite(*commands: Command) -> ElementTestSuite:
    tasks_queue = deque()
    output = MockSpeech()
    command_map = {}
    for command in commands:
        chan = command.meta().chan
        if chan not in command_map:
            command_map[chan] = {}
        command_map[chan][command.name()] = command
    stop_event = ThreadSafeEvent()
    ctx = CommandTaskElementContext(
        command_map,
        output,
        stop_event=stop_event,
    )
    root = ctx.new_root(tasks_queue.append, stream_id="test")
    token_parser = CTMLTokenParser(
        callback=root.on_token,
        stream_id="test",
    )
    return ElementTestSuite(
        ctx=ctx,
        parser=token_parser,
        root=root,
        queue=tasks_queue,
        stop_event=stop_event,
    )


@pytest.mark.asyncio
async def test_element_with_no_command():
    suite = new_test_suite()
    ctx, parser, root, q = suite.as_tuple()
    assert root.depth == 0
    content = ["<foo />", "hello", "<bar />", "world", "<baz />"]
    with parser:
        for c in content:
            parser.feed(c)

    # <ctml>, <foo></foo>, "hello", <bar></bar>, "world", <baz></baz>
    assert len(list(parser.parsed())) == (1 + 2 + 1 + 2 + 1 + 2 + 1)

    # 模拟执行所有的命令
    for cmd_task in q:
        if cmd_task is not None:
            await cmd_task.run()
    # 由于没有任何真实的 command, 所以实际上只有两个 output stream 被执行了.
    assert len(q) == 3
    # 最后一个 item 是毒丸.
    assert q[-1] is None

    # 假设有正确的输出.
    assert await ctx.speech.clear() == ["hello", "world"]

    children = list(suite.root.children.values())
    assert len(children) == 1
    assert children[0].depth == 1

    count = 0
    for child in children[0].children.values():
        assert child.depth == 2
        count += 1
    # 三个空命令.
    assert count == 3


@pytest.mark.asyncio
async def test_element_baseline():
    async def foo() -> int:
        return 123

    async def bar(a: int) -> int:
        return a

    suite = new_test_suite(PyCommand(foo), PyCommand(bar))
    await suite.parse(['<foo /><bar a="123">', "hello", "</bar>"], run=True)
    assert len(list(suite.parser.parsed())) == (1 + 2 + 1 + 1 + 1 + 1)
    assert len(suite.queue) == 4 + 1  # 最后一个是 None
    assert suite.queue.pop() is None
    assert [c._result for c in suite.queue] == [123, 123, None, None]
    # the <foo /> is changed to <foo/> for fewer tokens usage
    assert "".join(c.tokens for c in suite.queue) == '<foo/><bar a="123">hello</bar>'
    suite.root.destroy()


@pytest.mark.asyncio
async def test_element_in_chaos_order():
    async def foo() -> int:
        return 123

    async def bar(a: int) -> int:
        return a

    suite = new_test_suite(PyCommand(foo), PyCommand(bar))
    await suite.parse(["<fo", "o /><b", 'ar a="12', '3">he', "llo<", "/bar>"], run=True)
    assert suite.queue.pop() is None
    assert [c._result for c in suite.queue] == [123, 123, None, None]
    suite.root.destroy()


@pytest.mark.asyncio
async def test_parse_and_execute_in_parallel():
    async def foo() -> int:
        return 123

    async def bar(a: int) -> int:
        return a

    suite = new_test_suite(PyCommand(foo), PyCommand(bar))
    _queue: asyncio.Queue[BaseCommandTask | None] = asyncio.Queue()
    # 所有的 command task 都会发送给这个 queue
    suite.root.with_callback(_queue.put_nowait)

    def producer():
        # feed the inputs
        with suite.parser:
            for char in ["<fo", "o /><b", 'ar a="12', '3">he', "llo<", "/bar>"]:
                suite.parser.feed(delta=char)

    tasks = []
    results = []

    async def consumer():
        while True:
            task = await _queue.get()
            if task is None:
                # 最后一个是 None, 用来打破循环.
                # 也是测试循环是否被打破了.
                break
            else:
                tasks.append(task.run())

        # 让 results 来承接所有 task 的返回值.
        results.extend(await asyncio.gather(*tasks))

    main_tasks = [
        asyncio.to_thread(producer),
        asyncio.create_task(consumer()),
    ]
    await asyncio.gather(*main_tasks)

    # suite.queue 被 _queue 夺舍了.
    assert len(suite.queue) == 0

    assert results == [123, 123, None, None]


@pytest.mark.asyncio
async def test_parse_text_command():
    async def foo(text__: str) -> str:
        return text__

    suite = new_test_suite(PyCommand(foo))
    await suite.parse(["<foo/>"], run=True)

    assert len(suite.queue) == 2
    assert suite.queue[0]._result == ""
    assert suite.queue[0].tokens == "<foo/>"

    suite = new_test_suite(PyCommand(foo))
    await suite.parse(["<foo> </foo>"], run=True)
    assert suite.queue.pop() is None
    assert suite.queue[0]._result == " "
    assert "".join(t.tokens for t in suite.queue) == "<foo> </foo>"


@pytest.mark.asyncio
async def test_parse_text_command_with_kwargs():
    async def foo(a: str, b: str = " ", text__: str = "") -> str:
        return a + b + text__

    suite = new_test_suite(PyCommand(foo))
    content = '<foo a="hello">world</foo>'
    await suite.parse([content], run=True)
    assert suite.queue.pop() is None
    # a + b + text__
    assert suite.queue[0]._result == "hello world"
    assert "".join(t.tokens for t in suite.queue) == content


@pytest.mark.asyncio
async def test_parse_token_delta_command():
    async def foo(tokens__) -> str:
        result = ""
        async for token in tokens__:
            assert isinstance(token, CommandToken)
            result += token.content
        return result

    suite = new_test_suite(PyCommand(foo))
    content = "<foo><![CDATA[hello<bar/>world]]></foo>"
    await suite.parse([content], run=True)
    assert suite.queue[0]._result == "hello<bar/>world"

    suite = new_test_suite(PyCommand(foo))
    # test without CDATA
    content = "<foo>hello<bar/>world</foo>"
    await suite.parse([content], run=True)
    #  once without cdata, the self-closing tag will separate to start and end token
    assert suite.queue[0]._result == "hello<bar></bar>world"
