from typing import Iterable
import pytest

from ghoshell_moss.ctml.token_parser import CTMLTokenParser
from ghoshell_moss.ctml.elements import CommandTaskElementContext
from ghoshell_moss.concepts.command import PyCommand, BaseCommandTask, Command
from ghoshell_moss.concepts.interpreter import CommandTaskElement
from ghoshell_moss.mocks.outputs import ArrOutput
from collections import deque
from dataclasses import dataclass
import asyncio


@dataclass
class ElementTestSuite:
    ctx: CommandTaskElementContext
    parser: CTMLTokenParser
    root: CommandTaskElement
    queue: deque[BaseCommandTask]

    def as_tuple(self):
        return self.ctx, self.parser, self.root, self.queue

    async def parse(self, content: Iterable[str], run: bool = True) -> None:
        with self.parser:
            for c in content:
                self.parser.feed(c)
        if run:
            gather = []
            for task in self.queue:
                gather.append(task.run())
            await asyncio.gather(*gather, return_exceptions=True)


def new_test_suite(*commands: Command) -> ElementTestSuite:
    tasks_queue = deque()
    output = ArrOutput()
    command_map = {c.name(): c for c in commands}
    ctx = CommandTaskElementContext(
        command_map,
        output,
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
    )


@pytest.mark.asyncio
async def test_element_baseline():
    ctx, parser, root, q = new_test_suite().as_tuple()
    content = ["<foo />", "hello", "<bar />", "world", "<baz />"]
    with parser:
        for c in content:
            parser.feed(c)

    assert len(list(parser.parsed())) == (2 + 1 + 2 + 1 + 2)

    # 模拟执行所有的命令
    for cmd_task in q:
        await cmd_task.run()
    # 由于没有任何真实的 command, 所以实际上只有两个 output stream 被执行了.
    assert len(q) == 2
    assert ctx.output.clear() == ["hello", "world"]


@pytest.mark.asyncio
async def test_element_baseline():
    async def foo() -> int:
        return 123

    async def bar(a: int) -> int:
        return a

    suite = new_test_suite(PyCommand(foo), PyCommand(bar))
    await suite.parse(['<foo /><bar a="123">', "hello", "</bar>"], run=True)
    assert len(list(suite.parser.parsed())) == (2 + 1 + 1 + 1)
    assert len(suite.queue) == 4
    assert [c.result for c in suite.queue] == [123, 123, None, None]
    suite.root.destroy()
