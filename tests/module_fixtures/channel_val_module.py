"""Fixture module that provides an explicit Channel via __channel__."""

from ghoshell_moss.core import PyChannel

__channel__ = PyChannel(name="explicit", description="explicit channel")


@__channel__.build.command(name="hello")
async def hello() -> str:
    return "world"


def shadowed() -> str:
    # Should NOT be used when __channel__.py exists.
    return "no"
