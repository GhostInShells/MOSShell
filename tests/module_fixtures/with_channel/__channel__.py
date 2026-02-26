"""Package-level explicit channel for ModuleChannel tests."""

from ghoshell_moss.core import PyChannel

__channel__ = PyChannel(name="pkg_explicit", description="explicit channel from __channel__.py")


@__channel__.build.command(name="hello")
async def hello() -> str:
    return "world"
