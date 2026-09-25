"""Runtime error self-diagnosis — pull the bounded error tail | 诊断 | beta

A pull-only surface over ``RuntimeErrorLog``: the model reads recent ERROR+ / CRITICAL records
to self-diagnose startup and runtime degradations (NullSpeech / NullASR and the like). Pull
semantics keep it zero-cost when not called — only the command signature is reflected, no error
state is injected into context.

The collector is a logging.Handler on the moss logger, attached before provider bootstrap, so
startup failures are captured even before the shell is up.

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.runtime_error_channel import new_runtime_error_channel
    main = new_shell_main_channel()
    main.import_channels(new_runtime_error_channel())
"""

from __future__ import annotations

from ghoshell_moss.core import ChannelCtx
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.contracts.runtime_error import RuntimeErrorLog

__all__ = ["new_runtime_error_channel"]

_DEFAULT_N = 20


def new_runtime_error_channel(
        *,
        log: RuntimeErrorLog | None = None,
        name: str = "runtime_error",
        description: str = (
            "Runtime error self-diagnosis — pull recent ERROR+/CRITICAL records to see why a "
            "capability degraded"
        ),
) -> MutableChannel:
    """Build a runtime error channel.

    :param log: the collector to read. None → resolve ``RuntimeErrorLog`` from the IoC container
        at call time (the production path); a missing contract degrades to a notice, not a crash.
    """
    chan = new_channel(name=name, description=description)

    def _resolve() -> RuntimeErrorLog | None:
        if log is not None:
            return log
        try:
            return ChannelCtx.get_contract(RuntimeErrorLog)
        except Exception:
            return None

    @chan.build.command(name="pull", always_observe=True)
    async def pull(critical: bool = False, n: int = _DEFAULT_N) -> str:
        """Pull recent runtime error records — for self-diagnosis, not everyday use.

        When a capability is silently missing (speech / ASR degraded), read the tail to see why.
        Errors you caused yourself come back through your tool results; this surfaces background
        and startup failures that never reach you otherwise.

        :param critical: True → only the never-dropped CRITICAL buffer; False → the latest n ERROR+.
        :param n: how many of the latest records to return (ignored when critical=True).
        """
        collector = _resolve()
        if collector is None:
            return "[runtime_error] not available (no RuntimeErrorLog registered)"
        records = collector.pull_critical() if critical else collector.pull(n)
        if not records:
            return "[runtime_error] no errors recorded"
        return "\n".join(record.render() for record in records)

    return chan
