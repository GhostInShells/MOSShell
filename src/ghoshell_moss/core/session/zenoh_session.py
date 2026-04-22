from typing import Callable

from ghoshell_moss.contracts import Storage, LoggerItf
from ghoshell_moss.core.concepts.session import Session, OutputItem, Signal
from threading import Event
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

__all__ = [
    'MossSessionWithZenoh',
]


class MossSessionWithZenoh(Session):
    """
    Session implementation for host
    """

    def __init__(
            self,
            session_scope: str,
            session_storage: Storage,
            logger: LoggerItf,
            zenoh_session: zenoh.Session,
    ):
        self._session_scope = session_scope
        self._output_key_expr = f"MOSS/{session_scope}/outputs"
        self._input_signal_expr = f"MOSS/{session_scope}/signals"
        self._session_storage = session_storage
        self._closing_event = Event()
        self._output_listeners: list[Callable[[OutputItem], None]] = []
        self._zenoh_session = zenoh_session
        if zenoh_session.is_closed():
            raise RuntimeError(f'HostSession receive Zenoh session but closed')
        self._output_sub = zenoh_session.declare_subscriber(self._output_key_expr, self._on_zenoh_output)
        self._input_sub = zenoh_session.declare_subscriber(self._input_signal_expr, self._on_zenoh_signal_input)
        self._logger = logger
        self._log_prefix = f'<Session cls={self.__class__} id={session_scope}>'
        self._on_signal_callbacks: list[Callable[[Signal], None]] = []

    @property
    def session_scope(self) -> str:
        return self._session_scope

    @property
    def storage(self) -> Storage:
        return self._session_storage

    def _check_running(self) -> None:
        if self._zenoh_session.is_closed():
            raise RuntimeError(f'HostSession is closed')

    def input(self, signal: Signal) -> None:
        self._check_running()
        # todo: 未来加防蠢限频.
        # 现在有一种深刻的感觉, 不存在过度设计, 只存在过度实现.
        js = signal.to_json()
        self._zenoh_session.put(self._output_key_expr, js)

    def on_input(self, callback: Callable[[Signal], None]) -> None:
        self._on_signal_callbacks.append(callback)

    def _on_zenoh_signal_input(self, sample: zenoh.Sample) -> None:
        if len(self._on_signal_callbacks) == 0:
            return None
        try:
            signal = Signal.model_validate_json(sample.payload.to_bytes())
        except Exception as e:
            self._logger.error(
                f"%s failed to handle received signal sample %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )
            return None
        for callback in self._on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self._logger.exception(
                    "%s failed to callback received signal on %s: %s",
                    self._log_prefix, callback, e
                )
        return None

    def output(self, *items: OutputItem) -> None:
        self._check_running()
        for item in items:
            js = item.to_json()
            self._zenoh_session.put(self._output_key_expr, js)

    def _on_zenoh_output(self, sample: zenoh.Sample) -> None:
        if len(self._output_listeners) == 0:
            return
        try:
            item = OutputItem.model_validate_json(sample.payload.to_bytes())
            for listener in self._output_listeners:
                try:
                    listener(item)
                except Exception as e:
                    self._logger.error(
                        "%s failed to send output %s: %s",
                        self._log_prefix, item.id, e,
                    )
        except Exception as e:
            self._logger.error(
                "%s failed to send output %s: %s",
                self._log_prefix, sample.payload.to_string(), e,
            )

    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        self._output_listeners.append(callback)

    def clear(self) -> None:
        if self._output_sub and not self._zenoh_session.is_closed():
            self._output_sub.undeclare()
        if self._input_sub and not self._zenoh_session.is_closed():
            self._input_sub.undeclare()
