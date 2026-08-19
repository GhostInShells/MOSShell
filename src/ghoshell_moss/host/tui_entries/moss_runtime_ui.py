from typing import Iterable

from ghoshell_moss.core.blueprint.host import IHost, IShellRuntime
from ghoshell_moss.host.tui import TUIState, MossHostTUI
from ghoshell_moss.host.repl.repl_state import REPLState
from ghoshell_moss.host.repl.inspector_matrix import MatrixInspector
from ghoshell_moss.host.repl.inspector_manifests import ManifestsInspector
from ghoshell_moss.host.repl.inspector_moss_runtime import MOSSRuntimeInspector
from ghoshell_moss.core.blueprint.session import OutputItem

__all__ = ['MOSSRuntimeREPLState', 'MossRuntimeTUI']


class MOSSRuntimeREPLState(REPLState):

    def __init__(
            self,
            host: IHost,
            moss: IShellRuntime,
            name: str = 'MOSS',
    ) -> None:
        self._host = host
        self._moss_runtime = moss
        super().__init__(name)

    def _create_repl_inspectors(self) -> dict[str, object]:
        moss = self._moss_runtime
        mode = moss.mode if moss.is_running() else None
        return {
            "matrix": MatrixInspector(moss.matrix),
            "manifests": ManifestsInspector(
                moss.project.project_manifests(),
                mode.manifests() if mode else None,
            ),
            "moss": MOSSRuntimeInspector(moss, self.console),
        }

    def output_on_switch(self, enter_else_leave: bool) -> None:
        if enter_else_leave:
            self.console.info(
                "Enter MOSS runtime, use repl command start with  `/` or `?`, or input CTML for testing.\n\n"
                "QuickStart: \n"
                "- `/moss.instructions()`: meta instruction about MOSS and CTML.\n"
                "- `/moss.static()`: return static information about MOSS channels. \n"
                "- `/moss.dynamic()`: return dynamic information from MOSS channels. \n"
                "- `hello world`: test moss with raw CTML string (Logos). ",
            )
        else:
            self.console.info("Leave MOSS runtime")

    async def _on_text_input(self, console_input: str) -> None:
        result = await self._moss_runtime.exec_logos(console_input)
        self.console.output(OutputItem.new("Shell", *result, log="execution done"))


class MossRuntimeTUI(MossHostTUI[IShellRuntime]):

    def _get_runtime(self) -> IShellRuntime:
        return self.host.run()

    def _get_session(self):
        return self.runtime.session

    def _log_loop_exception(self, message: str, exception: BaseException | None) -> None:
        self.runtime.matrix.logger.exception("%s: %s", message, exception)

    def create_states(self) -> Iterable[TUIState]:
        yield MOSSRuntimeREPLState(self.host, self.runtime)


if __name__ == "__main__":
    repl = MossRuntimeTUI()
    repl.run()
