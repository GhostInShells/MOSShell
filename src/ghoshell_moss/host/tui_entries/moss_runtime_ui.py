from typing import Iterable

from ghoshell_moss.host.abcd import MossHost, MossRuntime
from ghoshell_moss.host.abcd.tui import TUIState, MossHostTUI
from ghoshell_moss.host.repl.repl_state import REPLState
from ghoshell_moss.host.repl.inspector_matrix import MatrixREPL
from ghoshell_moss.host.repl.inspector_manifests import ManifestsREPL
from ghoshell_moss.host.repl.inspector_app_store import AppStoreREPL
from ghoshell_moss.host.repl.inspector_moss_runtime import MOSSRuntimeInspector
from ghoshell_moss.core.blueprint.session import OutputItem

__all__ = ['MOSSRuntimeREPLState', 'MossRuntimeTUI']


class MOSSRuntimeREPLState(REPLState):

    def __init__(
            self,
            host: MossHost,
            toolset: MossRuntime,
            name: str = 'MOSS',
    ) -> None:
        self._host = host
        self._toolset = toolset
        super().__init__(name)

    def _create_repl_inspectors(self) -> dict[str, object]:
        return {
            "matrix": MatrixREPL(self._host.matrix()),
            "manifests": ManifestsREPL(self._host.manifests),
            "moss": MOSSRuntimeInspector(self._toolset, self.console),
            "apps": AppStoreREPL(self._toolset.apps)
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
        result = await self._toolset.moss_exec(console_input)
        self.console.output(OutputItem.new("Shell", *result, log="execution done"))


class MossRuntimeTUI(MossHostTUI[MossRuntime]):

    @classmethod
    def _get_runtime(cls, host: MossHost) -> MossRuntime:
        return host.run()

    def create_states(self) -> Iterable[TUIState]:
        yield MOSSRuntimeREPLState(self.host, self.runtime)


if __name__ == "__main__":
    repl = MossRuntimeTUI()
    repl.run()
