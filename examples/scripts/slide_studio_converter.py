import sys

from PyQt6.QtWidgets import QApplication

from ghoshell_moss_contrib.gui.slide_studio_creator import PPTXConverterWindow
from ghoshell_moss_contrib.example_ws import workspace_container
from ghoshell_common.contracts import Workspace


if __name__ == "__main__":
    import pathlib
    CURRENT_DIR = pathlib.Path(__file__).parent
    WORKSPACE_DIR = CURRENT_DIR.parent.joinpath(".workspace").absolute()
    with workspace_container(WORKSPACE_DIR) as _container:
        ws = _container.force_fetch(Workspace)
        storage = ws.assets().sub_storage("slide_studio")
        app = QApplication(sys.argv)
        window = PPTXConverterWindow(storage)
        window.show()
        sys.exit(app.exec())
