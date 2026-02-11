import sys
from os.path import join, dirname

from PyQt6.QtWidgets import QApplication
from ghoshell_common.contracts import Workspace

from ghoshell_moss.transports.zmq_channel import ZMQChannelProvider
from ghoshell_moss_contrib.channels.slide_studio import SlideStudio, SlideAssets
from ghoshell_moss_contrib.example_ws import workspace_container

current_dir = dirname(__file__)
workspace_dir = join(dirname(current_dir), ".workspace")


if __name__ == "__main__":
    with workspace_container(workspace_dir) as _container:
        _app = QApplication(sys.argv)
        ws = _container.force_fetch(Workspace)
        studio_storage = ws.assets().sub_storage("slide_studio")
        studio = SlideStudio(SlideAssets(studio_storage), _container)
        provider = ZMQChannelProvider(
            address=f"ipc://{__file__}.sock",
            container=_container,
        )
        provider.run_in_thread(studio.as_channel())
        studio.player.viewer.show()

        sys.exit(_app.exec())
