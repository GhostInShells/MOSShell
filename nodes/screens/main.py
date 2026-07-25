"""Screen node entry point — macOS PySide6/QML screen body.

Start:  moss nodes run nodes/screens
Debug:  python main.py              # full: QML + Matrix daemon
        python main.py --standalone  # QML only, no Matrix (GUI smoke test)

Threading model (Decision 6, 9):
  Main thread   — QApplication + QML engine + scene graph (Qt event loop)
  Daemon thread — Matrix asyncio (channel logic + Ghost communication)
  Bridge        — Signal(str) queued connection + Future for channel→GUI,
                  EventBucket for GUI→channel (peek/drain from g1 listener)
"""

import sys
import threading
from pathlib import Path

_NODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_NODE_DIR / "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QTimer


def main():
    standalone = "--standalone" in sys.argv

    app = QApplication(sys.argv)
    app.setApplicationName("screen")

    # Shared objects — created on main thread before Matrix starts.
    from screen_node.bucket import EventBucket
    from screen_node.bridge import ScreenBridge

    bucket = EventBucket()
    bridge = ScreenBridge(bucket)

    # QML engine
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("bridge", bridge)

    qml_path = str(_NODE_DIR / "src" / "screen_node" / "qml" / "Screen.qml")
    engine.load(qml_path)
    if not engine.rootObjects():
        sys.exit(1)

    root = engine.rootObjects()[0]
    bridge.set_root(root)

    # Initial snapshot so context_messages has data before first command.
    bridge._refresh_snapshot()

    # Heartbeat timer — keeps the event loop responsive to bridge signals.
    # Qt.QueuedConnection handles cross-thread dispatch without polling.
    timer = QTimer()
    timer.setInterval(100)
    timer.start()

    if standalone:
        print("screen: standalone mode (no Matrix)")
        sys.exit(app.exec())

    # Matrix daemon thread
    def run_matrix():
        from screen_node.channels.screen import build_screen_channel

        async def main_async(matrix):
            channel = build_screen_channel(bridge, bucket)
            await matrix.provide_channel(channel)

        from ghoshell_moss.core.blueprint.matrix import Matrix
        Matrix.discover().run(main_async)

    matrix_thread = threading.Thread(target=run_matrix, daemon=True, name="moss-matrix")
    matrix_thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
