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
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWebEngineQuick import QtWebEngineQuick
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineScript, QWebEngineProfile


def main():
    standalone = "--standalone" in sys.argv

    # QtWebEngine must initialize BEFORE QApplication.
    QtWebEngineQuick.initialize()

    app = QApplication(sys.argv)
    app.setApplicationName("screen")

    # Inject web scripts into default WebEngine profile (global, all views).
    _web_dir = _NODE_DIR / "src" / "screen_node" / "web"
    _profile = QWebEngineProfile.defaultProfile()
    for _name, _file, _point in [
        ("badge_intercept", "badge_intercept.js", QWebEngineScript.DocumentReady),
        ("inject_window_id", "inject_window_id.js", QWebEngineScript.DocumentCreation),
    ]:
        _script = QWebEngineScript()
        _script.setName(_name)
        _script.setSourceCode((_web_dir / _file).read_text())
        _script.setInjectionPoint(_point)
        _script.setWorldId(QWebEngineScript.MainWorld)
        _script.setRunsOnSubFrames(False)
        _profile.scripts().insert(_script)

    # Shared objects — created on main thread before Matrix starts.
    from screen_node.bucket import EventBucket
    from screen_node.bridge import ScreenBridge

    bucket = EventBucket()
    bridge = ScreenBridge(bucket)

    # QML engine
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("bridge", bridge)

    # QWebChannel — registers bridge so WebEngineView pages can call
    # bridge.web_badge_changed() via qt.webChannelTransport in JS.
    web_channel = QWebChannel()
    web_channel.registerObject("bridge", bridge)
    engine.rootContext().setContextProperty("webChannel", web_channel)

    # Expose web scripts directory for QML WebEngineScript source URLs.
    _web_dir = str(_NODE_DIR / "src" / "screen_node" / "web")
    engine.rootContext().setContextProperty("webScriptsDir", _web_dir)

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
