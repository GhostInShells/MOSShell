# screen-node 视觉原型 demo — 纯 QML, 无 MOSS 依赖.
# 验证: background 槽 (数字人占位) + 浮游 meta 层 + 聚焦层 + layout 切换动效.
#
# 依赖 (不进 pyproject, uv sync 会清掉, 重装即可):
#   uv pip install PySide6-Essentials -i https://pypi.tuna.tsinghua.edu.cn/simple
# 运行:
#   .venv/bin/python .ai_partners/features/workstreams/2026/07/screen-node/demo/screen_demo.py

import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


def main() -> None:
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load(str(Path(__file__).parent / "Screen.qml"))
    if not engine.rootObjects():
        sys.exit(1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
