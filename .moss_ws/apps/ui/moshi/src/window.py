"""MoshiWindow — 可拓展的原生桌面壳窗口。

QMainWindow 骨架 + QWebEngineView 主区域，带启动加载态：
Reflex 等本地服务启动慢于窗口，启动期间展示 loading 画面，
后台轮询检测目标 URL 可用后自动切到页面。

底部字幕条流式渲染 Ghost 输出（matrix.session.get_logos()），可关闭。
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QProgressBar, QStackedWidget,
    QTextEdit, QPushButton, QHBoxLayout,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


class _LoadingOverlay(QWidget):
    """深色加载画面：居中文字 + 不确定进度条。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("#0f0f1a"))
        self.setPalette(p)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

        self._label = QLabel("MOSHI 正在启动...")
        self._label.setStyleSheet(
            "color: #a0a0c0; font-size: 18px; font-family: sans-serif;"
        )
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setFixedWidth(300)
        self._bar.setFixedHeight(4)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            "QProgressBar { background: #1a1a2e; border: none; border-radius: 2px; }"
            "QProgressBar::chunk { background: #6c6cff; border-radius: 2px; }"
        )
        layout.addWidget(self._bar, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

    def set_message(self, text: str) -> None:
        self._label.setText(text)


class _SubtitleBar(QWidget):
    """底部字幕条：流式渲染 Ghost 输出，可关闭。"""

    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(160)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor("#0c0c1a"))
        self.setPalette(p)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 12, 10)
        layout.setSpacing(10)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._text.setStyleSheet(
            "QTextEdit {"
            "  background: transparent;"
            "  border: none;"
            "  color: #c8c8e0;"
            "  font-size: 15px;"
            "  font-family: 'PingFang SC', 'Noto Sans SC', sans-serif;"
            "}"
        )
        self._text.setFont(QFont("PingFang SC", 15))
        self._text.setPlaceholderText("等待 Ghost 输出...")
        layout.addWidget(self._text, stretch=1)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent;"
            "  border: none;"
            "  color: #555570;"
            "  font-size: 14px;"
            "}"
            "QPushButton:hover {"
            "  color: #ff5c5c;"
            "}"
        )
        close_btn.clicked.connect(self.hide)
        close_btn.clicked.connect(self.closed.emit)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignTop)

    def append_text(self, delta: str) -> None:
        """追加流式文本，自动滚底。"""
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(delta)
        scrollbar = self._text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    def clear_text(self) -> None:
        self._text.clear()

    def set_status(self, text: str) -> None:
        """设置状态文本（替换 placeholder，用于诊断信息）。"""
        if text:
            self._text.setPlaceholderText(text)
        else:
            self._text.setPlaceholderText("")


class MoshiWindow(QMainWindow):
    """可拓展的桌面壳窗口，内嵌 Chromium webview，带启动加载检测。

    底部字幕条流式渲染 Ghost 输出，默认隐藏，可通过 toggle 或
    直接调用 show_subtitle() 显示。
    """

    def __init__(
        self,
        url: str = "http://localhost:3000",
        title: str = "MOSHI",
        width: int = 1280,
        height: int = 800,
        check_interval_ms: int = 1000,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(width, height)

        central = QWidget()
        central.setAutoFillBackground(True)
        p_central = central.palette()
        p_central.setColor(central.backgroundRole(), QColor("#060610"))
        central.setPalette(p_central)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._loading = _LoadingOverlay()
        self.webview = QWebEngineView()
        self.webview.page().setBackgroundColor(QColor("#0f0f1a"))

        self._stack = QStackedWidget()
        self._stack.addWidget(self._loading)
        self._stack.addWidget(self.webview)
        layout.addWidget(self._stack)

        # 底部字幕条
        self.subtitle = _SubtitleBar()
        layout.addWidget(self.subtitle)

        self._target_url = url
        self._checking = False
        self._network = QNetworkAccessManager()
        self._network.finished.connect(self._on_check_response)

        self._check_timer = QTimer()
        self._check_timer.setInterval(check_interval_ms)
        self._check_timer.timeout.connect(self._check_server)

        self._stack.setCurrentIndex(0)
        self._check_timer.start()

    # ---- 健康检查 ----

    def _check_server(self) -> None:
        if self._checking:
            return
        self._checking = True
        req = QNetworkRequest(QUrl(self._target_url))
        self._network.head(req)

    def _on_check_response(self, reply: QNetworkReply) -> None:
        self._checking = False
        if reply.error() == QNetworkReply.NetworkError.NoError:
            self._check_timer.stop()
            self.webview.setUrl(QUrl(self._target_url))
            self._stack.setCurrentIndex(1)
        reply.deleteLater()

    # ---- public ----

    def load_url(self, url: str) -> None:
        self._target_url = url
        self._checking = False
        self._stack.setCurrentIndex(0)
        self._loading.set_message(f"正在连接 {url}...")
        self._check_timer.start()

    def eval_js(self, code: str) -> None:
        self.webview.page().runJavaScript(code)

    def show_subtitle(self) -> None:
        self.subtitle.show()

    def hide_subtitle(self) -> None:
        self.subtitle.hide()

    def toggle_subtitle(self) -> None:
        if self.subtitle.isVisible():
            self.subtitle.hide()
        else:
            self.subtitle.show()
