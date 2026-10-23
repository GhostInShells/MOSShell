"""表面的两件小事：它读主题文件时认的是"主题名"，不是文件名。

这一层很容易写错而看不出来：名字写错一个后缀，换主题照样成功，只是那一套
"场景"的内容被静默忽略 —— 颜色换过去了，桌上的人没换。
"""

from __future__ import annotations

import json
from pathlib import Path

from ghoshell_screen_manager.audio import MockAudioSource
from ghoshell_screen_manager.model import ScreenModel
from ghoshell_screen_manager.surface import ScreenSurface


def _surface(themes_dir: Path) -> ScreenSurface:
    return ScreenSurface(
        ScreenModel(),
        MockAudioSource(),
        send_signal=lambda _msg: None,
        self_identity="test",
        html_path=Path(__file__),
        themes_dir=themes_dir,
    )


def test_theme_reads_by_name_without_the_suffix(tmp_path: Path):
    """``theme(name)`` 收的是名字（``violet``）；后缀是这一层自己补的。"""
    (tmp_path / "violet.json").write_text(
        json.dumps({"name": "violet", "presence": "works/who.html"}), encoding="utf-8")
    surface = _surface(tmp_path)
    assert surface.themes() == ["violet"]
    assert surface.theme("violet") == {"name": "violet", "presence": "works/who.html"}


def test_theme_refuses_anything_that_is_not_a_bare_name(tmp_path: Path):
    """路径、空名、带层级的名字都不该被当成主题读出来 —— 这是读文件的入口。"""
    (tmp_path / "violet.json").write_text("{}", encoding="utf-8")
    surface = _surface(tmp_path)
    for bad in ("", "../violet", "sub/violet", "/etc/passwd"):
        assert surface.theme(bad) is None


def test_theme_missing_file_is_none_not_an_error(tmp_path: Path):
    surface = _surface(tmp_path)
    assert surface.theme("nope") is None
