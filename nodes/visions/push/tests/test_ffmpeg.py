import pytest

from push_node.ffmpeg import build_argv


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        build_argv("bogus")


def test_numeric_knobs_are_clamped_not_rejected():
    argv = build_argv("screen", fps=60, max_width=99999, quality=100)
    # fps caps at 30, width at 3840, quality at 31.
    joined = " ".join(argv)
    assert "framerate" in joined
    assert "min(3840,iw)" in joined
    assert "-q:v 31" in joined


def test_screen_gets_the_pixel_format_the_camera_does_not():
    screen = build_argv("screen")
    camera = build_argv("camera")
    assert "-pixel_format" in screen and "uyvy422" in screen
    assert "-pixel_format" not in camera


def test_argv_is_an_exec_list_not_a_shell_line():
    # The whole authorization surface is that the model can never smuggle a
    # shell metacharacter — argv goes to exec, never to a shell.
    for source in ("screen", "camera"):
        argv = build_argv(source)
        assert isinstance(argv, list)
        assert all(";" not in part and "|" not in part for part in argv)
        assert "pipe:1" in argv, "frames stream on stdout"


def test_the_requested_index_beats_the_node_default(monkeypatch):
    monkeypatch.setenv("PUSH_SCREEN_INDEX", "7")
    assert "-i 7:none" in " ".join(build_argv("screen")), "no index → node config"
    assert "-i 2:none" in " ".join(build_argv("screen", index=2)), "index → as asked"


def test_a_stock_mac_needs_no_configuration(monkeypatch):
    monkeypatch.delenv("PUSH_SCREEN_INDEX", raising=False)
    monkeypatch.delenv("PUSH_CAMERA_INDEX", raising=False)
    assert "-i 1:none" in " ".join(build_argv("screen"))
    assert "-i 0:none" in " ".join(build_argv("camera"))


def test_the_default_rate_is_the_source_s_own():
    """Capture devices take only the rates they advertise — an unadvertised rate
    fails the open instead of being clamped, so the default must be per source."""
    assert "-framerate 30" in " ".join(build_argv("camera"))
    assert "-framerate 10" in " ".join(build_argv("screen"))
    assert "-framerate 15" in " ".join(build_argv("camera", fps=15)), "asked → as asked"
