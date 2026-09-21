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
