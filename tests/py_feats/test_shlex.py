import shlex


def test_shlex():
    parts = shlex.split("foo bar='abc' --opt --v -t=abc")
    assert len(parts) == 5
