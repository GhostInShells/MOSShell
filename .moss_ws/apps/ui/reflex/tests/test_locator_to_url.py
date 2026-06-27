import pytest
from ghoshell_moss import CommandError
from framework.runtime.event_generator import _locator_to_url


def test_locator_to_url_basic():
    url = _locator_to_url("local-video://workspace-assets/xgo.mp4")
    assert url == "http://127.0.0.1:20880/resources/local-video/workspace-assets/xgo.mp4"


def test_locator_to_url_with_subpath():
    url = _locator_to_url("pil-image://default/photos/sunset")
    assert url == "http://127.0.0.1:20880/resources/pil-image/default/photos/sunset"


def test_locator_to_url_invalid_no_scheme():
    with pytest.raises(CommandError, match="Invalid locator"):
        _locator_to_url("no-scheme-just-string")


def test_locator_to_url_minimal_path():
    url = _locator_to_url("scheme://host/p")
    assert url == "http://127.0.0.1:20880/resources/scheme/host/p"

