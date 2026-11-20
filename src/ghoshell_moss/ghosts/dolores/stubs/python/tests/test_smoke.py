"""Smoke: the capability tree is a package and imports without side effects."""

import DOLORES


def test_package_loads() -> None:
    assert DOLORES.__name__ == "DOLORES"
