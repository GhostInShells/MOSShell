"""Dolores memento owner registry — names resolve under one home, owners are weakly held."""

from pathlib import Path

import pytest

from ghoshell_moss.ghosts.dolores.mementos import FsDoloresMementos
from ghoshell_moss.memento import MementoReadonlyError


def test_get_creates_an_owner_only_when_asked(tmp_path: Path):
    mementos = FsDoloresMementos(tmp_path / ".memento")

    with pytest.raises(FileNotFoundError):
        mementos.get("ego")

    owner = mementos.get("ego", or_create=True)
    assert owner.owner == "ego"
    assert owner.root == tmp_path / ".memento" / "ego"


def test_a_nested_owner_keeps_its_full_name(tmp_path: Path):
    mementos = FsDoloresMementos(tmp_path / ".memento")

    owner = mementos.get("dreamers/journal", or_create=True, writable=False)
    assert owner.owner == "dreamers/journal"
    assert owner.root == tmp_path / ".memento" / "dreamers" / "journal"


def test_the_same_owner_comes_back_while_it_is_held(tmp_path: Path):
    mementos = FsDoloresMementos(tmp_path / ".memento")

    first = mementos.get("ego", or_create=True)
    assert mementos.get("ego") is first


def test_get_upgrades_an_open_owner_to_writable(tmp_path: Path):
    mementos = FsDoloresMementos(tmp_path / ".memento")
    reader = mementos.get("ego", or_create=True, writable=False)

    with pytest.raises(MementoReadonlyError):
        reader.create_branch("main")

    writer = mementos.get("ego", writable=True)
    assert writer is reader
    assert writer.create_branch("main").owner == "ego"
