"""Where the CLI reads its trajectory from — and where it refuses to look.

Two rules are pinned here, both of them about not guessing:

- no ``--root`` means ``.memento/ego`` under the working directory, and nothing else;
- a ``--root`` written before the subcommand is accepted like one written after it.

The second one used to fail with "No such option", which is a person's hand being wrong
about a tool rather than the tool being unclear.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ghoshell_moss.cli.memento_cli import DEFAULT_EGO_DIR, memento_app
from ghoshell_moss.cli.utils import set_ai_mode


@pytest.fixture(autouse=True)
def ai_mode():
    """Plain output: the mode these errors are actually read in, and the one click captures."""
    set_ai_mode(True)
    yield
    set_ai_mode(False)


@pytest.fixture()
def ego(tmp_path):
    """A working directory holding a trajectory, and the path to it."""
    root = tmp_path / DEFAULT_EGO_DIR
    root.mkdir(parents=True)
    return tmp_path, root


def test_cwd_default(ego, monkeypatch):
    cwd, root = ego
    monkeypatch.chdir(cwd)
    result = CliRunner().invoke(memento_app, ["status"])
    assert result.exit_code == 0, result.output
    assert str(root.resolve()) in result.output


def test_no_trajectory_is_an_error_not_a_search(tmp_path, monkeypatch):
    """No ``.memento/ego`` here: say so, and name the option that fixes it."""
    elsewhere = tmp_path / ".moss" / "ghosts" / "someone-else" / DEFAULT_EGO_DIR
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(memento_app, ["status"])
    assert result.exit_code == 2
    assert "--root" in result.output
    # The other ghost home sitting under the same tree is not an answer.
    assert "someone-else" not in result.output


def test_root_before_the_subcommand(ego, monkeypatch, tmp_path):
    """`memento --root X log` and `memento log --root X` both work."""
    _, root = ego
    monkeypatch.chdir(tmp_path)
    before = CliRunner().invoke(memento_app, ["--root", str(root), "status"])
    after = CliRunner().invoke(memento_app, ["status", "--root", str(root)])
    assert before.exit_code == 0, before.output
    assert after.exit_code == 0, after.output
    assert before.output == after.output


def test_missing_root_directory_is_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(memento_app, ["status", "--root", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "not a directory" in result.output
