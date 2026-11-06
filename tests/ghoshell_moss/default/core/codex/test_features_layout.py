"""Features-directory discovery — the layout constants and the two-stage walk.

The contract under test:

- a project is anchored by ``.git``; without one, discovery fails rather than
  inventing ``$CWD/.ai_partners/features`` the way it used to;
- inside a project, the tree is found from ``$CWD`` up to the project home,
  nearest directory first, ``.features`` before the legacy ``.ai_partners/features``;
- an explicit ``--dir`` is taken as given, never redirected;
- ``init`` does not discover at all — it creates ``$CWD/.features``.
"""
from pathlib import Path

import pytest

from ghoshell_moss.core.codex.features import (
    FEATURES_DIRNAME,
    LEGACY_FEATURES_DIRNAME,
    FeaturesDirNotFoundError,
    find_features_dir,
    find_features_root,
    find_project_home,
    init_features,
    require_features_dir,
    resolve_features_dir,
)


def _make_features(root: Path, name: str | Path = FEATURES_DIRNAME) -> Path:
    """Create a minimal features tree at ``root / name`` and return it."""
    fd = root / name
    (fd / "workstreams").mkdir(parents=True)
    (fd / "README.md").write_text("# features\n", encoding="utf-8")
    return fd


def _make_project(root: Path, marker_is_dir: bool = True) -> Path:
    """Anchor a project at ``root`` — a ``.git`` directory, or a worktree's file."""
    root.mkdir(parents=True, exist_ok=True)
    if marker_is_dir:
        (root / ".git").mkdir()
    else:
        (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# stage 1 — the project home
# ---------------------------------------------------------------------------

def test_project_home_is_the_nearest_git_above(tmp_path, monkeypatch):
    project = _make_project(tmp_path / "proj")
    deep = project / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert find_project_home() == project


def test_project_home_accepts_a_git_file(tmp_path, monkeypatch):
    """A worktree or submodule has `.git` as a file — `exists()` is the test, not `is_dir()`."""
    project = _make_project(tmp_path / "proj", marker_is_dir=False)
    monkeypatch.chdir(project)
    assert find_project_home() == project


def test_nested_repository_resolves_to_itself(tmp_path, monkeypatch):
    outer = _make_project(tmp_path / "outer")
    inner = _make_project(outer / "vendor" / "inner")
    monkeypatch.chdir(inner)
    assert find_project_home() == inner


def test_no_git_means_no_project(tmp_path, monkeypatch):
    loose = tmp_path / "loose"
    loose.mkdir()
    monkeypatch.chdir(loose)
    assert find_project_home() is None


# ---------------------------------------------------------------------------
# stage 2 — the features tree
# ---------------------------------------------------------------------------

def test_discovery_walks_up_to_the_project_root(tmp_path, monkeypatch):
    project = _make_project(tmp_path / "proj")
    expected = _make_features(project)
    deep = project / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert resolve_features_dir() == expected


def test_nearest_tree_wins(tmp_path, monkeypatch):
    project = _make_project(tmp_path / "proj")
    near = _make_features(project / "src")
    _make_features(project)
    deep = project / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert resolve_features_dir() == near


def test_canonical_name_beats_the_legacy_one(tmp_path, monkeypatch):
    project = _make_project(tmp_path / "proj")
    canonical = _make_features(project)
    _make_features(project, LEGACY_FEATURES_DIRNAME)
    monkeypatch.chdir(project)
    assert resolve_features_dir() == canonical


def test_legacy_tree_is_still_found(tmp_path, monkeypatch):
    """`.ai_partners/features` is the historical layout and keeps working."""
    project = _make_project(tmp_path / "proj")
    legacy = _make_features(project, LEGACY_FEATURES_DIRNAME)
    monkeypatch.chdir(project)
    assert resolve_features_dir() == legacy


def test_a_directory_without_workstreams_is_not_a_tree(tmp_path, monkeypatch):
    project = _make_project(tmp_path / "proj")
    (project / FEATURES_DIRNAME).mkdir()
    monkeypatch.chdir(project)
    assert resolve_features_dir() is None


def test_discovery_does_not_escape_the_project_home(tmp_path, monkeypatch):
    """A tree above the project root belongs to somebody else."""
    _make_features(tmp_path)
    project = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project)
    assert resolve_features_dir() is None


# ---------------------------------------------------------------------------
# explicit path — taken as given
# ---------------------------------------------------------------------------

def test_explicit_dir_is_never_redirected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a project, nothing to discover
    given = tmp_path / "anywhere" / "features"
    assert resolve_features_dir(given) == given
    assert require_features_dir(given) == given


# ---------------------------------------------------------------------------
# refusal — the point of the change
# ---------------------------------------------------------------------------

def test_require_raises_outside_a_project(tmp_path, monkeypatch):
    loose = tmp_path / "loose"
    loose.mkdir()
    monkeypatch.chdir(loose)
    with pytest.raises(FeaturesDirNotFoundError) as exc:
        require_features_dir()
    assert ".git" in str(exc.value)


def test_require_raises_inside_a_project_without_a_tree(tmp_path, monkeypatch):
    project = _make_project(tmp_path / "proj")
    monkeypatch.chdir(project)
    with pytest.raises(FeaturesDirNotFoundError) as exc:
        require_features_dir()
    assert ".features" in str(exc.value) and ".ai_partners" in str(exc.value)


# ---------------------------------------------------------------------------
# creation — no discovery
# ---------------------------------------------------------------------------

def test_init_creates_dot_features_without_a_project(tmp_path, monkeypatch):
    loose = tmp_path / "loose"
    loose.mkdir()
    monkeypatch.chdir(loose)
    fd = init_features()
    assert fd == loose / FEATURES_DIRNAME
    assert (fd / "workstreams").is_dir()


def test_init_honours_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "custom-tree"
    assert init_features(target) == target
    assert (target / "workstreams").is_dir()
    # `--dir` points at the tree itself, so it must not gain a `.features` level.
    assert not (target / FEATURES_DIRNAME).exists()


# ---------------------------------------------------------------------------
# recovery from a FEATURE.md path
# ---------------------------------------------------------------------------

def test_find_features_root_recognizes_both_names(tmp_path):
    for name in (FEATURES_DIRNAME, LEGACY_FEATURES_DIRNAME):
        fd = _make_features(tmp_path / str(name).replace("/", "_"), name)
        feature_md = fd / "workstreams" / "2026" / "11" / "x" / "FEATURE.md"
        feature_md.parent.mkdir(parents=True)
        feature_md.write_text("---\ntitle: x\n---\n", encoding="utf-8")
        assert find_features_root(feature_md) == fd


def test_find_features_dir_is_none_outside_a_project(tmp_path, monkeypatch):
    loose = tmp_path / "loose"
    loose.mkdir()
    monkeypatch.chdir(loose)
    assert find_features_dir() is None
