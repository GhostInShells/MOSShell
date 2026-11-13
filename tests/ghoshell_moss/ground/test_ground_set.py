"""DefaultGroundSet — 一棵场树: 一个根 + 一个边界, 按位置渲染.

测的是协议承诺 (SPEC §6.1): 位置决定看哪个场, 边界之上永不咨询, 渲染是冷读,
失败是结果. 不测任何内部缓存 —— 因为现在已经没有缓存了.
"""

import asyncio
from pathlib import Path

from ghoshell_moss.ground import DefaultGroundSet


def _ground(root: Path, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "GROUND.md").write_text(text, encoding="utf-8")


def run(coro):
    return asyncio.run(coro)


class TestRootAndBoundary:
    def test_boundary_defaults_to_root(self, tmp_path: Path):
        gs = DefaultGroundSet(root=tmp_path)
        assert gs.root == tmp_path.resolve()
        assert gs.boundary == tmp_path.resolve()

    def test_boundary_can_be_wider_than_root(self, tmp_path: Path):
        inner = tmp_path / "inner"
        inner.mkdir()
        gs = DefaultGroundSet(root=inner, boundary=tmp_path)
        assert gs.root == inner.resolve()
        assert gs.boundary == tmp_path.resolve()


class TestGroundAt:
    def test_position_with_its_own_ground(self, tmp_path: Path):
        _ground(tmp_path / "a", "---\nname: a\n---\n")
        gs = DefaultGroundSet(root=tmp_path)
        assert gs.ground_at(tmp_path / "a") == (tmp_path / "a").resolve()

    def test_position_without_ground_uses_nearest_ancestor(self, tmp_path: Path):
        _ground(tmp_path / "a", "---\nname: a\n---\n")
        (tmp_path / "a" / "deep" / "deeper").mkdir(parents=True)
        gs = DefaultGroundSet(root=tmp_path)
        assert gs.ground_at(tmp_path / "a" / "deep" / "deeper") == (tmp_path / "a").resolve()

    def test_never_looks_above_the_boundary(self, tmp_path: Path):
        _ground(tmp_path, "---\nname: outer\n---\n")
        inner = tmp_path / "inner"
        inner.mkdir()
        gs = DefaultGroundSet(root=inner)
        # 边界是 inner, 外面的 tmp_path/GROUND.md 不被咨询
        assert gs.ground_at(inner) is None

    def test_position_outside_boundary_is_none(self, tmp_path: Path):
        gs = DefaultGroundSet(root=tmp_path / "home")
        assert gs.ground_at(tmp_path) is None


class TestRender:
    def test_renders_the_root_face(self, tmp_path: Path):
        _ground(tmp_path, "---\nname: g\n---\n\nTHE BODY\n")
        out = run(DefaultGroundSet(root=tmp_path).render())
        assert 'name="g"' in out
        assert "THE BODY" in out

    def test_no_ground_is_a_result_not_an_exception(self, tmp_path: Path):
        out = run(DefaultGroundSet(root=tmp_path).render())
        assert 'status="empty"' in out

    def test_render_is_a_cold_read(self, tmp_path: Path):
        """改盘上的 GROUND.md, 下一次渲染就看得见 —— 没有缓存, 没有 changed 标志."""
        _ground(tmp_path, "---\nname: g\n---\n\nBEFORE\n")
        gs = DefaultGroundSet(root=tmp_path)
        assert "BEFORE" in run(gs.render())
        (tmp_path / "GROUND.md").write_text("---\nname: g\n---\n\nAFTER\n", encoding="utf-8")
        assert "AFTER" in run(gs.render())

    def test_render_a_subdirectory_walks(self, tmp_path: Path):
        _ground(tmp_path, "---\nname: g\n---\n\nBODY\n")
        (tmp_path / "sub").mkdir()
        out = run(DefaultGroundSet(root=tmp_path).render(tmp_path / "sub"))
        assert "<cwd" in out
        assert "BODY" not in out

    def test_template_by_name(self, tmp_path: Path):
        _ground(tmp_path, "---\nname: g\n---\n\nREAL BODY\n")
        tables = tmp_path / ".grounds"
        tables.mkdir()
        (tables / "lens.md").write_text("---\nname: lens\n---\n\nLENS BODY\n", encoding="utf-8")
        out = run(DefaultGroundSet(root=tmp_path).render(template="lens"))
        assert "LENS BODY" in out
        assert "REAL BODY" not in out

    def test_unknown_template_is_an_error_face(self, tmp_path: Path):
        _ground(tmp_path, "---\nname: g\n---\n\nBODY\n")
        out = run(DefaultGroundSet(root=tmp_path).render(template="nope"))
        assert 'status="error"' in out
        assert "template not found" in out
