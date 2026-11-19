"""Dolores ghost_home 认知场骨架测试 — 场装配的协议承诺, 与 ghost 运行时无关.

被装配的是 Dolores 包里随包分发的 stubs 树; 这里验的是 ground 的机制, 不是 ghost.
- root 场只渲染 ``imports`` 点名的认知子场, 不穿透进子件内部
- existence 场 @ 装载 purpose/behaviors (冷层法), file pin 装载 identity (warm 帧)
- journal 是目录标记, 不是认知场: 它不出现在 root 的渲染里, 单独 render 时给出共享轨迹说明

装配的是这棵树的**副本**: 渲染可能跑 pin, 直接渲染包内的树等于把源码树写脏 ——
那是测试的错, 不是 pin 的错.
"""

import asyncio
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from ghoshell_moss.ground import DefaultGroundSet

# 被测的桩树住在 dolores 包里; 测试自己的地址说明不了它在哪.
STUBS = Path(str(files("ghoshell_moss.ghosts.dolores"))) / "stubs"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def stub_tree(tmp_path: Path) -> Path:
    """The packaged skeleton, copied — the only tree a test may render."""
    dest = tmp_path / "stubs"
    shutil.copytree(STUBS, dest)
    return dest


class TestRootGroundDisclosure:
    def test_renders_imported_subfields_only(self, stub_tree: Path):
        """root 场只渲染 imports 点名的认知子场; journal 是目录, 不是场."""

        async def scenario():
            return await DefaultGroundSet(root=stub_tree).render()

        text = run(scenario())
        assert 'name="existence"' in text
        assert 'name="people"' in text
        assert 'name="skills"' in text
        # journal 已从认知面摘除: 它仍是目录, 但不作为子场渲染.
        assert 'name="journal"' not in text
        # startup 是操作性的, 不进面.
        assert 'name="startup"' not in text
        # root 不穿透进 people 子场, 不披露具体人物.
        assert "thirdgerb" not in text


class TestExistenceGroundDisclosure:
    def test_identity_pinned_and_law_expanded(self, stub_tree: Path):
        """existence 场: identity 走 file pin (warm), purpose/behaviors 走 @ (冷层)."""

        async def scenario():
            return await DefaultGroundSet(root=stub_tree / "existence").render()

        text = run(scenario())
        # identity 经 file pin 装载 (事实自我内容).
        assert "## 当前状态" in text
        # purpose / behaviors 经 @ 装载 (冷层法).
        assert "## 意义" in text
        assert "# Behaviors" in text


class TestJournalGroundMarker:
    def test_journal_is_a_directory_marker_not_a_field(self, stub_tree: Path):
        """journal/GROUND.md 只说明目录与共享轨迹; 它不是 root 认知面的一部分."""

        marker = run(DefaultGroundSet(root=stub_tree / "journal").render())
        assert "Ghost 所有 Session 共享的轨迹" in marker

        root = run(DefaultGroundSet(root=stub_tree).render())
        assert 'name="journal"' not in root
