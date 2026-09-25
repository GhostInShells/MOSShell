"""Dolores ghost_home 认知场骨架测试 — 独立于 test_dolores.py.

覆盖 ground 装配的协议承诺:
- root 场渐进披露子件类别身份 (frontmatter pin 扫 */GROUND.md), 不穿透进子件内部
- existence 场 @ 装载 purpose/behaviors (冷层法), file pin 装载 identity (warm 帧)
- journal 场 exec pin 物化 Y/M/D 占位 + 分层呈现 (years/months 存在性, today 展开)
"""

import asyncio
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from ghoshell_moss.ground import DefaultGroundSet

STUBS = Path(__file__).parent / "stubs"


def run(coro):
    return asyncio.run(coro)


class TestRootGroundDisclosure:
    def test_discloses_subfield_categories_not_persons(self):
        """root 场只披露子件类别身份, 不穿透披露具体人物."""

        async def scenario():
            async with DefaultGroundSet(workspace_root=STUBS) as gs:
                ground = await gs.open(STUBS)
                return str(await ground.render())

        text = run(scenario())
        # 子件类别身份被渐进披露.
        assert "existence/GROUND.md" in text
        assert "people/GROUND.md" in text
        assert "skills/GROUND.md" in text
        assert "journal/GROUND.md" in text
        assert "startup/GROUND.md" in text
        assert "features/GROUND.md" in text
        # root 不穿透进 people 子场, 不披露具体人物.
        assert "thirdgerb" not in text


class TestExistenceGroundDisclosure:
    def test_identity_pinned_and_law_expanded(self):
        """existence 场: identity 走 file pin (warm), purpose/behaviors 走 @ (冷层)."""

        async def scenario():
            async with DefaultGroundSet(workspace_root=STUBS) as gs:
                ground = await gs.open(STUBS / "existence")
                return str(await ground.render())

        text = run(scenario())
        # identity 经 file pin 装载 (事实自我内容).
        assert "## 当前状态" in text
        # purpose / behaviors 经 @ 装载 (冷层法).
        assert "## 意义" in text
        assert "# Behaviors" in text


TIMELINE = STUBS / "journal" / "timeline.py"


class TestJournalTimelineScript:
    def _run(self, tmp_path) -> str:
        return subprocess.run(
            [sys.executable, str(TIMELINE)],
            env=dict(os.environ, GROUND=str(tmp_path)),
            capture_output=True, text=True,
        ).stdout

    def test_layered_view(self, tmp_path):
        """分层呈现: years/months 存在性, today 展开全文."""
        today = date.today()
        day_dir = tmp_path / f"Y{today.year}" / f"M{today.month:02d}" / f"D{today.day:02d}"
        day_dir.mkdir(parents=True)
        (day_dir / "daily.md").write_text(
            "---\nsummary: today\nstatus: writing\n---\n\nbody today\n"
        )
        # 去年 + 上月: 只应有存在性, 不应展开全文.
        prior_year = tmp_path / f"Y{today.year - 1}"
        prior_year.mkdir(parents=True)
        (prior_year / "yearly.md").write_text(
            "---\nsummary: last year\nstatus: closed\n---\n\nbody last year\n"
        )
        prior_month = tmp_path / f"Y{today.year}" / f"M{max(1, today.month - 1):02d}"
        prior_month.mkdir(parents=True)
        (prior_month / "monthly.md").write_text(
            "---\nsummary: last month\nstatus: closed\n---\n\nbody last month\n"
        )

        out = self._run(tmp_path)

        assert f"Y{today.year}" in out           # 今年存在性
        assert f"Y{today.year - 1}" in out       # 去年存在性
        assert f"M{today.month:02d}" in out      # 本月存在性
        assert "body today" in out               # 今天展开全文
        assert "body last year" not in out       # 去年不展开
        assert "body last month" not in out      # 上月不展开

    def test_materializes_missing_placeholders(self, tmp_path):
        """空场: 脚本物化当天/当月/当年占位, status=pending, 只建不写已存在的."""
        today = date.today()
        out = self._run(tmp_path)

        day = tmp_path / f"Y{today.year}" / f"M{today.month:02d}" / f"D{today.day:02d}" / "daily.md"
        month = tmp_path / f"Y{today.year}" / f"M{today.month:02d}" / "monthly.md"
        year = tmp_path / f"Y{today.year}" / "yearly.md"

        assert day.is_file()
        assert month.is_file()
        assert year.is_file()
        assert "status: pending" in day.read_text(encoding="utf-8")
        # 占位内容进入 today 展开段.
        assert "(尚未撰写)" in out
