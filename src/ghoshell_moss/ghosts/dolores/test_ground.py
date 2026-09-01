"""Dolores ghost_home 认知场骨架测试 — 独立于 test_dolores.py.

覆盖 ground 装配的协议承诺:
- root 场渐进披露子件类别身份 (frontmatter pin 扫 */GROUND.md), 不穿透进子件内部
- existence 场 @ 装载 purpose/behaviors (冷层法), file pin 装载 identity (warm 帧)
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
        # 三个子件类别身份被渐进披露.
        assert "existence/GROUND.md" in text
        assert "people/GROUND.md" in text
        assert "skills/GROUND.md" in text
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


TIMELINE = STUBS / "existence" / "timeline.py"


class TestTimelineScript:
    def _seed(self, tmp_path):
        daily = tmp_path / "memory" / "daily"
        monthly = tmp_path / "memory" / "monthly"
        daily.mkdir(parents=True)
        monthly.mkdir(parents=True)
        today = date.today().isoformat()
        (daily / f"{today}.md").write_text(
            "---\ndescription: today\n---\n\nbody today\n"
        )
        (daily / "2026-08-31.md").write_text(
            "---\ndescription: yesterday\n---\n\nbody yesterday\n"
        )
        (monthly / "2026-08.md").write_text(
            "---\ndescription: august\n---\n\nbody august\n"
        )

    def test_outputs_today_full_and_recent_summaries(self, tmp_path):
        """timeline 视图: 今天全文, 最近 N 天/月只出 description, 倒序."""
        self._seed(tmp_path)
        out = subprocess.run(
            [sys.executable, str(TIMELINE)],
            env=dict(os.environ, GROUND=str(tmp_path)),
            capture_output=True, text=True,
        ).stdout

        assert "body today" in out               # 今天全文
        assert "2026-08-31: yesterday" in out     # 最近 N 天摘要
        assert "2026-08: august" in out           # 最近 N 月摘要
        assert "body yesterday" not in out        # 昨天只出摘要, 不出全文
