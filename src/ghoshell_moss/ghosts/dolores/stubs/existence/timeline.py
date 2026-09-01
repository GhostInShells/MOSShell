"""timeline — 记忆时间线视图函数 (exec pin, mode=python).

输出倒序视图:
- 今天: 今天的 daily 日记全文
- 最近 N 天: daily 的 description 摘要
- 最近 N 月: monthly 的 description 摘要

只读, 不写。数据不够时输出空段。exec 的 cwd = $GROUND (本场根)。
排序是脚本策略, 不由协议承担 — 协议只管注视与预算。
"""

import os
from datetime import date
from pathlib import Path

GROUND = Path(os.environ.get("GROUND", Path.cwd()))

DAYS = 14
MONTHS = 6


def _description(text: str) -> str:
    """提取 frontmatter 的 description (一行摘要)."""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 4)
    if end == -1:
        return ""
    for line in text[4:end].splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def _list(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted([p for p in d.glob("*.md")], reverse=True)


def main() -> None:
    today = date.today().isoformat()
    daily = _list(GROUND / "memory" / "daily")
    monthly = _list(GROUND / "memory" / "monthly")

    print("## 今天")
    today_file = GROUND / "memory" / "daily" / f"{today}.md"
    if today_file.is_file():
        print(today_file.read_text(encoding="utf-8", errors="replace").rstrip())
    else:
        print("(no diary for today)")

    print("\n## 最近 N 天")
    recent_days = [p for p in daily if p.stem != today][:DAYS]
    for p in recent_days:
        desc = _description(p.read_text(encoding="utf-8", errors="replace"))
        print(f"- {p.stem}: {desc or '(no description)'}")
    if not recent_days:
        print("(none)")

    print("\n## 最近 N 月")
    for p in monthly[:MONTHS]:
        desc = _description(p.read_text(encoding="utf-8", errors="replace"))
        print(f"- {p.stem}: {desc or '(no description)'}")
    if not monthly[:MONTHS]:
        print("(none)")


if __name__ == "__main__":
    main()
