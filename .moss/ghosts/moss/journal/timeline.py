"""journal timeline — 时间线视图 + 占位物化 (exec pin, mode=python).

exec 是 ground 里唯一执行代码的动词, 本脚本是它的授权副作用面:

- 物化: 缺当天/当月/当年的占位文档时, 从 template.md 建 (只建不写).
- 呈现: years 存在性 / months 存在性 / today 展开.

cwd = $GROUND (journal 场根). 幂等, 便宜 — 每次 render 都会跑一次进程.
"""

import os
from datetime import date
from pathlib import Path

GROUND = Path(os.environ.get("GROUND", Path.cwd()))
_TEMPLATE = "template.md"

_DEFAULT_TEMPLATE = '---\nsummary: ""\nstatus: pending\n---\n\n(尚未撰写)\n'


def _template_text() -> str:
    tp = GROUND / _TEMPLATE
    if tp.is_file():
        return tp.read_text(encoding="utf-8")
    return _DEFAULT_TEMPLATE


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_template_text(), encoding="utf-8")
    return path


def _dirs(pattern: str) -> list[str]:
    return sorted((p.name for p in GROUND.glob(pattern) if p.is_dir()), reverse=True)


def main() -> None:
    today = date.today()
    year_dir = GROUND / f"Y{today.year}"
    month_dir = year_dir / f"M{today.month:02d}"

    _ensure(year_dir / "yearly.md")
    _ensure(month_dir / "monthly.md")
    day_file = _ensure(month_dir / f"D{today.day:02d}" / "daily.md")

    print("## Years")
    for name in _dirs("Y*"):
        print(f"- {name}")

    print(f"\n## Months — Y{today.year}")
    for name in _dirs(f"Y{today.year}/M*"):
        print(f"- {name}")

    print("\n## Today")
    print(day_file.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
