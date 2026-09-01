"""list_features recent-window contract — default is a rolling 60-day window.

Protocol commitment: without ``--all``, ``list_features`` returns workstreams
touched within the last 60 days — measured by each feature's own ``updated``
(``created`` as fallback), NOT by the creation-month bucket in the path. The
path only encodes creation month, so a month-bucket scan reaches back too
shallowly and misses workstreams created earlier but still active.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path


def _write_feature(features_dir: Path, created: date, updated: date) -> str:
    name = f"ws-{created.isoformat()}-{updated.isoformat()}"
    feat_dir = features_dir / "workstreams" / str(created.year) / f"{created.month:02d}" / name
    feat_dir.mkdir(parents=True, exist_ok=True)
    (feat_dir / "FEATURE.md").write_text(
        f"""---
title: {name}
status: draft
priority: P2
created: {created.isoformat()}
updated: {updated.isoformat()}
depends: []
---
# {name}
""",
        encoding="utf-8",
    )
    return name


def test_recent_window_is_rolling_60_days_by_activity(tmp_path):
    from ghoshell_moss.core.codex._features import list_features

    today = date.today()
    fd = tmp_path / "features"

    # Touched within the window.
    name_active = _write_feature(fd, today, today - timedelta(days=1))
    # Created well outside the window but updated recently -> still active.
    # Its path (creation month) is what a month-bucket scan would have skipped.
    name_recent = _write_feature(fd, today - timedelta(days=45), today - timedelta(days=10))
    # Entirely stale — outside the window by activity.
    name_stale = _write_feature(fd, today - timedelta(days=45), today - timedelta(days=61))

    recent, _ = list_features(str(fd))
    assert {f["_feature_dir"] for f in recent} == {name_active, name_recent}

    all_time, _ = list_features(str(fd), all_months=True)
    assert {f["_feature_dir"] for f in all_time} == {name_active, name_recent, name_stale}
