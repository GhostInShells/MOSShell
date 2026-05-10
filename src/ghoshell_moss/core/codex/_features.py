"""
AI-Native Feature Tracking core functions.

File-system-based feature tracking using FEATURE.md + YAML frontmatter.
This is the logic layer — the CLI is a thin convention enforcer that wraps these functions.
"""
import csv
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

import frontmatter


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

VALID_STATUSES = {"draft", "in-progress", "completed", "abandoned", "blocked"}


def parse_frontmatter(filepath: str | Path) -> Optional[dict]:
    """Parse YAML frontmatter from a FEATURE.md file.  Returns None if missing or unparseable."""
    try:
        post = frontmatter.load(str(filepath))
        return dict(post.metadata) if post.metadata else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def list_features(
    features_dir: str | Path,
    status_filter: Optional[str] = None,
) -> list[dict]:
    """
    List all features in the active/ directory.

    Returns a list of frontmatter dicts sorted by priority then created date.
    Each dict includes a '_feature_dir' key with the directory name.
    """
    active_dir = Path(features_dir) / "active"
    if not active_dir.is_dir():
        return []

    results = []
    for feat_dir in sorted(active_dir.iterdir()):
        if not feat_dir.is_dir():
            continue
        fm_path = feat_dir / "FEATURE.md"
        if not fm_path.is_file():
            continue
        meta = parse_frontmatter(fm_path)
        if meta is None:
            continue
        meta["_feature_dir"] = feat_dir.name
        if status_filter and meta.get("status") != status_filter:
            continue
        results.append(meta)

    # Sort by priority (P0 first), then created date
    def _sort_key(m: dict) -> tuple:
        p = m.get("priority", "P9")
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return (order.get(str(p), 9), str(m.get("created", "")))

    results.sort(key=_sort_key)
    return results


def get_feature(features_dir: str | Path, feature_id: str) -> Optional[dict]:
    """Get a single feature's frontmatter by id."""
    active_dir = Path(features_dir) / "active" / feature_id
    if not active_dir.is_dir():
        return None
    fm_path = active_dir / "FEATURE.md"
    if not fm_path.is_file():
        return None
    meta = parse_frontmatter(fm_path)
    if meta is not None:
        meta["_feature_dir"] = feature_id
    return meta


# ---------------------------------------------------------------------------
# Mutate
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = """\
---
id: {feature_id}
title: {feature_title}
status: draft
priority: P2
created: {created_date}
updated: {created_date}
depends: []
milestone:
description: >-
  Brief one-line summary of what this feature is about.
---

# {feature_title}

## Motivation

Why does this feature need to exist? What gap does it fill?

## Scope

What is in scope for this feature? What is explicitly out of scope?

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

<!-- Record important decisions made during development -->

## Implementation Notes

<!-- Implementation details, gotchas, and context for future AI incarnations -->

## Related

- Depends on: (list feature IDs)
- Related features: (list feature IDs)
"""


def create_feature(
    features_dir: str | Path,
    name: str,
    template_path: Optional[str | Path] = None,
) -> Path:
    """
    Create a new feature directory from template.

    Returns the path to the created FEATURE.md.
    Raises FileExistsError if the feature already exists.
    """
    features_dir = Path(features_dir)
    active_dir = features_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    feat_dir = active_dir / name
    if feat_dir.exists():
        raise FileExistsError(f"Feature '{name}' already exists at {feat_dir}")

    feat_dir.mkdir(parents=True)

    today = date.today().isoformat()
    title = name.replace("-", " ").title()

    if template_path and Path(template_path).is_file():
        content = Path(template_path).read_text(encoding="utf-8")
        content = content.replace("$FEATURE_ID", name)
        content = content.replace("$FEATURE_TITLE", title)
        content = content.replace("$CREATED_DATE", today)
    else:
        content = DEFAULT_TEMPLATE.format(
            feature_id=name,
            feature_title=title,
            created_date=today,
        )

    fm_path = feat_dir / "FEATURE.md"
    fm_path.write_text(content, encoding="utf-8")
    return fm_path


def update_feature_status(
    features_dir: str | Path,
    feature_id: str,
    status: str,
) -> bool:
    """
    Update the status (and updated date) of a feature's frontmatter.

    Returns True on success, False if feature not found.
    """
    fm_path = Path(features_dir) / "active" / feature_id / "FEATURE.md"
    if not fm_path.is_file():
        return False

    post = frontmatter.load(str(fm_path))
    post["status"] = status
    post["updated"] = date.today().isoformat()
    fm_path.write_text(frontmatter.dumps(post), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

ARCHIVE_INDEX_HEADER = ["id", "title", "status", "archived_date", "year", "month"]


def archive_feature(features_dir: str | Path, feature_id: str) -> bool:
    """
    Archive a completed or abandoned feature:

    1. Confirm status is completed/abandoned
    2. Move directory to archived/<year>/<month>/<name>/
    3. Append to index/features.csv

    Returns True on success, False if feature not found or not archivable.
    """
    features_dir = Path(features_dir)
    active_dir = features_dir / "active"
    feat_dir = active_dir / feature_id

    if not feat_dir.is_dir():
        return False

    meta = get_feature(str(features_dir), feature_id)
    if meta is None:
        return False

    status = meta.get("status", "")
    if status not in ("completed", "abandoned"):
        return False

    updated = meta.get("updated", date.today())
    if isinstance(updated, date):
        updated = updated.isoformat()
    else:
        updated = str(updated)
    try:
        year, month, _ = updated.split("-")
    except ValueError:
        year = str(date.today().year)
        month = str(date.today().month).zfill(2)

    # Move to archived
    archived_target = features_dir / "archived" / year / month / feature_id
    archived_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(feat_dir), str(archived_target))

    # Append to index
    index_dir = features_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "features.csv"
    write_header = not index_path.exists()
    with open(index_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(ARCHIVE_INDEX_HEADER)
        writer.writerow([
            feature_id,
            meta.get("title", ""),
            status,
            date.today().isoformat(),
            year,
            month,
        ])

    return True


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

FEATURES_README_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / ".ai_partners" / "features" / "README.md"


def init_features(project_root: str | Path) -> Path:
    """
    Create the `.ai_partners/features/` skeleton in the given project root.

    Copies README.md and TEMPLATE.md from MOSShell's own features directory
    if available, otherwise generates minimal versions.

    Returns the path to the created features directory.
    """
    project_root = Path(project_root)
    features_dir = project_root / ".ai_partners" / "features"

    for sub in ("active", "archived", "index"):
        (features_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy specification and template from MOSShell's own features
    moss_features = Path(__file__).resolve().parents[5] / ".ai_partners" / "features"

    readme_src = moss_features / "README.md"
    if readme_src.is_file():
        shutil.copy2(str(readme_src), str(features_dir / "README.md"))
    else:
        (features_dir / "README.md").write_text(
            "# MOSS Features\n\nAI-Native Feature Tracking Convention.\n",
            encoding="utf-8",
        )

    template_src = moss_features / "TEMPLATE.md"
    if template_src.is_file():
        shutil.copy2(str(template_src), str(features_dir / "TEMPLATE.md"))
    else:
        (features_dir / "TEMPLATE.md").write_text(
            DEFAULT_TEMPLATE.format(
                feature_id="$FEATURE_ID",
                feature_title="$FEATURE_TITLE",
                created_date="$CREATED_DATE",
            ),
            encoding="utf-8",
        )

    return features_dir
