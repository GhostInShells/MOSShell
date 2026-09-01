"""
AI-Native Feature Tracking core functions.

File-system-based feature tracking using FEATURE.md + YAML frontmatter.
This is the logic layer — the CLI is a thin convention enforcer that wraps these functions.

Directory topology: workstreams/<year>/<month>/<name>/FEATURE.md
Path encodes creation date. No archive/move — status changes are frontmatter-only.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

# Open vocabulary: free-form statuses are allowed; these reserved values are the stability contract.
RESERVED_STATUSES = {"draft", "in-progress", "completed", "dropped"}

# Default listing window ("last 2 months") is a rolling 60-day lookback. A naive
# month-bucket count reaches back too shallowly: on the 1st of a month it spans
# only 1 day of the current month plus the prior month.
DEFAULT_RECENT_DAYS = 60


def parse_frontmatter(filepath: str | Path) -> Optional[dict]:
    """Parse YAML frontmatter from a FEATURE.md file.  Returns None if missing or unparseable.

    For error reporting, use :func:`parse_frontmatter_with_error` which returns
    the parse error dict on failure instead of None.
    """
    try:
        import frontmatter
        post = frontmatter.load(str(filepath))
        return dict(post.metadata) if post.metadata else None
    except Exception:
        return None


def parse_frontmatter_with_error(filepath: str | Path) -> tuple[Optional[dict], Optional[dict]]:
    """Parse YAML frontmatter, returning (meta, error) — error is a dict with
    ``path``, ``error``, and ``hint`` keys on parse failure.
    """
    try:
        import frontmatter
        post = frontmatter.load(str(filepath))
        meta = dict(post.metadata) if post.metadata else None
        if meta is None:
            return None, None
        return meta, None
    except Exception as exc:
        fpath = Path(filepath)
        return None, {
            "feature_dir": fpath.parent.name,
            "path": str(fpath),
            "error": str(exc),
            "hint": "YAML frontmatter parse error — check for unquoted colons, "
                    "bad indentation, or invalid YAML syntax in the frontmatter block.",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_feature_dir(features_dir: Path, feature_id: str) -> Optional[Path]:
    """Find a feature directory by name via glob search across year/month trees."""
    hits = sorted(features_dir.glob(f"**/{feature_id}"))
    for hit in hits:
        if hit.is_dir() and (hit / "FEATURE.md").is_file():
            return hit
    return None


def _coerce_date(value) -> Optional[date]:
    """Normalize a frontmatter date value (date, datetime, or ISO string) to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _recent_date(meta: dict) -> Optional[date]:
    """Effective activity date for a feature: ``updated``, falling back to ``created``.

    Returns None when neither field parses as a date; callers then include the
    feature rather than hiding an undatable file.
    """
    for key in ("updated", "created"):
        d = _coerce_date(meta.get(key))
        if d is not None:
            return d
    return None


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def list_features(
    features_dir: str | Path,
    status_filter: Optional[str] = None,
    all_months: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    List features from workstreams/ directory.

    By default, lists features active within a rolling 60-day window (touched
    since today-60d). Pass all_months=True to scan all time.
    Returns (features, parse_errors) — features sorted by updated date descending,
    parse_errors is a list of dicts with feature_dir, path, error, and hint keys.

    Features whose FEATURE.md frontmatter fails to parse are NOT included in the
    feature list — they appear in parse_errors instead.
    """
    workstreams_dir = Path(features_dir) / "workstreams"
    if not workstreams_dir.is_dir():
        return [], []

    cutoff = None if all_months else date.today() - timedelta(days=DEFAULT_RECENT_DAYS)

    # Walk every FEATURE.md. The path only encodes the creation month, so a
    # month-bucket scan would miss workstreams created earlier but still active;
    # filter by each feature's own activity date instead (O(n), fine at this scale).
    feat_dirs = [fm_path.parent for fm_path in sorted(workstreams_dir.glob("**/FEATURE.md"))]

    results = []
    parse_errors = []
    for feat_dir in feat_dirs:
        fm_path = feat_dir / "FEATURE.md"
        meta, err = parse_frontmatter_with_error(fm_path)
        if err is not None:
            parse_errors.append(err)
            continue
        if meta is None:
            continue
        if cutoff is not None:
            recent = _recent_date(meta)
            if recent is not None and recent < cutoff:
                continue
        meta["_feature_dir"] = feat_dir.name
        meta["_feature_path"] = str(feat_dir.relative_to(workstreams_dir))
        if status_filter and meta.get("status") != status_filter:
            continue
        results.append(meta)

    def _sort_key(m: dict) -> str:
        v = m.get("updated", "")
        if isinstance(v, date):
            v = v.isoformat()
        return str(v) or ""

    results.sort(key=_sort_key, reverse=True)
    return results, parse_errors


def get_feature(features_dir: str | Path, feature_id: str) -> tuple[Optional[dict], Optional[dict]]:
    """Get a single feature's frontmatter by id. Searches workstreams/ tree via glob.

    Returns (meta, parse_error). If the feature dir is not found, returns (None, None).
    If the FEATURE.md exists but frontmatter is unparseable, returns (None, error_dict).
    """
    workstreams_dir = Path(features_dir) / "workstreams"
    feat_dir = _find_feature_dir(workstreams_dir, feature_id)
    if feat_dir is None:
        return None, None
    fm_path = feat_dir / "FEATURE.md"
    meta, err = parse_frontmatter_with_error(fm_path)
    if err is not None:
        return None, err
    if meta is not None:
        meta["_feature_dir"] = feature_id
        meta["_feature_path"] = str(feat_dir.relative_to(workstreams_dir))
    return meta, None


# ---------------------------------------------------------------------------
# Mutate
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = """\
---
title: {feature_title}
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: {created_date}
updated: {created_date}
depends: []
milestone:
description: >-
  Brief one-line summary of what this feature is about.
---

# {feature_title}

> Use `moss features set-status {feature_id} <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Why does this feature need to exist? What gap does it fill?

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

## Related

- Depends on: (list feature names)
- Related features: (list feature names)
"""


def create_feature(
    features_dir: str | Path,
    name: str,
    template_path: Optional[str | Path] = None,
) -> Path:
    """
    Create a new feature directory from template.

    Directory: workstreams/<year>/<month>/<name>/ — path encodes creation date, never moves.
    Returns the path to the created FEATURE.md.
    Raises FileExistsError if the feature already exists (anywhere under workstreams/).
    """
    features_dir = Path(features_dir)
    workstreams_dir = features_dir / "workstreams"
    workstreams_dir.mkdir(parents=True, exist_ok=True)

    # Check for duplicate name anywhere in the tree
    if _find_feature_dir(workstreams_dir, name) is not None:
        raise FileExistsError(f"Feature '{name}' already exists under workstreams/")

    today = date.today()
    year = str(today.year)
    month = f"{today.month:02d}"

    feat_dir = workstreams_dir / year / month / name
    feat_dir.mkdir(parents=True)

    today_iso = today.isoformat()
    title = name.replace("-", " ").title()

    if template_path and Path(template_path).is_file():
        content = Path(template_path).read_text(encoding="utf-8")
        content = content.replace("$FEATURE_ID", name)
        content = content.replace("$FEATURE_TITLE", title)
        content = content.replace("$CREATED_DATE", today_iso)
    else:
        content = DEFAULT_TEMPLATE.format(
            feature_id=name,
            feature_title=title,
            created_date=today_iso,
        )

    fm_path = feat_dir / "FEATURE.md"
    fm_path.write_text(content, encoding="utf-8")
    return fm_path


def update_feature_status(
    features_dir: str | Path,
    feature_id: str,
    status: str,
    status_note: Optional[str] = None,
) -> bool:
    """
    Update the status (and updated date) of a feature's frontmatter — in place, no move.

    Terminal statuses (completed/dropped) are just frontmatter updates.
    The file stays where it was created.

    Returns True on success, False if feature not found.
    """
    workstreams_dir = Path(features_dir) / "workstreams"
    feat_dir = _find_feature_dir(workstreams_dir, feature_id)
    if feat_dir is None:
        return False

    fm_path = feat_dir / "FEATURE.md"

    import frontmatter
    post = frontmatter.load(str(fm_path))
    post["status"] = status
    post["updated"] = date.today().isoformat()
    if status_note is not None:
        post["status_note"] = status_note
    fm_path.write_text(frontmatter.dumps(post), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

# _features.py lives at ghoshell_moss/core/codex/_features.py
# Use package root as anchor — more robust than counting from a deep file.
_PKG_ROOT = Path(__file__).resolve().parents[2]  # ghoshell_moss/


def _find_templates_dir() -> Optional[Path]:
    """Find the bundled features template files.

    Priority:
    1. Packaged sibling dir — always available (source or installed).
    2. Source checkout .ai_partners/ — additional fallback for MOSShell dev.
    """
    # 1. Packaged alongside _features.py — most reliable
    p = Path(__file__).resolve().parent / "_features_templates"
    if (p / "README.md").is_file():
        return p
    # 2. Source checkout: repo/.ai_partners/features/
    p = _PKG_ROOT.parents[2] / ".ai_partners" / "features"
    if (p / "README.md").is_file():
        return p
    return None


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_features(project_root: str | Path) -> Path:
    """
    Create the `.ai_partners/features/` skeleton in the given project root.

    Copies README.md, TOPOLOGY.md, and TEMPLATE.md from the bundled templates
    if available, otherwise generates minimal versions.

    Returns the path to the created features directory.
    """
    import shutil

    project_root = Path(project_root)
    features_dir = project_root / ".ai_partners" / "features"

    (features_dir / "workstreams").mkdir(parents=True, exist_ok=True)

    templates_dir = _find_templates_dir()

    readme_src = templates_dir / "README.md" if templates_dir else None
    if readme_src and readme_src.is_file():
        shutil.copy2(str(readme_src), str(features_dir / "README.md"))
    else:
        (features_dir / "README.md").write_text(
            "# MOSS Features\n\nAI-Native Feature Tracking Convention.\n",
            encoding="utf-8",
        )

    template_src = templates_dir / "TEMPLATE.md" if templates_dir else None
    if template_src and template_src.is_file():
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

    topology_src = templates_dir / "TOPOLOGY.md" if templates_dir else None
    if topology_src and topology_src.is_file():
        shutil.copy2(str(topology_src), str(features_dir / "TOPOLOGY.md"))

    return features_dir


# ---------------------------------------------------------------------------
# Future optimization anchor
# ---------------------------------------------------------------------------
# When feature count grows (>>100), list_features() frontmatter parsing becomes
# linear in O(n).  A .state text file (one word per feature) can be introduced
# alongside FEATURE.md for fast status scans without YAML overhead.
# For now (~10 features) frontmatter parsing is more than sufficient.
