"""
Model-native feature tracking core functions.

File-system-based feature tracking using FEATURE.md + YAML frontmatter.
This is the logic layer — the CLI is a thin convention enforcer that wraps these functions.

Directory topology: workstreams/<year>/<month>/<name>/FEATURE.md
Path encodes creation date. No archive/move — status changes are frontmatter-only.
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, cast, get_args


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ReservedStatus = Literal["draft", "in-progress", "completed", "dropped", "parked"]

# Frontmatter values are raw as parsed from YAML: an ISO date arrives as `date`,
# an empty value as `None`, and any field may be absent entirely.
DateLike = date | str


class WorkstreamMeta(TypedDict, total=False):
    """FEATURE.md frontmatter, plus the source keys injected by the query layer."""
    title: str | None
    status: str  # open vocabulary; reserved values in ReservedStatus
    status_note: str | None
    priority: str | None  # P0 | P1 | P2 | P3 by convention
    created: DateLike | None
    updated: DateLike | None
    depends: list[str] | None
    milestone: str | None
    description: str | None
    _feature_dir: str
    _feature_path: str


class StageMeta(TypedDict, total=False):
    """STAGE.md frontmatter, plus the source key injected by the query layer."""
    status: str  # planning | active | completed | cancelled
    period: str | None
    delivery: str | None
    _stage_dir: str


class RegressionMeta(TypedDict, total=False):
    """REGRESSION.md frontmatter, plus the source key injected by the query layer."""
    title: str | None
    version: int | str
    status: str
    priority: str | None
    created: DateLike | None
    updated: DateLike | None
    scope: str | None
    depends: list[str] | None
    description: str | None
    _regression_dir: str


class ParseError(TypedDict):
    """A FEATURE.md whose frontmatter failed to parse, reported instead of listed."""
    feature_dir: str
    path: str
    error: str
    hint: str


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

# Open vocabulary: free-form statuses are allowed; these reserved values are the stability contract.
RESERVED_STATUSES: frozenset[ReservedStatus] = frozenset(get_args(ReservedStatus))

# A parked workstream is a *formed proposal deliberately set aside* — a technical plan kept on
# file for reference, carrying no attention debt. Quiet statuses are dropped from the listing
# unless explicitly requested by status filter, so they neither pollute the default view nor
# reach the pre-commit `check`. Retrieve them with `--status parked`.
QUIET_STATUSES = {"parked"}

# Default listing window ("last 2 months") is a rolling 60-day lookback. A naive
# month-bucket count reaches back too shallowly: on the 1st of a month it spans
# only 1 day of the current month plus the prior month.
DEFAULT_RECENT_DAYS = 60


def parse_frontmatter(filepath: str | Path) -> Optional[dict[str, Any]]:
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


def parse_frontmatter_with_error(
    filepath: str | Path,
) -> tuple[Optional[dict[str, Any]], Optional[ParseError]]:
    """Parse YAML frontmatter, returning (meta, error) — error is a
    :class:`ParseError` describing the broken file on parse failure.
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
        return None, ParseError(
            feature_dir=fpath.parent.name,
            path=str(fpath),
            error=str(exc),
            hint=(
                "YAML frontmatter parse error — check for unquoted colons, "
                "bad indentation, or invalid YAML syntax in the frontmatter block."
            ),
        )


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


def _coerce_date(value: Any) -> Optional[date]:
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


def _recent_date(meta: WorkstreamMeta) -> Optional[date]:
    """Effective activity date for a feature: ``updated``, falling back to ``created``.

    Returns None when neither field parses as a date; callers then include the
    feature rather than hiding an undatable file.
    """
    for key in ("updated", "created"):
        d = _coerce_date(meta.get(key))
        if d is not None:
            return d
    return None


def _as_workstream_meta(
    raw: dict[str, Any], feature_dir: str, feature_path: str,
) -> WorkstreamMeta:
    """Narrow parsed YAML to :class:`WorkstreamMeta` and attach its source location.

    This is trust, not validation: FEATURE.md is hand-authored, so its field
    types are a convention the file is expected to follow, not an enforced
    contract. Everything below this line can rely on the declared shape.
    """
    meta = cast(WorkstreamMeta, raw)
    meta["_feature_dir"] = feature_dir
    meta["_feature_path"] = feature_path
    return meta


def _as_stage_meta(raw: dict[str, Any], stage_dir: str) -> StageMeta:
    """Narrow parsed YAML to :class:`StageMeta`; trust, not validation."""
    meta = cast(StageMeta, raw)
    meta["_stage_dir"] = stage_dir
    return meta


def _as_regression_meta(raw: dict[str, Any], regression_dir: str) -> RegressionMeta:
    """Narrow parsed YAML to :class:`RegressionMeta`; trust, not validation."""
    meta = cast(RegressionMeta, raw)
    meta["_regression_dir"] = regression_dir
    return meta


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def list_features(
    features_dir: str | Path,
    status_filter: Optional[str] = None,
    all_months: bool = False,
) -> tuple[list[WorkstreamMeta], list[ParseError]]:
    """
    List features from workstreams/ directory.

    By default, lists features active within a rolling 60-day window (touched
    since today-60d). Pass all_months=True to scan all time.
    Returns (features, parse_errors) — features sorted by updated date descending,
    parse_errors is a list of dicts with feature_dir, path, error, and hint keys.

    Quiet statuses (see QUIET_STATUSES) are excluded unless status_filter names
    one explicitly — a parked proposal is retrieved by asking for it, not by
    browsing. all_months widens the time window only; it does not lift this filter.

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

    results: list[WorkstreamMeta] = []
    parse_errors: list[ParseError] = []
    for feat_dir in feat_dirs:
        fm_path = feat_dir / "FEATURE.md"
        raw, err = parse_frontmatter_with_error(fm_path)
        if err is not None:
            parse_errors.append(err)
            continue
        if raw is None:
            continue
        meta = _as_workstream_meta(
            raw, feat_dir.name, str(feat_dir.relative_to(workstreams_dir)),
        )
        if cutoff is not None:
            recent = _recent_date(meta)
            if recent is not None and recent < cutoff:
                continue
        status = meta.get("status")
        if status_filter:
            if status != status_filter:
                continue
        elif status in QUIET_STATUSES:
            continue
        results.append(meta)

    def _sort_key(m: WorkstreamMeta) -> str:
        v = m.get("updated", "")
        if isinstance(v, date):
            v = v.isoformat()
        return str(v) or ""

    results.sort(key=_sort_key, reverse=True)
    return results, parse_errors


def get_feature(
    features_dir: str | Path, feature_id: str,
) -> tuple[Optional[WorkstreamMeta], Optional[ParseError]]:
    """Get a single feature's frontmatter by id. Searches workstreams/ tree via glob.

    Returns (meta, parse_error). If the feature dir is not found, returns (None, None).
    If the FEATURE.md exists but frontmatter is unparseable, returns (None, error_dict).
    """
    workstreams_dir = Path(features_dir) / "workstreams"
    feat_dir = _find_feature_dir(workstreams_dir, feature_id)
    if feat_dir is None:
        return None, None
    fm_path = feat_dir / "FEATURE.md"
    raw, err = parse_frontmatter_with_error(fm_path)
    if err is not None:
        return None, err
    if raw is None:
        return None, None
    meta = _as_workstream_meta(
        raw, feature_id, str(feat_dir.relative_to(workstreams_dir)),
    )
    return meta, None


def list_stages(features_dir: str | Path) -> list[StageMeta]:
    """List stages from stages/*/STAGE.md, reading each STAGE.md frontmatter."""
    stages_dir = Path(features_dir) / "stages"
    if not stages_dir.is_dir():
        return []
    results: list[StageMeta] = []
    for st_path in sorted(stages_dir.glob("*/STAGE.md")):
        if st_path.parent.name == "_template":
            continue
        raw, _ = parse_frontmatter_with_error(st_path)
        if raw is None:
            continue
        results.append(_as_stage_meta(raw, st_path.parent.name))
    return results


def list_regressions(features_dir: str | Path) -> list[RegressionMeta]:
    """List regression sets from regressions/*/REGRESSION.md, reading frontmatter."""
    regressions_dir = Path(features_dir) / "regressions"
    if not regressions_dir.is_dir():
        return []
    results: list[RegressionMeta] = []
    for r_path in sorted(regressions_dir.glob("*/REGRESSION.md")):
        raw, _ = parse_frontmatter_with_error(r_path)
        if raw is None:
            continue
        results.append(_as_regression_meta(raw, r_path.parent.name))
    return results


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

<!-- Record each meaningful design choice. This is what the next model incarnation reads first. -->

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

# This module lives at ghoshell_moss/core/codex/features.py.
# Use package root as anchor — more robust than counting from a deep file.
_PKG_ROOT = Path(__file__).resolve().parents[2]  # ghoshell_moss/


def find_stubs_dir() -> Optional[Path]:
    """Find the bundled feature stubs (scaffolding files).

    Priority:
    1. Packaged sibling dir — always available (source or installed).
    2. Source checkout .ai_partners/features/ — additional fallback for MOSShell dev.
    """
    # 1. Packaged alongside this module — most reliable
    p = Path(__file__).resolve().parent / "_features_stubs"
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

def init_features(features_dir: str | Path | None = None) -> Path:
    """
    Create the features/ skeleton at the given directory (default `$CWD/features`).

    Copies the bundled stubs (full overwrite) and ensures workstreams/ exists.

    Returns the path to the created features directory.
    """
    import shutil

    features_dir = Path(features_dir) if features_dir is not None else Path.cwd() / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    stubs_dir = find_stubs_dir()
    if stubs_dir is not None:
        # Full overwrite: copy every stub file/dir into the target.
        for item in stubs_dir.iterdir():
            dst = features_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dst))
    else:
        (features_dir / "TEMPLATE.md").write_text(
            DEFAULT_TEMPLATE.format(
                feature_id="$FEATURE_ID",
                feature_title="$FEATURE_TITLE",
                created_date="$CREATED_DATE",
            ),
            encoding="utf-8",
        )

    (features_dir / "workstreams").mkdir(parents=True, exist_ok=True)
    return features_dir


# ---------------------------------------------------------------------------
# Future optimization anchor
# ---------------------------------------------------------------------------
# When feature count grows (>>100), list_features() frontmatter parsing becomes
# linear in O(n).  A .state text file (one word per feature) can be introduced
# alongside FEATURE.md for fast status scans without YAML overhead.
# For now (~10 features) frontmatter parsing is more than sufficient.
