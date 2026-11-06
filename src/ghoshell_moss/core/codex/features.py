"""
Model-native feature tracking core functions.

File-system-based feature tracking using FEATURE.md + YAML frontmatter.
This is the logic layer — the CLI is a thin convention enforcer that wraps these functions.

Directory topology: workstreams/<year>/<month>/<name>/FEATURE.md
Path encodes creation date. No archive/move — status changes are frontmatter-only.
"""
from datetime import date, datetime, timedelta
from operator import itemgetter
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


class ReviewPerspective(TypedDict):
    """A review perspective doc: ``<features>/review/*.md``, overridable per feature."""
    name: str
    when: str
    description: str
    content: str
    path: Path


class FeatureSource(TypedDict):
    """A resolved workstream: features root, frontmatter, and FEATURE.md path."""
    features_dir: Path
    meta: WorkstreamMeta
    fm_path: Path


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

# Canonical display/aggregation order — derived from the Literal so the two cannot drift.
RESERVED_STATUS_ORDER: list[str] = list(get_args(ReservedStatus))

# A terminal state releases attention on a workstream: it stays listed, but the
# pre-commit `check` stops reminding about it. Not the same as quiet — a dropped
# workstream is terminal *and* still visible.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "dropped"})

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
# Layout
# ---------------------------------------------------------------------------

README_FILENAME = "README.md"
TEMPLATE_FILENAME = "TEMPLATE.md"
SURFACE_FILENAME = "SURFACE.md"

#: Where a project states how it governs issues — a slot the features surface
#: exposes, not a document the framework owns.
ISSUES_CONVENTION = Path("issues") / "SKILL.md"


#: The features tree's directory name, relative to whichever directory holds it.
#: Canonical name first, legacy name second — the order *is* the precedence when
#: one directory carries both.
FEATURES_DIRNAME = Path(".features")
LEGACY_FEATURES_DIRNAME = Path(".ai_partners") / "features"
FEATURES_DIRNAMES: tuple[Path, ...] = (FEATURES_DIRNAME, LEGACY_FEATURES_DIRNAME)

#: A project is the directory tree anchored by this marker: a directory in a
#: plain checkout, a *file* (``gitdir: …``) inside a worktree or submodule — so
#: the test is ``exists()``, never ``is_dir()``.
PROJECT_MARKER = ".git"

#: What makes a directory a features root rather than merely a directory.
FEATURES_ANCHOR = "workstreams"


class FeaturesDirNotFoundError(LookupError):
    """No features tree exists between the working directory and the project home."""


def find_project_home(start: str | Path | None = None) -> Optional[Path]:
    """Walk up from ``start`` to the nearest directory holding ``.git``.

    Nearest wins, so a nested repository resolves to itself rather than to
    whatever contains it. ``None`` when nothing above carries a marker.
    """
    here = (Path(start) if start is not None else Path.cwd()).resolve()
    for path in (here, *here.parents):
        if (path / PROJECT_MARKER).exists():
            return path
    return None


def find_features_dir(start: str | Path | None = None) -> Optional[Path]:
    """Discover the features tree between ``start`` and the project home.

    Two stages, in this order:

    1. find the project home — the nearest ``.git`` above ``start``; without one
       there is no project, so discovery fails;
    2. walk from ``start`` up to that home (inclusive), nearest directory first,
       trying :data:`FEATURES_DIRNAMES` in order at each level.

    A directory counts only when it carries :data:`FEATURES_ANCHOR`. ``None`` is
    returned rather than inventing a path: "not a features project" is a fact the
    caller has to report, not paper over.
    """
    here = (Path(start) if start is not None else Path.cwd()).resolve()
    home = find_project_home(here)
    if home is None:
        return None
    for path in (here, *here.parents):
        for name in FEATURES_DIRNAMES:
            candidate = path / name
            if (candidate / FEATURES_ANCHOR).is_dir():
                return candidate
        if path == home:
            break
    return None


def resolve_features_dir(features_dir: str | Path | None = None) -> Optional[Path]:
    """Resolve the features root for reading.

    An explicit path is taken as-is: it points *at* the features directory and is
    never redirected, never fallen back from. Without one, discovery runs — see
    :func:`find_features_dir`. ``None`` means the working directory is not inside
    a features project.
    """
    if features_dir is not None:
        return Path(features_dir)
    return find_features_dir()


def require_features_dir(features_dir: str | Path | None = None) -> Path:
    """Resolve the features root, or raise :class:`FeaturesDirNotFoundError`.

    The raise is the point: a command that reads features has nothing to say when
    there are no features, and "no workstreams found" in a directory that is not a
    project hides the difference from one that is merely empty.
    """
    fd = resolve_features_dir(features_dir)
    if fd is not None:
        return fd
    cwd = Path.cwd()
    home = find_project_home(cwd)
    names = " or ".join(str(name) for name in FEATURES_DIRNAMES)
    if home is None:
        raise FeaturesDirNotFoundError(
            f"Not inside a project: no {PROJECT_MARKER} in {cwd} or any parent directory."
        )
    raise FeaturesDirNotFoundError(
        f"No features tree between {cwd} and the project home {home} "
        f"(looked for {names}, each holding a {FEATURES_ANCHOR}/ directory)."
    )


def _default_features_dir() -> Path:
    """Where ``features init`` creates a tree when none is named.

    Creation never discovers: a new project gets ``$CWD/.features`` — including
    one that is not a git repository yet.
    """
    return Path.cwd() / FEATURES_DIRNAME


def resolve_feature_path(raw: str | Path) -> Optional[Path]:
    """Resolve a FEATURE.md file or feature-directory path to the FEATURE.md file."""
    p = Path(raw).expanduser()
    if p.is_file() and p.name == "FEATURE.md":
        return p.resolve()
    if p.is_dir() and (p / "FEATURE.md").is_file():
        return (p / "FEATURE.md").resolve()
    return None


def find_features_root(fm_path: str | Path) -> Optional[Path]:
    """Walk up from a FEATURE.md to the features root (the dir holding ``workstreams/``).

    Matched by :data:`FEATURES_DIRNAMES`, so both the canonical and the legacy
    directory name are recognized.
    """
    basenames = {name.name for name in FEATURES_DIRNAMES}
    for parent in Path(fm_path).parents:
        if parent.name in basenames and (parent / FEATURES_ANCHOR).is_dir():
            return parent
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_terminal(status: Optional[str]) -> bool:
    """Whether a workstream status releases attention (see TERMINAL_STATUSES)."""
    return status in TERMINAL_STATUSES


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


def locate_feature(
    feature_part: str, features_dir: str | Path | None = None,
) -> tuple[Optional[FeatureSource], Optional[ParseError]]:
    """Resolve a feature name or a FEATURE.md / feature-directory path.

    Returns (source, parse_error). A name that does not resolve is retried as a
    filesystem path, and the features root is recovered from the path itself.
    Outside a features project the name lookup is skipped — the path retry still
    works, so an absolute FEATURE.md is reachable from anywhere.
    """
    fd = resolve_features_dir(features_dir)
    if fd is not None:
        meta, err = get_feature(str(fd), feature_part)
        if err is not None:
            return None, err
        if meta is not None:
            return FeatureSource(
                features_dir=fd,
                meta=meta,
                fm_path=fd / "workstreams" / meta["_feature_path"] / "FEATURE.md",
            ), None

    fm_path = resolve_feature_path(feature_part)
    if fm_path is None:
        return None, None
    root = find_features_root(fm_path)
    if root is None:
        return None, None
    meta, err = get_feature(str(root), fm_path.parent.name)
    if err is not None:
        return None, err
    if meta is None:
        return None, None
    return FeatureSource(features_dir=root, meta=meta, fm_path=fm_path), None


def count_statuses(features: list[WorkstreamMeta]) -> dict[str, int]:
    """Status distribution — reserved statuses keyed individually, free-form
    aggregated under ``others``."""
    counts: dict[str, int] = {}
    others = 0
    for meta in features:
        status = meta.get("status", "?")
        if status in RESERVED_STATUSES:
            counts[status] = counts.get(status, 0) + 1
        else:
            others += 1
    if others:
        counts["others"] = others
    return counts


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


def _load_review_doc(path: Path) -> Optional[tuple[dict[str, Any], str]]:
    """Load a review doc's (metadata, content). Returns None if unparseable."""
    try:
        import frontmatter
        post = frontmatter.load(str(path))
        return dict(post.metadata) if post.metadata else {}, post.content
    except Exception:
        return None


def discover_perspectives(
    features_dir: str | Path,
    feature_path: str | None = None,
) -> tuple[list[ReviewPerspective], list[Path]]:
    """Discover review perspective docs, per-feature overriding global by name.

    Reads ``<features>/review/*.md`` and, when ``feature_path`` is given,
    ``<features>/workstreams/<feature_path>/review/*.md``. A doc must carry
    ``when`` and ``description`` frontmatter. Returns ``(perspectives, broken)``
    — broken is the list of docs that failed to parse or lack those fields
    (surfaced, not silently skipped).
    """
    root = Path(features_dir)
    review_dirs = [root / "review"]
    if feature_path is not None:
        review_dirs.append(root / "workstreams" / feature_path / "review")

    perspectives: dict[str, ReviewPerspective] = {}
    broken: list[Path] = []
    for review_dir in review_dirs:  # global first; per-feature overrides it
        if not review_dir.is_dir():
            continue
        for doc in sorted(review_dir.glob("*.md")):
            loaded = _load_review_doc(doc)
            if loaded is None:
                broken.append(doc)
                continue
            meta, content = loaded
            if not meta.get("when") or not meta.get("description"):
                broken.append(doc)
                continue
            perspectives[doc.stem] = ReviewPerspective(
                name=doc.stem,
                when=str(meta["when"]),
                description=str(meta["description"]),
                content=content,
                path=doc,
            )
    return sorted(perspectives.values(), key=itemgetter("name")), broken


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
    Uses ``<features>/TEMPLATE.md`` when present, else the built-in template.
    Returns the path to the created FEATURE.md.
    Raises FileExistsError if the feature already exists (anywhere under workstreams/).
    """
    features_dir = Path(features_dir)
    workstreams_dir = features_dir / "workstreams"
    workstreams_dir.mkdir(parents=True, exist_ok=True)

    if template_path is None:
        template_path = features_dir / TEMPLATE_FILENAME

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
    if (p / README_FILENAME).is_file():
        return p
    # 2. Source checkout: repo/.ai_partners/features/
    p = _PKG_ROOT.parents[2] / ".ai_partners" / "features"
    if (p / README_FILENAME).is_file():
        return p
    return None


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_features(features_dir: str | Path | None = None) -> Path:
    """
    Create the features skeleton at the given directory (default ``$CWD/.features``).

    Creation is not discovery: an explicit path is used as given, and with none
    the tree lands in ``$FEATURES_DIRNAME`` under the working directory, git
    repository or not.

    Copies the bundled stubs (full overwrite) and ensures workstreams/ exists.

    Returns the path to the created features directory.
    """
    import shutil

    features_dir = Path(features_dir) if features_dir is not None else _default_features_dir()
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
        (features_dir / TEMPLATE_FILENAME).write_text(
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
