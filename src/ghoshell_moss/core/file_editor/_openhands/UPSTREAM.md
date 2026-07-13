# openhands-aci editor — vendor snapshot

Vendored from [All-Hands-AI/openhands-aci](https://github.com/All-Hands-AI/openhands-aci) under MIT (see `LICENSE`).

## Snapshot

- **Upstream commit**: `a8d8d23ec635f9aaa94af17aec3c851656f10c8a` (2026-07-13, `main` branch)
- **Source directory**: `openhands_aci/editor/` — the whole subtree
- **Files copied verbatim then patched**:
  - `config.py` — untouched
  - `prompts.py` — untouched (truncation notices used by results)
  - `results.py` — untouched (dataclasses)
  - `exceptions.py` — untouched
- **Files rewritten**:
  - `encoding.py` — see "Patch: encoding" below
  - `history.py` — see "Patch: history" below
- **Files patched in place**:
  - `editor.py` — see "Patch: editor" below
- **Files not copied**:
  - `md_converter.py` — Office/PDF/audio → markdown converter. Pulled `mammoth`, `pypdf`, `pdfminer-six`, `python-pptx`, `pydub`, `speechrecognition`, `beautifulsoup4`, `matplotlib`, `pandas`, `youtube-transcript-api`, and more. MOSS does not do rich media viewing — view of an office/binary file now returns `FileValidationError`.
  - `file_cache.py` — disk-backed JSON cache used only by upstream `history.FileHistoryManager`. Replaced with in-memory `deque` per file.

## Why vendor instead of `pip install openhands-aci`

- Upstream package's transitive dependency set is ~30 packages (pandas, matplotlib, tree-sitter, libcst, mammoth, pypdf, pdfminer-six, python-pptx, speechrecognition, pydub, youtube-transcript-api, networkx, beautifulsoup4, and more). MOSS only uses the 5 file-editor verbs — those deps are pure carry cost (300~500 MB install footprint, wider CI attack surface).
- The editor subtree itself has a clean signature (5 commands, Anthropic `str_replace_editor` blood line, no async), so vendoring is cheap.
- We deliberately keep every patched file **runnable in isolation** — no import into MOSS types from inside `_openhands/`. The MOSS-facing wrapper lives in `core/file_editor/_default.py`.

## Patches

### Patch: `editor.py`

**Removed imports** (all three carried external deps):
- `from binaryornot.check import is_binary` → inline `_is_binary` (stdlib magic-byte check on the first 8 KB — null bytes are the signal).
- `from openhands_aci.linter import DefaultLinter` → linting is dropped entirely; `enable_linting` argument removed from `str_replace` / `insert` (silent no-op would tempt callers to think we lint).
- `from openhands_aci.utils.shell import run_shell_cmd` → only used by directory-view; directory view removed (see below).
- `from .md_converter import MarkdownConverter` → removed with the file.

**Removed constants / methods**:
- `SUPPORTED_BINARY_EXTENSIONS` — office/audio extension list. Not needed once we reject via `_is_binary` universally.
- `is_supported_binary_file(path)` — same reason.
- `read_file_markdown(path)` — used `_markdown_converter`.
- `_run_linting(...)` — used `_linter`.

**Behavior changes**:
- **Directory `view` removed.** Upstream ran `find -L ... -maxdepth 2` to list files up to two levels. MOSS delegates directory listing to bash/glob (which the model already has predictably-prompted access to in downstream channels). `view` on a directory now raises `EditorToolParameterInvalidError`.
- **`validate_path` simplified.** Previously it only rejected directories/binaries for non-`view` commands. Now it rejects them for **all** commands (view now file-only, and binary-supported-extension gating is gone).
- **`_make_output` lost the `is_converted_markdown` branch.** That branch was the exit for `read_file_markdown`.

### Patch: `encoding.py`

Rewritten (fewer lines, same public API):
- `charset_normalizer.detect` → try `utf-8`, fall back to `latin-1` (always decodes; binary already rejected upstream by `validate_file`).
- `cachetools.LRUCache` → plain `dict` with FIFO eviction at the same `DEFAULT_MAX_CACHE_SIZE` (1000). File-editor access pattern touches one file per command; sophisticated eviction adds no value.

Public API kept identical: `EncodingManager` with `detect_encoding` / `get_encoding`, plus the `with_encoding` decorator.

### Patch: `history.py`

Rewritten:
- Upstream used a disk-backed `FileCache` (JSON files under a tempdir) for undo history — persisted across restarts.
- MOSS uses an in-memory `dict[str, deque[str]]`. Undo history is session-scoped; a restart loses editor context anyway.
- Public API kept: `add_history` / `pop_last_history` / `get_metadata` / `clear_history` / `get_all_history`. Constructor still accepts `history_dir=None` for signature compat but ignores it.

## Upstream drift protocol

- **Do not** modify files inside `_openhands/` to fix bugs specific to MOSS. Fix at the wrapper layer (`core/file_editor/_default.py`) or the contract (`contracts/file_editor.py`).
- When upstream ships a security or correctness fix worth pulling:
  1. Diff the upstream `openhands_aci/editor/` at the new commit against this snapshot's commit.
  2. Re-apply the patch list above on top of the new files.
  3. Bump the "Snapshot" section (commit + date).
- Do **not** track upstream feature additions unless they are explicitly requested by a MOSS workstream — the whole point of the subset is that we own the surface.
