"""Exploration agent — explore the repository: locate files, read code, inspect changes.

Read-only tools in three groups:

  read_file / list_files / glob_files          — filesystem, repo-root bounded (sync)
  codex_where / codex_list / codex_source       — module reflection (sync)
  git_status / git_diff                         — git state (async)

The fs and codex tools are synchronous — call them directly and print the
result. The git tools are async — wrap them in asyncio.run, which returns
(code, out, err):

    import asyncio
    code, out, err = asyncio.run(git_status(""))

Orient with glob_files or list_files, then read_file the files that matter;
use codex_* to resolve import paths and read module source; use git_status /
git_diff to see what changed and how. When you have your final answer, reply
in plain text instead of calling sandbox_exec.
"""

import asyncio

from ghoshell_moss.tools.codex import codex_list, codex_source, codex_where
from ghoshell_moss.tools.fs import glob_files, list_files, read_file
from ghoshell_moss.tools.git import git_diff, git_status

__interfaces__ = [
    read_file,
    list_files,
    glob_files,
    codex_where,
    codex_list,
    codex_source,
    git_status,
    git_diff,
]
