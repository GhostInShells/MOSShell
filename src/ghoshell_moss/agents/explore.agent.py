"""Exploration agent — explore the repository: locate files, read code, inspect changes.

All tools are read-only and resolve against the repository root. The filesystem
tools (read_file / list_files / glob_files) are synchronous — call them
directly. The git tools (git_status / git_diff) are async — wrap them in
asyncio.run, which returns (code, out, err):

    import asyncio
    code, out, err = asyncio.run(git_status(""))

Orient with glob_files or list_files, then read_file the files that matter.
Use git_status / git_diff to see what changed and how. When you have your
final answer, reply in plain text instead of calling sandbox_exec.
"""

import asyncio

from ghoshell_moss.tools.fs import glob_files, list_files, read_file
from ghoshell_moss.tools.git import git_diff, git_status

__interfaces__ = [read_file, list_files, glob_files, git_status, git_diff]
