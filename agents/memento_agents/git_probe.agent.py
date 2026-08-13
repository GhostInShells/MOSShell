"""Git probe — verify @cli + fs tool import and interface in the agent sandbox.

All tools are read-only and bound to the repo root.

- git_status / git_diff — @cli-wrapped, async. Call via asyncio.run:

      import asyncio
      code, out, err = asyncio.run(git_status(""))

- read_file / list_files / glob_files — synchronous fs tools, call directly.

Return out as plain text.
"""
import asyncio

from ghoshell_moss.tools.fs import glob_files, list_files, read_file
from ghoshell_moss.tools.git import git_diff, git_status

__interfaces__ = [git_status, git_diff, read_file, list_files, glob_files]
