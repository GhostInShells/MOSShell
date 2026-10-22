"""branch_view — put the branch directory's ground under the branch description.

The field is the answer to "what is this line now": the plan we agreed on, which tasks are
open, where each one stands. Rendered here rather than in the channel, so `view` does not
have to know what a branch directory looks like — the directory says it, this reads it out.
"""

import asyncio
import os
from pathlib import Path

from ghoshell_moss.channels.ground_channel import _render_dir


def main() -> None:
    directory = os.environ.get("MEMENTO_BRANCH_DIR", "")
    if not directory:
        return
    workspace = Path(os.environ.get("MEMENTO_ROOT") or directory)
    print(asyncio.run(_render_dir(directory, workspace)))


if __name__ == "__main__":
    main()
