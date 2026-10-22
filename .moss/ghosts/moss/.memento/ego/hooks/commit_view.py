"""commit_view — put the anchor directory's ground under the note.

The field is the answer to "what did that anchor leave behind": whatever the working area
handed over when it landed, and whatever a hook wrote itself. The directory is empty for
most anchors, and empty is not broken.
"""

import asyncio
import os
from pathlib import Path

from ghoshell_moss.channels.ground_channel import _render_dir


def main() -> None:
    directory = os.environ.get("MEMENTO_COMMIT_DIR", "")
    if not directory:
        return
    workspace = Path(os.environ.get("MEMENTO_ROOT") or directory)
    print(asyncio.run(_render_dir(directory, workspace)))


if __name__ == "__main__":
    main()
