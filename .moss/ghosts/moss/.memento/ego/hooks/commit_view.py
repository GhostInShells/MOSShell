"""commit_view — put the anchor directory's ground under the note.

The field is the answer to "what did that anchor leave behind": whatever the working area
handed over when it landed, and whatever a hook wrote itself. The directory is empty for
most anchors, and empty is not broken.
"""

import asyncio

from ghoshell_moss.ground import DefaultGroundSet
from ghoshell_moss.memento.hooks import MementoHookEnv


def main() -> None:
    directory = MementoHookEnv.COMMIT_DIR.path()
    if directory is None:
        return
    template_base = MementoHookEnv.ROOT.path() or directory
    print(asyncio.run(DefaultGroundSet.peek_dir(directory, template_base=template_base)))


if __name__ == "__main__":
    main()
