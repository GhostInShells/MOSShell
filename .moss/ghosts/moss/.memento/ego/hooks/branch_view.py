"""branch_view — put the branch directory's ground under the branch description.

The field is the answer to "what is this line now": the plan we agreed on, which tasks are
open, where each one stands. Rendered here rather than in the channel, so `view` does not
have to know what a branch directory looks like — the directory says it, this reads it out.
"""

import asyncio

from ghoshell_moss.ground import DefaultGroundSet
from ghoshell_moss.memento.hooks import MementoHookEnv


def main() -> None:
    directory = MementoHookEnv.BRANCH_DIR.path()
    if directory is None:
        return
    template_base = MementoHookEnv.ROOT.path() or directory
    print(asyncio.run(DefaultGroundSet.peek_dir(directory, template_base=template_base)))


if __name__ == "__main__":
    main()
