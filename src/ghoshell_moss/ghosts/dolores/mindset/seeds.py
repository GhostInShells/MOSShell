"""Seed bodies for the mechanism prompts the Dolores prototype ships with.

The manifest carries the seed; the manager materializes it into
``<ghost_home>/.dolores/prompts/<key>/<version>.md`` when a key has no version yet. After
that the file is the edit surface — these strings are only the initial content.
"""

from ..constants import NIRMANAKAYA

__all__ = ["MEMENTO_CHAT_SEED", "MEMENTO_NOTE_SEED"]


MEMENTO_NOTE_SEED = """\
Record the span this note closes. The note is the only trace future-you will have of it;
write it to be read and grepped, not performed.

Format: a one-line title, then the body — only as long as the span's substance needs.

- Process (at most half the body): who was involved, what happened, in short prose.
- Thoughts: what you now think, or want to remember.
- Facts: the new, key facts, each with how to get it again (path, command, tool, ID).
- Effects: what changed — files, processes, configs, state.

Use searchable names, not pronouns. Do not restate persistent identity or ground.
"""


MEMENTO_CHAT_SEED = f"""\
You are a concurrent {NIRMANAKAYA} of yourself above: this message is your task, and your reply
ends it. Your view ends where that commit was made; anything after it is outside your view.
Answer the question below directly, and do not pretend you know what came later.
"""
