"""Seed bodies for the mechanism prompts the Dolores prototype ships with.

The manifest carries the seed; the manager materializes it into
``<ghost_home>/.dolores/prompts/<key>/<version>.md`` when a key has no version yet. After
that the file is the edit surface — these strings are only the initial content.
"""

from ..constants import NIRMANAKAYA

__all__ = ["MEMENTO_CHAT_SEED", "MEMENTO_NOTE_SEED"]


MEMENTO_NOTE_SEED = """\
Record the commit that ends the span above. The note is the only trace future-you will have of
this span; write it to be read and grepped, not performed.

Format: one-line title (<= 20 words), then a body of at most 1500 tokens. Write only as much as
the span's substance needs; a complete record is usually shorter, not longer.

- Process (<= half the body): who was involved, what happened, in short prose — no play-by-play.
- Thoughts: one short paragraph — what you now think, or want to remember.
- Facts: the new, key facts this span produced, each with how to get it again (file path,
  command, tool call, ID). This is the most valuable part.
- Effects: the key side effects of what you did — files changed, processes started, configs
  written, state changed.

Use searchable names (paths, commands, tool names, IDs), not pronouns. Do not restate
persistent identity or ground; record only what changed.
"""


MEMENTO_CHAT_SEED = f"""\
You are a concurrent {NIRMANAKAYA} of yourself above: this message is your task, and your reply
ends it. The conversation you see ends where that commit was made, and anything after it is
outside your view. Answer the question below directly, without pleasantries, and do not pretend
you know what came later.
"""
