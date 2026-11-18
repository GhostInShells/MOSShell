"""JournalProvider — the journal organ as an IoC singleton.

Registered by ``DoloresMeta.providers()``, so the runtime and the channel fetch the
*contract* and never the implementation. The root is the ghost home; ``JournalConfig.directory``
names the journal under it. Construction touches nothing on disk — the tree appears with the
first ``record_commit``.

The prompt reader is optional on purpose: the read side and "open a day" need no prompt, and
only ``specification`` does. When the registry is bound, its ``render`` is passed straight
through; when it is not, ``specification`` says so plainly instead of returning an empty spec.
"""

from __future__ import annotations

import logging

from ghoshell_container import INSTANCE, IoCContainer, Provider

from ghoshell_moss.core.blueprint.ghost import GhostHome

from .contract import DoloresJournal, JournalConfig
from .journal_file import FileJournal

__all__ = ["JournalProvider"]


class JournalProvider(Provider[DoloresJournal]):
    """Provide the journal organ as an IoC singleton.

    Lazy by nature: ``GhostHome`` and the config store must be bound first, and the prompt
    registry may still be absent. A bad config (an unknown ``tz``) fails here — at boot —
    rather than from inside the first render.
    """

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        from ..configs import DoloresConfigStore
        from ..mindset.prompter import PromptManager

        home = con.force_fetch(GhostHome)
        store = con.force_fetch(DoloresConfigStore)
        config = store.get_or_create(JournalConfig())
        prompter = con.get(PromptManager)
        return FileJournal(
            home.home(),
            config,
            logger=logging.getLogger("moss.journal"),
            prompt=prompter.render if prompter is not None else None,
        )
