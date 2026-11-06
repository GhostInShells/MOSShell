"""The blueprint — pure contracts, the shape a consumer programs against.

Two ingredients, and nothing else:

- **data models** (`blueprint.models`) — values that cross the boundary: session events, refs,
  anchors' neighbours, RPC envelopes, verb payloads, attachments, a session's state. One
  definition, so no generation can quietly invent a second.
- **contracts** (this package's modules) — abstract, transport-free, config-free.

What belongs here is decided by one question: *given only the blueprint, can a consumer do the
basic thing?* If the shape only parses when you also know which transport or which dsh version
answers, it is not a contract yet.

What does **not** belong here, on purpose:

- configs, ports, profiles, tokens, probe paths — they are concrete. A launcher required to take
  them here would have made every future transport edit the constitution.
- anything that exists to make one implementation convenient. Convenience is the scaffolding
  layer's business, and that layer may be rewritten or replaced wholesale.

### How the pieces relate

```
Launcher              owns a lifetime; start() hands back a SessionSource
  SessionSource       the minimum: list, create, attach conversations
    Connection        dsh's own surface: workspaces, event fan-out, close
      Client          the protocol layer underneath: typed calls, plugin routes, rpc errors
    DshController     the reinforced surface: governed listing, create/fork with a spec,
                      rename/archive, workspace governance
      AgentSession            our verbs: run, anchor, chat_off_record
        DshAgentSession       dsh's verbs: prompt, cancel, history, events, state
```

Two relationships are worth stating because they are easy to get wrong:

- **`Connection` and `DshController` are parallel faces, not a stack.** Neither inherits the
  other, and a controller does not promise to hold a connection. A concrete generation (v0.2.0
  will be the first) implements **both**, so a caller takes the native verbs or the governed ones
  from the same object and never loses one by asking for the other.
- **`AgentSession` sits on `DshAgentSession`.** The lower one speaks dsh's verbs as dsh states
  them; the upper one adds ours. Mixing them in a single contract is how a protocol turns into a
  framework.

`Client` is a contract rather than an implementation detail because our plugin patches call it:
the anchor call and the session read are plugin routes, and a route is not a domain verb.

**Not wired to anything yet.** Nothing outside this package imports it; the v0.2.0 implementation
still speaks to its own modules. That is deliberate — the shape is settled first, and the
implementation is pointed at it in the next step.
"""

from .agent import AgentSession, ChatReply, RunResult, SessionSource
from .anchor import DshSessionAnchor
from .client import Client, DshRpcError
from .connection import Connection
from .controller import DshController, ForkSpec, SessionSpec
from .launcher import Exit, Launcher
from .session import WILDCARD_EVENT, DshAgentSession

__all__ = [
    "WILDCARD_EVENT",
    "AgentSession",
    "ChatReply",
    "Client",
    "Connection",
    "DshAgentSession",
    "DshController",
    "DshRpcError",
    "DshSessionAnchor",
    "Exit",
    "ForkSpec",
    "Launcher",
    "RunResult",
    "SessionSource",
    "SessionSpec",
]
