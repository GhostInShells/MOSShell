"""The dsh 0.2.0 seam — the node entry, the engine's face, and the channels on top.

The engine lives in ``ghoshell_dsh.v0_2_0``; this package is the MOSS-facing side for
that generation. The parent ``deepseek_harness`` keeps the older implementation, so a
new generation never has to edit the one before it.

Consumers — the ghost's runtime, the hive, ordinary tests — import from here and
nowhere deeper. What is listed below is *this generation's face*, not a mirror of the
engine's module tree: a name is here because MOSS uses it, and importing ``ghoshell_dsh``
directly is how a consumer re-couples itself to a version it does not own.
"""

from ghoshell_dsh.blueprint.models.refs import DshSessionRef
from ghoshell_dsh.blueprint.models.session_events import (
    AgentCancelCause,
    AssistantChunk,
    AssistantMessageEvent,
    RequestHeader,
    SessionEvent,
    SessionEventMeta,
    TokenUsage,
    ToolCallEvent,
    TurnEnd,
    TurnEndReason,
)
from ghoshell_dsh.blueprint.models.sessions import (
    ModelCatalog,
    ModelCatalogModel,
    ModelProviderGroup,
    ModelReasoning,
    ModelReasoningEffort,
)
from ghoshell_dsh.v0_2_0.launcher import (
    DshConnection,
    DshConnectionConfig,
    DshExit,
    DshLauncher,
    DshLauncherConfig,
)
from ghoshell_dsh.v0_2_0.session import DshSession
from ghoshell_dsh.v0_2_0.trajectory import render_transcript
from ghoshell_dsh.v0_2_0.wire import event_from_dict, event_to_dict, model_from_event, model_to_event

from .agent import DshAgentSession
from .hive import DshHive

__all__ = [
    # a node on this generation
    "DshAgentSession",
    "DshHive",
    # process + connection
    "DshLauncher",
    "DshLauncherConfig",
    "DshConnection",
    "DshConnectionConfig",
    "DshExit",
    # one session
    "DshSession",
    # what a session hands back
    "render_transcript",
    "event_from_dict",
    "event_to_dict",
    "model_from_event",
    "model_to_event",
    "DshSessionRef",
    "SessionEvent",
    "SessionEventMeta",
    "TurnEnd",
    "TurnEndReason",
    "AgentCancelCause",
    "AssistantMessageEvent",
    "AssistantChunk",
    "RequestHeader",
    "TokenUsage",
    "ToolCallEvent",
    "ModelCatalog",
    "ModelCatalogModel",
    "ModelProviderGroup",
    "ModelReasoning",
    "ModelReasoningEffort",
]
