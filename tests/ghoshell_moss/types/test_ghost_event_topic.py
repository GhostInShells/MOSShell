"""The ghost event has one schema and two identities.

It is defined once, in the ghost abstraction, and the topic module gives the same
object a second life on the bus. These tests pin the part that would rot silently:
that the two identities really are one schema, and that a consumer which only
narrates can work from ``content`` alone.
"""

from ghoshell_moss.core.blueprint.ghost import GhostEvent
from ghoshell_moss.types.topics.ghost import (
    GHOST_THINKING,
    GhostEventTopic,
)


def test_the_topic_is_the_event_plus_the_envelope():
    """No second copy of the fields: subclassing is what keeps them in step."""
    assert issubclass(GhostEventTopic, GhostEvent)
    event_fields = set(GhostEvent.model_fields)
    topic_fields = set(GhostEventTopic.model_fields)
    assert event_fields <= topic_fields


def test_content_is_the_only_field_a_narrating_surface_needs():
    """A screen draws `content`; the payload stays the producer's business."""
    event = GhostEventTopic(event=GHOST_THINKING, content="thinking about the screen")
    topic = event.to_topic()
    assert topic.data["event"] == GHOST_THINKING
    assert topic.data["content"] == "thinking about the screen"


def test_content_is_optional():
    """An event that has nothing to narrate is still a valid event."""
    assert GhostEventTopic(event="ghost/tool/call").content == ""


def test_the_lifecycle_names_are_namespaced():
    """Consumers dispatch by name; a bare "thinking" would collide across ghosts."""
    for name in (GHOST_THINKING,):
        assert name.startswith("ghost/")
