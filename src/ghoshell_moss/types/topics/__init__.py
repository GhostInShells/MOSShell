"""
Topic models for MOSS runtime events.

Typed event schemas — they declare what gets published/consumed via
TopicService at runtime, but carry no wiring. Declared in the types layer
so core, host, matrix and cli all share one schema definition.

Only real topics live here. Test fixtures and experimental schemas stay in
core.concepts.topic and are imported from there directly.
"""

from .audio import AudioSampleTopic, ClauseTopic
from .vision import FaceTopic
