"""
Shared data types for MOSS runtime events.

Types are pure data — schemas, field definitions, self-describing protocols.
They are consumed by core, host, matrix and cli layers.

Layout:
  - types/topics/: TopicModel declarations — typed event schemas published and
    consumed via TopicService at runtime. No wiring lives here.
"""
