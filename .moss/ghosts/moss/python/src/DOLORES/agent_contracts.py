"""Small imported contracts used by the reflection debug."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExplorerQuestion(BaseModel):
    """A structured question the explorer may answer."""

    text: str = Field(description="The question text.")
    depth: int = Field(default=1, description="How many follow-ups to allow.")


class QuestionPlan:
    """A plain plan with an explicit method surface."""

    def __init__(self, text: str, depth: int = 1) -> None:
        self.text = text
        self.depth = depth

    def expand(self) -> list[str]:
        """Expand the question into a list of probes."""
        return [self.text] * self.depth


def expand_question(text: str, depth: int = 1) -> list[str]:
    """Expand a question into a list of probes."""
    return [text] * depth
