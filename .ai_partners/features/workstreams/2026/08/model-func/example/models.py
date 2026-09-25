"""Response model for utterance end detection benchmark.

Usage:
    moss llms call "I think tomorrow" -r <this_module>:UtteranceEndScore -j
"""

from pydantic import BaseModel, Field


class UtteranceEndScore(BaseModel):
    """0-9 completeness rating of an utterance.

    0 = mid-word / mid-phrase, clearly incomplete.
    5 = ambiguous.
    9 = clearly a complete thought or question.
    """

    score: int = Field(ge=0, le=9, description="completeness rating")
    reason: str = Field(default="", description="brief justification")
