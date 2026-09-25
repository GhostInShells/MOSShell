"""Response model for utterance end detection benchmark — single-token score.

The model outputs ONE 0-9 digit, no reason sentence. Scoring rubric lives in
`rubric.txt` (a strategy variable — placed in instruction or thinking).
"""

from pydantic import BaseModel, Field


class UtteranceEndScore(BaseModel):
    """0-9 completeness rating of an utterance.

    0 = mid-word / mid-phrase, clearly incomplete.
    5 = ambiguous.
    9 = clearly a complete thought or question.
    """

    score: int = Field(ge=0, le=9, description="completeness rating")
