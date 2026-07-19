"""Conservative token estimation and context-overflow detection for Aurelius.

Aurelius has no provider tokenizer wired in. Every model call must fit the input
inside ``context_window - max_output_tokens``; the ``window(detail_n, summary_m)``
knobs only bound *frame count*, not tokens, so a few multimodal or verbose frames
can silently blow the input budget and fail the request outright.

This module gives a deliberately *over*-estimating token counter (bias toward
compressing early — a false-positive compaction is cheap, a blown request is not)
plus a provider-agnostic overflow-error classifier for the retry fallback. The
estimate is intentionally crude: char/CJK divisor for text, a flat nominal cost
per image (we do NOT count the base64 payload length — that would over-estimate
an image by ~1000x and starve the real text budget).
"""

from pydantic_ai.messages import (
    ModelMessage,
    TextPart,
    UserPromptPart,
)

__all__ = [
    "IMAGE_NOMINAL_TOKENS",
    "estimate_history_tokens",
    "estimate_tokens",
    "is_context_overflow",
]

# One image costs a flat nominal budget regardless of resolution. Real vision
# token cost varies by provider/tiling; this is a conservative placeholder that
# keeps images from being free without letting their base64 length dominate.
IMAGE_NOMINAL_TOKENS = 1600

# Latin text trends ~4 chars/token; CJK trends ~1.5 chars/token. Use the smaller
# CJK-leaning divisor as the single global rule so mixed/Chinese content (this
# project's daily language) is never under-counted.
_CHARS_PER_TOKEN = 2.5

# Flat per-message structural overhead (role markers, delimiters) the provider
# adds around each turn — small, but non-zero across a long history.
_MESSAGE_OVERHEAD_TOKENS = 8

_OVERFLOW_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
    "input is too long",
)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _estimate_content_tokens(content: object) -> int:
    # Text-bearing content (TextContent/TextPart carry ``.content``); images and
    # other modalities are billed the flat nominal cost, never by payload length.
    text = getattr(content, "content", None)
    if isinstance(text, str):
        return estimate_text_tokens(text)
    return IMAGE_NOMINAL_TOKENS


def _estimate_part_tokens(part: object) -> int:
    if isinstance(part, TextPart):
        return estimate_text_tokens(part.content)
    if isinstance(part, UserPromptPart):
        content = part.content
        if isinstance(content, str):
            return estimate_text_tokens(content)
        return sum(_estimate_content_tokens(item) for item in content)
    # Unknown part: fall back to its string form so it is never counted as free.
    return estimate_text_tokens(str(getattr(part, "content", "")))


def estimate_tokens(message: ModelMessage) -> int:
    parts = getattr(message, "parts", None) or []
    total = _MESSAGE_OVERHEAD_TOKENS
    for part in parts:
        total += _estimate_part_tokens(part)
    return total


def estimate_history_tokens(history: list[ModelMessage]) -> int:
    return sum(estimate_tokens(message) for message in history)


def is_context_overflow(error: BaseException) -> bool:
    """True if an error reads like an input-context-window overflow.

    pydantic-ai 2.x surfaces no dedicated overflow exception type across
    providers, so we match on normalized error text. This must NOT match the
    output-side "token limit (provider default) exceeded" (a max_tokens issue,
    handled separately) nor attention-abort errors (never overflow-worded).
    """
    text = str(error).casefold()
    return any(marker in text for marker in _OVERFLOW_MARKERS)
