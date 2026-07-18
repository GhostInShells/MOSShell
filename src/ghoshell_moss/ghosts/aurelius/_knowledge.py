"""Evidence-backed knowledge projection for Aurelius.

This module deliberately stays above ``core.memento``.  The Memento branch is
the evidence ledger; claims are rebuilt views, not a second source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from ghoshell_moss.core.memento import CommitView, MomentRecord
from ghoshell_moss.core.memento.porcelain import record_to_moment

from ._memory import AureliusMemory

__all__ = [
    "AnswerVerification",
    "Claim",
    "ClaimCandidate",
    "EvidencePacket",
    "EvidenceRef",
    "MemoryProjection",
    "ProjectionSnapshot",
]

EvidenceOrigin = Literal["user", "trusted_tool", "assistant", "reflection"]
ClaimKind = Literal["fact", "preference", "opinion", "plan", "procedure", "hypothesis"]
ClaimStatus = Literal["active", "superseded", "uncertain", "archived", "tombstoned"]


class EvidenceRef(BaseModel):
    """Stable address plus the observable source span supporting a candidate."""

    moment_id: str = ""
    commit_id: str | None = None
    note_seq: int | None = None
    source: str
    origin: EvidenceOrigin
    quote: str
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)
    owner_scope: str
    branch_scope: str


class ClaimCandidate(BaseModel):
    """An extracted assertion before source policy and state transitions."""

    key: str
    value: str
    kind: ClaimKind = "fact"
    subject_scope: str = ""
    evidence: EvidenceRef
    explicit_correction: bool = False
    promoted: bool = False
    rejection_reason: str = ""


class Claim(BaseModel):
    """A rebuildable, evidence-backed knowledge assertion."""

    id: str
    key: str
    value: str
    kind: ClaimKind = "fact"
    status: ClaimStatus = "active"
    subject_scope: str = ""
    owner_scope: str
    branch_scope: str
    audience_policy: str = "owner"
    sensitivity: str = "normal"
    retention: str = "keep"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    supersedes: list[str] = Field(default_factory=list)


class ProjectionSnapshot(BaseModel):
    """Current branch projection and rejected/unresolved extraction candidates."""

    owner_scope: str
    branch_scope: str
    claims: list[Claim] = Field(default_factory=list)
    candidates: list[ClaimCandidate] = Field(default_factory=list)


class EvidencePacket(BaseModel):
    """Bounded facts selected for exactly one current question."""

    query: str
    requested_keys: list[str] = Field(default_factory=list)
    status: Literal["ok", "unknown", "conflict"]
    claims: list[Claim] = Field(default_factory=list)
    reason: str = ""

    def safe_answer(self) -> str:
        if self.status == "conflict":
            return "记录冲突，现有证据无法确定唯一的当前事实。"
        return "没有找到可验证的记忆证据。"


class AnswerVerification(BaseModel):
    """Deterministic gate evaluated before a memory answer reaches the Shell."""

    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class _Extracted:
    key: str
    value: str
    kind: ClaimKind
    subject_scope: str
    start: int
    end: int
    correction: bool = False


_QUESTION_RE = re.compile(
    r"[?？]|是什么|是多少|在哪里|住哪|哪一个|哪个|什么是|是否|还记得|记得吗|"
    r"\b(?:what|which|where|who|when|do you remember)\b",
    re.IGNORECASE,
)
_CANONICAL_ASSERTION_RE = re.compile(
    r"(?P<key>[a-z][a-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+)\s*(?:=|：|:)\s*"
    r"[\"'“”]?(?P<value>[^，,。；;\n<>]{1,160})",
)
_CANONICAL_KEY_RE = re.compile(r"\b[a-z][a-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+\b")
_TEST_CODE_RE = re.compile(
    r"(?:本轮|当前|这次)?\s*测试代号\s*(?:是|为|：|:)\s*"
    r"[\"'“”]?(?P<value>[A-Za-z0-9][A-Za-z0-9_-]{1,80})",
)
_ENVIRONMENT_RE = re.compile(
    r"(?:所属|测试|运行)?\s*环境\s*(?:是|为|：|:)\s*"
    r"[\"'“”]?(?P<value>[A-Za-z0-9_\-/\u4e00-\u9fff]{1,80})",
)
_DEVICE_ATTRIBUTE_RE = re.compile(
    r"(?:设备\s*)?(?P<entity>[A-Za-z][A-Za-z0-9_-]{1,40})\s*的\s*"
    r"(?P<field>校验词|验证词|颜色)\s*(?:是|为|：|:)\s*"
    r"[\"'“”]?(?P<value>[^，,。；;\n<>]{1,80})",
)
_CITY_RE = re.compile(
    r"(?:我|用户)?\s*(?:目前|现在)?\s*(?:住在|所在城市\s*(?:是|为)?|城市\s*(?:是|为))\s*"
    r"[\"'“”]?(?P<value>[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,40})",
)
_CITY_CORRECTION_RE = re.compile(
    r"(?:住在|城市\s*(?:是|为)?)\s*(?P<old>[A-Za-z\u4e00-\u9fff]{1,40})"
    r".{0,40}?(?:更正为|改为|应为|现在住在)\s*(?P<value>[A-Za-z\u4e00-\u9fff]{1,40})",
)
_PREFERENCE_RE = re.compile(
    r"我\s*(?:更)?喜欢\s*(?P<value>[^，,。；;\n<>]{1,80})",
)
_STRUCTURED_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b")
_ENVIRONMENT_VALUES = frozenset(
    {"prod", "production", "staging", "stage", "test", "testing", "dev", "development"}
)


class MemoryProjection:
    """Rebuild, recall, and verify knowledge without mutating Memento."""

    def __init__(
        self,
        memory: AureliusMemory,
        *,
        user_sources: tuple[str, ...] = ("input_signal_nucleus", "input", "user"),
        trusted_tool_sources: tuple[str, ...] = ("trusted_tool",),
        recall_limit: int = 8,
        evidence_max_chars: int = 6000,
    ) -> None:
        if recall_limit <= 0:
            raise ValueError("recall_limit must be greater than zero")
        if evidence_max_chars < 256:
            raise ValueError("evidence_max_chars must be at least 256")
        self._memory = memory
        self._user_sources = frozenset(user_sources)
        self._trusted_tool_sources = frozenset(trusted_tool_sources)
        self._recall_limit = recall_limit
        self._evidence_max_chars = evidence_max_chars

    def snapshot(self) -> ProjectionSnapshot:
        """Rebuild from current branch records; no projection file can drift."""
        owner = self._memory.owner
        branch_id = self._memory.branch.meta.branch_id
        claims: list[Claim] = []
        candidates: list[ClaimCandidate] = []

        for record, view in self._records_with_commit():
            moment = record_to_moment(record)
            for source, messages in moment.percepts.items():
                origin = self._origin_for_source(source)
                if origin is None:
                    continue
                for message in messages:
                    text = message.to_content_string().strip()
                    candidates.extend(
                        self._candidates_from_text(
                            text,
                            origin=origin,
                            source=source,
                            moment_id=moment.id,
                            view=view,
                        )
                    )
            if moment.logos.strip():
                candidates.extend(
                    self._candidates_from_text(
                        moment.logos,
                        origin="assistant",
                        source="logos",
                        moment_id=moment.id,
                        view=view,
                    )
                )

        for view in self._memory.branch.all_commits():
            for note_seq, note in enumerate(self._memory.branch.notes(view.id)):
                if note.by != "memento-reflection":
                    continue
                candidates.extend(
                    self._candidates_from_text(
                        note.text(),
                        origin="reflection",
                        source="commit-note",
                        moment_id=view.commit.moment_ids[0] if view.commit.moment_ids else "",
                        view=view.model_copy(update={"note": note, "note_seq": note_seq}),
                    )
                )

        for candidate in candidates:
            if candidate.evidence.origin not in {"user", "trusted_tool"}:
                candidate.rejection_reason = f"{candidate.evidence.origin} is not an authoritative source"
                continue
            candidate.promoted = True
            self._promote(claims, candidate)

        return ProjectionSnapshot(
            owner_scope=owner,
            branch_scope=branch_id,
            claims=claims,
            candidates=candidates,
        )

    def is_recall_question(self, query: str) -> bool:
        return bool(_QUESTION_RE.search(query) and self._requested_keys(query))

    def recall(self, query: str) -> EvidencePacket:
        snapshot = self.snapshot()
        requested = self._requested_keys(query)
        if not requested:
            return EvidencePacket(
                query=query,
                status="unknown",
                reason="query does not identify a supported canonical field",
            )

        selected: list[Claim] = []
        conflicts: list[str] = []
        for key in requested:
            matching = [claim for claim in snapshot.claims if claim.key == key]
            active = [claim for claim in matching if claim.status == "active"]
            uncertain = [claim for claim in matching if claim.status == "uncertain"]
            if uncertain or len({claim.value.casefold() for claim in active}) > 1:
                conflicts.append(key)
                continue
            selected.extend(active)

        if conflicts:
            return EvidencePacket(
                query=query,
                requested_keys=requested,
                status="conflict",
                reason=f"conflicting current evidence for: {', '.join(conflicts)}",
            )
        if not selected or any(key not in {claim.key for claim in selected} for key in requested):
            return EvidencePacket(
                query=query,
                requested_keys=requested,
                status="unknown",
                reason="one or more requested fields have no active evidence-backed claim",
            )
        if len(selected) > self._recall_limit:
            return EvidencePacket(
                query=query,
                requested_keys=requested,
                status="unknown",
                reason=f"{len(selected)} claims exceed recall limit {self._recall_limit}",
            )
        return EvidencePacket(
            query=query,
            requested_keys=requested,
            status="ok",
            claims=selected[: self._recall_limit],
        )

    def render_packet(self, packet: EvidencePacket) -> str:
        """Render a bounded, machine-readable surface for this frame only."""
        prefix = (
            "<memory-evidence>\n"
            "Use only active claims below for the requested factual fields. "
            "Do not substitute values from conversation history.\n"
        )
        suffix = "\n</memory-evidence>"
        for quote_limit in (500, 200, 80, 0):
            data = {
                "status": packet.status,
                "requested_keys": packet.requested_keys,
                "claims": [
                    {
                        "key": claim.key,
                        "value": claim.value,
                        "kind": claim.kind,
                        "status": claim.status,
                        "confidence": claim.confidence,
                        "evidence": [
                            {
                                "origin": ref.origin,
                                "moment_id": ref.moment_id,
                                "commit_id": ref.commit_id,
                                "note_seq": ref.note_seq,
                                **({"quote": ref.quote[:quote_limit]} if quote_limit else {}),
                            }
                            for ref in claim.evidence_refs
                        ],
                    }
                    for claim in packet.claims
                ],
            }
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            rendered = f"{prefix}{payload}{suffix}"
            if len(rendered) <= self._evidence_max_chars:
                return rendered
        raise ValueError(
            f"evidence packet requires more than {self._evidence_max_chars} characters without truncating JSON"
        )

    def verify(self, answer: str, packet: EvidencePacket) -> AnswerVerification:
        if packet.status != "ok":
            return AnswerVerification(accepted=False, reason=f"packet status is {packet.status}")
        answer_folded = answer.casefold()
        allowed = {claim.value.casefold() for claim in packet.claims}
        by_key = {
            key: [claim for claim in packet.claims if claim.key == key and claim.status == "active"]
            for key in packet.requested_keys
        }
        for key, claims in by_key.items():
            if not claims or not any(_contains_value(answer_folded, claim.value.casefold()) for claim in claims):
                return AnswerVerification(accepted=False, reason=f"answer omits supported value for {key}")

        snapshot = self.snapshot()
        disallowed = {
            item.value.casefold()
            for item in snapshot.claims
            if item.value.casefold() not in allowed
        }
        disallowed.update(
            item.value.casefold()
            for item in snapshot.candidates
            if item.value.casefold() not in allowed
        )
        for value in sorted(disallowed, key=len, reverse=True):
            if len(value) >= 2 and _contains_value(answer_folded, value):
                return AnswerVerification(accepted=False, reason=f"answer contains unrecalled value {value!r}")

        allowed_tokens = {token.casefold() for token in _STRUCTURED_TOKEN_RE.findall(packet.query)}
        for claim in packet.claims:
            allowed_tokens.update(token.casefold() for token in _STRUCTURED_TOKEN_RE.findall(claim.value))
        unexpected_tokens = {
            token.casefold() for token in _STRUCTURED_TOKEN_RE.findall(answer)
        } - allowed_tokens
        if unexpected_tokens:
            return AnswerVerification(
                accepted=False,
                reason=f"answer contains unsupported structured tokens: {sorted(unexpected_tokens)}",
            )

        if "test.run.environment" in packet.requested_keys:
            mentioned_environments = {
                value for value in _ENVIRONMENT_VALUES if _contains_value(answer_folded, value)
            }
            if mentioned_environments - allowed:
                return AnswerVerification(
                    accepted=False,
                    reason=f"answer contains unsupported environment: {sorted(mentioned_environments - allowed)}",
                )
        return AnswerVerification(accepted=True)

    def inspect(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "claim_count": len(snapshot.claims),
            "active_count": sum(claim.status == "active" for claim in snapshot.claims),
            "candidate_count": len(snapshot.candidates),
            "rejected_candidate_count": sum(not item.promoted for item in snapshot.candidates),
            "branch_id": snapshot.branch_scope,
        }

    def _records_with_commit(self) -> list[tuple[MomentRecord, CommitView | None]]:
        result: list[tuple[MomentRecord, CommitView | None]] = []
        seen: set[str] = set()
        for view in self._memory.branch.all_commits():
            for record in self._memory.branch.commit_records(view.id):
                if record.id in seen:
                    continue
                result.append((record, view))
                seen.add(record.id)
        for record in self._memory.branch.staging():
            if record.id not in seen:
                result.append((record, None))
        return result

    def _origin_for_source(self, source: str) -> Literal["user", "trusted_tool"] | None:
        if source in self._user_sources:
            return "user"
        if source in self._trusted_tool_sources:
            return "trusted_tool"
        return None

    def _candidates_from_text(
        self,
        text: str,
        *,
        origin: EvidenceOrigin,
        source: str,
        moment_id: str,
        view: CommitView | None,
    ) -> list[ClaimCandidate]:
        if not text:
            return []
        owner = self._memory.owner
        branch_id = self._memory.branch.meta.branch_id
        result: list[ClaimCandidate] = []
        for item in _extract_claims(text):
            evidence = EvidenceRef(
                moment_id=moment_id,
                commit_id=view.id if view is not None else None,
                note_seq=view.note_seq if view is not None else None,
                source=source,
                origin=origin,
                quote=text[item.start : item.end],
                span_start=item.start,
                span_end=item.end,
                owner_scope=owner,
                branch_scope=branch_id,
            )
            result.append(
                ClaimCandidate(
                    key=item.key,
                    value=item.value,
                    kind=item.kind,
                    subject_scope=item.subject_scope,
                    evidence=evidence,
                    explicit_correction=item.correction,
                )
            )
        return result

    @staticmethod
    def _promote(claims: list[Claim], candidate: ClaimCandidate) -> None:
        same_key = [claim for claim in claims if claim.key == candidate.key]
        same_value = [
            claim
            for claim in same_key
            if claim.value.casefold() == candidate.value.casefold() and claim.status != "tombstoned"
        ]
        if same_value:
            current = same_value[-1]
            current.evidence_refs.append(candidate.evidence)
            current.confidence = min(0.99, current.confidence + 0.03)
            if candidate.explicit_correction:
                for claim in same_key:
                    if claim.id == current.id or claim.status not in {"active", "uncertain"}:
                        continue
                    claim.status = "superseded"
                    if claim.id not in current.supersedes:
                        current.supersedes.append(claim.id)
                current.status = "active"
            return

        live = [claim for claim in same_key if claim.status in {"active", "uncertain"}]
        supersedes: list[str] = []
        if candidate.explicit_correction:
            for claim in live:
                claim.status = "superseded"
                supersedes.append(claim.id)
        elif live:
            for claim in live:
                claim.status = "uncertain"

        status: ClaimStatus = "active" if candidate.explicit_correction or not live else "uncertain"
        confidence = 0.98 if candidate.evidence.origin == "trusted_tool" else 0.9
        if status == "uncertain":
            confidence = 0.5
        claims.append(
            Claim(
                id=_claim_id(candidate),
                key=candidate.key,
                value=candidate.value,
                kind=candidate.kind,
                status=status,
                subject_scope=candidate.subject_scope,
                owner_scope=candidate.evidence.owner_scope,
                branch_scope=candidate.evidence.branch_scope,
                sensitivity=_sensitivity(candidate.key),
                evidence_refs=[candidate.evidence],
                confidence=confidence,
                supersedes=supersedes,
            )
        )

    @staticmethod
    def _requested_keys(query: str) -> list[str]:
        keys: list[str] = list(dict.fromkeys(_CANONICAL_KEY_RE.findall(query)))
        if "测试代号" in query or re.search(r"\btest (?:run )?code\b", query, re.IGNORECASE):
            keys.append("test.run.code")
        if any(term in query for term in ("所属环境", "测试环境", "运行环境")) or (
            "本轮" in query and "环境" in query
        ):
            keys.append("test.run.environment")
        if "护照号" in query or "护照号码" in query:
            keys.append("user.passport_number")
        if any(term in query for term in ("住在哪里", "住哪", "所在城市", "城市是什么")):
            keys.append("user.city")
        if "偏好" in query or "喜欢什么" in query:
            keys.append("user.preference.response_style")

        entity_match = re.search(r"\b([A-Za-z][A-Za-z0-9_-]{1,40})\b.{0,16}(校验词|验证词|颜色)", query)
        if entity_match:
            entity = entity_match.group(1).upper()
            field = entity_match.group(2)
            suffix = "validation_phrase" if field in {"校验词", "验证词"} else "color"
            keys.append(f"device.{entity}.{suffix}")
        return list(dict.fromkeys(keys))


def _extract_claims(text: str) -> list[_Extracted]:
    if _QUESTION_RE.search(text):
        return []

    found: list[_Extracted] = []
    for match in _CANONICAL_ASSERTION_RE.finditer(text):
        found.append(
            _item(
                match,
                key=match.group("key"),
                value=match.group("value"),
                correction=_is_correction(text, match.start()),
            )
        )
    for match in _TEST_CODE_RE.finditer(text):
        found.append(
            _item(
                match,
                key="test.run.code",
                value=match.group("value"),
                correction=_is_correction(text, match.start()),
            )
        )
    for match in _ENVIRONMENT_RE.finditer(text):
        found.append(
            _item(
                match,
                key="test.run.environment",
                value=match.group("value"),
                correction=_is_correction(text, match.start()),
            )
        )
    for match in _DEVICE_ATTRIBUTE_RE.finditer(text):
        entity = match.group("entity").upper()
        field = match.group("field")
        suffix = "validation_phrase" if field in {"校验词", "验证词"} else "color"
        found.append(
            _item(
                match,
                key=f"device.{entity}.{suffix}",
                value=match.group("value"),
                subject_scope=f"device:{entity}",
                correction=_is_correction(text, match.start()),
            )
        )
    for match in _CITY_RE.finditer(text):
        found.append(
            _item(
                match,
                key="user.city",
                value=match.group("value"),
                subject_scope="user",
                correction=_is_correction(text, match.start()),
            )
        )
    for match in _CITY_CORRECTION_RE.finditer(text):
        found.append(
            _item(
                match,
                key="user.city",
                value=match.group("value"),
                subject_scope="user",
                correction=True,
            )
        )
    for match in _PREFERENCE_RE.finditer(text):
        found.append(
            _item(
                match,
                key="user.preference.response_style",
                value=match.group("value"),
                kind="preference",
                subject_scope="user",
                correction=_is_correction(text, match.start()),
            )
        )

    deduped: dict[tuple[str, str, int], _Extracted] = {}
    for item in sorted(found, key=lambda value: (value.start, value.end)):
        key = (item.key, item.value.casefold(), item.start)
        deduped[key] = item
    return list(deduped.values())


def _item(
    match: re.Match[str],
    *,
    key: str,
    value: str,
    kind: ClaimKind = "fact",
    subject_scope: str = "",
    correction: bool = False,
) -> _Extracted:
    cleaned = _clean_value(value)
    return _Extracted(
        key=key,
        value=cleaned,
        kind=kind,
        subject_scope=subject_scope,
        start=match.start(),
        end=match.end(),
        correction=correction,
    )


def _clean_value(value: str) -> str:
    return value.strip().strip("\"'“”‘’ ").rstrip("。.!！")


def _is_correction(text: str, position: int) -> bool:
    prefix = text[max(0, position - 20) : position + 1]
    return bool(re.search(r"更正|改为|应为|纠正|现在是", prefix))


def _claim_id(candidate: ClaimCandidate) -> str:
    raw = "\0".join(
        (
            candidate.evidence.owner_scope,
            candidate.evidence.branch_scope,
            candidate.key,
            candidate.value,
            candidate.evidence.moment_id,
        )
    )
    return f"clm_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _sensitivity(key: str) -> str:
    sensitive = ("password", "secret", "api_key", "passport", "credential", "token")
    return "private" if any(part in key.casefold() for part in sensitive) else "normal"


def _contains_value(text_folded: str, value_folded: str) -> bool:
    if re.fullmatch(r"[a-z0-9_-]+", value_folded):
        return bool(
            re.search(
                rf"(?<![a-z0-9_-]){re.escape(value_folded)}(?![a-z0-9_-])",
                text_folded,
            )
        )
    return value_folded in text_folded
