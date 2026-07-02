"""
CatalogMind — Layer 6: Context Extractor.

Extracts structured hiring context from the full conversation history.
This is the "memory" of the agent — it reads the entire message history
and produces a single ConversationContext object that downstream layers
(decision, recommender, comparator) use to make decisions.

Design decisions:
  - Rule-based extraction first, no LLM needed here.
    Keyword matching covers 90% of real cases and is fast/deterministic.

  - We scan the FULL conversation history, not just the latest message.
    Users give context piecemeal: turn 1 gives role, turn 3 adds seniority,
    turn 5 adds a constraint. The extractor accumulates all of it.

  - Refinements override earlier values. Later mentions of role/seniority
    overwrite earlier ones. Skills accumulate (additive).

  - has_enough_context() is a heuristic — requires role OR skill OR JD.
    If neither is present, decision layer asks for clarification.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    role: str = ""
    seniority: str = ""
    skills: list[str] = field(default_factory=list)
    test_type_prefs: list[str] = field(default_factory=list)
    job_description: str = ""
    remote_required: bool = False
    compare_targets: list[str] = field(default_factory=list)
    raw_query: str = ""
    has_jd: bool = False
    turn_count: int = 0
    latest_user_message: str = ""

    def has_enough_context(self) -> bool:
        return bool(self.role or self.skills or self.has_jd)

    def build_search_query(self) -> str:
        parts = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        if self.skills:
            parts.append(" ".join(self.skills))
        if self.test_type_prefs:
            type_names = [_CODE_TO_NAME.get(c, c) for c in self.test_type_prefs]
            parts.append(" ".join(type_names))
        if self.has_jd:
            parts.append(self.job_description[:300])
        if not parts:
            parts.append(self.raw_query)
        return " ".join(parts)


_CODE_TO_NAME: dict[str, str] = {
    "A": "ability aptitude cognitive",
    "P": "personality behavior",
    "K": "knowledge skills technical",
    "B": "biodata situational judgment",
    "S": "simulation exercise",
    "C": "competency",
    "D": "development 360",
    "E": "assessment exercise",
}

_SENIORITY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(c-suite|ceo|cto|cfo|chief)\b", "executive"),
    (r"\b(executive|vp |vice president)\b", "executive"),
    (r"\b(director)\b", "director"),
    (r"\b(senior|sr\.?|lead|principal|staff)\b", "senior"),
    (r"\b(mid.level|mid level|middle|intermediate|experienced)\b", "mid"),
    (r"\b([3-9]\s*(?:\+\s*)?years?|1[0-9]\s*years?)\b", "mid"),
    (r"\b(junior|jr\.?|early.career|entry.level|entry level)\b", "junior"),
    (r"\b(graduate|grad|fresher|intern|new.grad)\b", "graduate"),
    (r"\b(1[-–]?2\s*years?|0[-–]?2\s*years?|less than 2)\b", "junior"),
]

_TEST_TYPE_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(personality|behaviour|behavioral|opq|psq)\b", "P"),
    (r"\b(cognitive|ability|aptitude|reasoning|verbal|numerical|inductive|"
     r"deductive|abstract|iq)\b", "A"),
    (r"\b(knowledge|technical|skills?\s*test|coding|programming)\b", "K"),
    (r"\b(situational.?judgment|sjt|biodata)\b", "B"),
    (r"\b(simulation|job simulation|work sample)\b", "S"),
    (r"\b(competenc|360|feedback)\b", "C"),
]

_ROLE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(software|data|ml|machine learning|ai|devops|frontend|backend|"
     r"fullstack|full.stack)\s+(engineer|developer|scientist)\b", None),
    (r"\b(java|python|javascript|js|react|node|sql|cloud|aws|azure|gcp)"
     r"\s+(developer|engineer|programmer)\b", None),
    (r"\b(product\s+manager|project\s+manager|program\s+manager)\b", None),
    (r"\b(sales\s+(manager|executive|representative|rep|associate))\b", None),
    (r"\b(customer\s+service|customer\s+support|contact\s+center)\s*"
     r"(agent|representative|rep|associate)?\b", None),
    (r"\b(finance|financial|accounting|accountant|analyst)\b", None),
    (r"\b(hr|human\s+resources?|recruiter|talent\s+acquisition)\b", None),
    (r"\b(manager|director|executive|leader|supervisor|head\s+of)\b", None),
    (r"\b(developer|engineer|programmer|architect|scientist)\b", None),
    (r"\b(analyst|consultant|advisor|specialist|coordinator)\b", None),
    (r"\b(graduate|intern|trainee|associate)\b", None),
]

_SKILL_KEYWORDS: list[str] = [
    "java", "python", "javascript", "typescript", "react", "node",
    "sql", "nosql", "mongodb", "postgresql", "aws", "azure", "gcp",
    "docker", "kubernetes", "ml", "machine learning", "deep learning",
    "data analysis", "data science", "excel", "power bi", "tableau",
    "c++", "c#", ".net", "go", "rust", "scala", "kotlin", "swift",
    "salesforce", "sap", "communication", "leadership", "negotiation",
    "stakeholder management", "project management",
]

_JD_SIGNALS: list[str] = [
    "job description", "jd:", "job spec", "here is the",
    "responsibilities:", "requirements:", "qualifications:",
    "we are looking for", "the role involves", "duties include",
    "key skills:", "must have:", "nice to have:",
]

_COMPARE_PATTERNS: list[str] = [
    r"(?:difference|compare|comparison|versus|vs\.?)\s+(?:between\s+)?(.+?)\s+and\s+(.+?)(?:\?|$)",
    r"(.+?)\s+(?:vs\.?|versus|or)\s+(.+?)(?:\s+difference|\s+comparison|\?|$)",
    r"what(?:'s| is) (?:the difference|better).*?between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
]

_KNOWN_ASSESSMENTS: list[str] = [
    "opq", "opq32", "opq32r", "gsa", "global skills", "verify",
    "sjt", "mq", "motivation questionnaire", "csq", "call simulation",
    "hipo", "hi-po", "enterprise leadership",
]


def _extract_seniority(text: str) -> str:
    lower = text.lower()
    for pattern, label in _SENIORITY_PATTERNS:
        if re.search(pattern, lower):
            return label
    return ""


def _extract_role(text: str) -> str:
    lower = text.lower()
    for pattern, _ in _ROLE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            return match.group(0).strip()
    return ""


def _extract_skills(text: str) -> list[str]:
    lower = text.lower()
    return [skill for skill in _SKILL_KEYWORDS if skill in lower]


def _extract_test_type_prefs(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for pattern, code in _TEST_TYPE_KEYWORDS:
        if re.search(pattern, lower) and code not in found:
            found.append(code)
    return found


def _detect_job_description(text: str) -> bool:
    lower = text.lower()
    signal_count = sum(1 for sig in _JD_SIGNALS if sig in lower)
    return signal_count >= 2 or (len(text) > 300 and signal_count >= 1)


def _extract_compare_targets(text: str) -> list[str]:
    lower = text.lower()
    for pattern in _COMPARE_PATTERNS:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            t1 = match.group(1).strip().rstrip(".,;")
            t2 = match.group(2).strip().rstrip(".,;")
            if t1 and t2:
                return [t1, t2]
    found = [a for a in _KNOWN_ASSESSMENTS if a in lower]
    if len(found) >= 2:
        return found[:2]
    return []


def extract_context(messages: list[dict]) -> ConversationContext:
    """
    Extract structured context from full conversation history.

    Scans ALL user messages in order — later messages override
    earlier ones for scalar fields (role, seniority), while
    list fields (skills, test_type_prefs) accumulate across turns.
    """
    ctx = ConversationContext()
    ctx.turn_count = len(messages)

    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ctx

    ctx.latest_user_message = user_messages[-1]["content"]
    ctx.raw_query = " ".join(m["content"] for m in user_messages)

    for msg in user_messages:
        text = msg["content"]

        role = _extract_role(text)
        if role:
            ctx.role = role

        seniority = _extract_seniority(text)
        if seniority:
            ctx.seniority = seniority

        for skill in _extract_skills(text):
            if skill not in ctx.skills:
                ctx.skills.append(skill)

        for code in _extract_test_type_prefs(text):
            if code not in ctx.test_type_prefs:
                ctx.test_type_prefs.append(code)

        if any(kw in text.lower() for kw in ["remote", "online", "virtual"]):
            ctx.remote_required = True

        if _detect_job_description(text):
            ctx.job_description = text
            ctx.has_jd = True

        targets = _extract_compare_targets(text)
        if targets:
            ctx.compare_targets = targets

    logger.debug(
        "Context extracted: role='%s' seniority='%s' skills=%s "
        "test_type_prefs=%s has_jd=%s compare=%s",
        ctx.role, ctx.seniority, ctx.skills,
        ctx.test_type_prefs, ctx.has_jd, ctx.compare_targets,
    )

    return ctx