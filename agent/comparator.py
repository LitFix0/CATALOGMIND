"""
CatalogMind — Layer 9: Comparator.

Handles "What is the difference between OPQ and GSA?" style queries.

Design decisions:
  - Comparison is grounded ENTIRELY in catalog metadata — no LLM prior
    knowledge about assessments is used. The LLM (Layer 11) receives
    a structured comparison dict and is instructed to narrate it in
    natural language without adding facts not in the dict.

  - We compare on every available metadata dimension:
      * test_type codes, description, duration
      * remote_testing, adaptive_irt
      * job_levels, keys (assessment categories), languages

  - If one or both assessment names can't be found in the catalog,
    we return found=False with a clear fallback message.
    This prevents the LLM from hallucinating about unknown assessments.

  - The comparator does NOT return recommendations (empty list per spec)
    unless the user explicitly asked for a shortlist too.

  - Name matching uses fuzzy substring via retrieve_by_name() so
    "OPQ" matches "Occupational Personality Questionnaire OPQ32r".
"""

import logging
from dataclasses import dataclass, field

from retrieval.retriever import retrieve_by_name

logger = logging.getLogger(__name__)


@dataclass
class ComparisonDimension:
    """One dimension of a side-by-side comparison."""
    label: str
    value_a: str
    value_b: str

    @property
    def is_same(self) -> bool:
        return self.value_a.strip().lower() == self.value_b.strip().lower()

    @property
    def differs(self) -> bool:
        return not self.is_same and bool(self.value_a or self.value_b)


@dataclass
class ComparisonResult:
    """
    Full result of a two-assessment comparison.

    Attributes:
        found:        True if both assessments were found in catalog.
        name_a/b:     Resolved catalog names.
        url_a/b:      Catalog URLs (from shl_catalog.json only).
        dimensions:   Structured comparison across all metadata fields.
        not_found:    Terms that didn't match any catalog item.
        fallback_msg: Human-readable message if comparison can't be done.
    """
    found: bool
    name_a: str = ""
    name_b: str = ""
    url_a: str = ""
    url_b: str = ""
    dimensions: list[ComparisonDimension] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    fallback_msg: str = ""

    def to_prompt_dict(self) -> dict:
        """
        Serialize to a dict for injection into the LLM prompt.
        The LLM narrates this dict — it never invents its own facts.
        """
        return {
            "assessment_a": {"name": self.name_a, "url": self.url_a},
            "assessment_b": {"name": self.name_b, "url": self.url_b},
            "comparison": [
                {
                    "dimension": d.label,
                    "a": d.value_a,
                    "b": d.value_b,
                    "same": d.is_same,
                }
                for d in self.dimensions
            ],
        }


_TYPE_CODE_NAMES: dict[str, str] = {
    "A": "Ability & Aptitude",
    "P": "Personality & Behavior",
    "K": "Knowledge & Skills",
    "B": "Biodata & Situational Judgment",
    "S": "Simulations",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
}


def _expand_type_codes(codes_str: str) -> str:
    if not codes_str:
        return "Not specified"
    codes = [c.strip() for c in codes_str.split(",")]
    return ", ".join(_TYPE_CODE_NAMES.get(c, c) for c in codes if c)


def _fmt_bool(val: bool) -> str:
    return "Yes" if val else "No"


def _fmt_list(lst: list[str], max_items: int = 6) -> str:
    if not lst:
        return "Not specified"
    shown = lst[:max_items]
    suffix = f" (+{len(lst) - max_items} more)" if len(lst) > max_items else ""
    return ", ".join(shown) + suffix


def _fmt_duration(duration: str) -> str:
    if not duration:
        return "Not specified"
    try:
        return f"{int(duration)} minutes"
    except (ValueError, TypeError):
        return duration


def _build_dimensions(item_a: dict, item_b: dict) -> list[ComparisonDimension]:
    """
    Build structured comparison dimensions from two catalog items.
    Every value comes from shl_catalog.json — no invented data.
    """
    return [
        ComparisonDimension(
            label="Assessment Type",
            value_a=_expand_type_codes(item_a.get("test_type", "")),
            value_b=_expand_type_codes(item_b.get("test_type", "")),
        ),
        ComparisonDimension(
            label="Duration",
            value_a=_fmt_duration(item_a.get("duration", "")),
            value_b=_fmt_duration(item_b.get("duration", "")),
        ),
        ComparisonDimension(
            label="Remote Testing",
            value_a=_fmt_bool(item_a.get("remote_testing", False)),
            value_b=_fmt_bool(item_b.get("remote_testing", False)),
        ),
        ComparisonDimension(
            label="Adaptive / IRT",
            value_a=_fmt_bool(item_a.get("adaptive_irt", False)),
            value_b=_fmt_bool(item_b.get("adaptive_irt", False)),
        ),
        ComparisonDimension(
            label="Job Levels",
            value_a=_fmt_list(item_a.get("job_levels", [])),
            value_b=_fmt_list(item_b.get("job_levels", [])),
        ),
        ComparisonDimension(
            label="Assessment Categories",
            value_a=_fmt_list(item_a.get("keys", [])),
            value_b=_fmt_list(item_b.get("keys", [])),
        ),
        ComparisonDimension(
            label="Languages Available",
            value_a=_fmt_list(item_a.get("languages", []), max_items=4),
            value_b=_fmt_list(item_b.get("languages", []), max_items=4),
        ),
        ComparisonDimension(
            label="Description",
            value_a=item_a.get("description", "Not available")[:300],
            value_b=item_b.get("description", "Not available")[:300],
        ),
    ]


def compare(target_a: str, target_b: str) -> ComparisonResult:
    """
    Compare two SHL assessments by name, using catalog metadata only.

    Args:
        target_a: Name or abbreviation of the first assessment.
        target_b: Name or abbreviation of the second assessment.

    Returns:
        ComparisonResult with all dimensions populated,
        or found=False with fallback_msg if either wasn't in catalog.
    """
    logger.info("Comparing '%s' vs '%s'", target_a, target_b)

    item_a = retrieve_by_name(target_a)
    item_b = retrieve_by_name(target_b)

    not_found = []
    if item_a is None:
        not_found.append(target_a)
    if item_b is None:
        not_found.append(target_b)

    if not_found:
        return ComparisonResult(
            found=False,
            not_found=not_found,
            fallback_msg=(
                f"I couldn't find {' or '.join(repr(n) for n in not_found)} "
                f"in the SHL catalog. I can only compare assessments that "
                f"exist in the catalog. Could you check the name and try again, "
                f"or ask me to recommend assessments for a specific role instead?"
            ),
        )

    dimensions = _build_dimensions(item_a, item_b)
    logger.info(
        "Comparison built: %d dimensions, %d differ",
        len(dimensions),
        sum(1 for d in dimensions if d.differs),
    )

    return ComparisonResult(
        found=True,
        name_a=item_a["name"],
        name_b=item_b["name"],
        url_a=item_a["url"],
        url_b=item_b["url"],
        dimensions=dimensions,
    )


def compare_from_targets(targets: list[str]) -> ComparisonResult:
    """
    Convenience wrapper — takes compare_targets from ConversationContext.

    Args:
        targets: List of exactly 2 assessment name strings.
    """
    if len(targets) < 2:
        return ComparisonResult(
            found=False,
            fallback_msg=(
                "I need two assessment names to compare. "
                "For example: 'What is the difference between OPQ and Verify?'"
            ),
        )
    return compare(targets[0], targets[1])