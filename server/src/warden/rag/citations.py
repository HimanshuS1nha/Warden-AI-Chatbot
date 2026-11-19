"""Checks the model's citation markers against what policy search returned.

The model cites excerpts as [1] or [1, 3]. A number is real only if a
search_policies call in the same turn returned an excerpt with that number
(numbers run across the turn's searches). Anything else is invented and is
removed from the reply, so a citation shown to the user always points at
text the model was actually given.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

# [1], [1, 3], [1,2,3]. Not [a], [], or markdown links like [text](url).
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")
# Space left before a marker that was removed entirely: "rule [9]." -> "rule."
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([.,;:!?)])")

# The fields of a search_policies artifact entry shown under a reply.
CITATION_FIELDS = ("n", "doc_slug", "title", "version", "section", "page", "text")


def cited_numbers(text: str) -> list[int]:
    """Every number cited in `text`, in order of first appearance."""
    numbers: list[int] = []
    for marker in _MARKER.finditer(text):
        for part in marker.group(1).split(","):
            n = int(part)
            if n not in numbers:
                numbers.append(n)
    return numbers


def resolve_citations(
    text: str, sources: Mapping[int, Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """The reply with invented markers removed, and the citations it makes,
    in order of first appearance.

    `sources` maps each excerpt number returned this turn to its artifact
    entry (see `sources_from_artifacts`).
    """

    def keep_real(marker: re.Match[str]) -> str:
        real = [p.strip() for p in marker.group(1).split(",") if int(p) in sources]
        return f"[{', '.join(real)}]" if real else ""

    cleaned = _MARKER.sub(keep_real, text)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    citations = [
        {field: sources[n].get(field) for field in CITATION_FIELDS} for n in cited_numbers(cleaned)
    ]
    return cleaned, citations


def sources_from_artifacts(
    artifacts: Iterable[Iterable[Mapping[str, Any]]],
) -> dict[int, Mapping[str, Any]]:
    """Excerpt number -> artifact entry, from the artifacts of this turn's
    search_policies calls."""
    return {entry["n"]: entry for artifact in artifacts for entry in artifact}
