from typing import Any

from warden.rag.citations import cited_numbers, resolve_citations, sources_from_artifacts


def source(n: int) -> dict[str, Any]:
    return {
        "n": n,
        "chunk_id": f"leave-policy:2:{n}",
        "doc_slug": "leave-policy",
        "title": "Leave Policy",
        "version": 2,
        "section": f"Section {n}",
        "page": n,
        "text": f"Excerpt {n}.",
        "score": 0.71,
    }


SOURCES = {1: source(1), 2: source(2), 6: source(6)}


def test_cited_numbers_reads_single_and_grouped_markers_in_order() -> None:
    text = "Casual leave lapses [2]. Earned leave carries over [1, 6] up to 8 days [2,1]."

    assert cited_numbers(text) == [2, 1, 6]


def test_cited_numbers_ignores_links_and_non_numbers() -> None:
    text = "See [the policy](https://example.com) or [a] or [] or [1](x)."

    assert cited_numbers(text) == []


def test_invented_markers_are_dropped_and_real_ones_kept() -> None:
    text = "Five days carry forward [1, 9]. Managers decide [7]. Sick leave needs a note [6]."

    cleaned, citations = resolve_citations(text, SOURCES)

    assert cleaned == "Five days carry forward [1]. Managers decide. Sick leave needs a note [6]."
    assert [c["n"] for c in citations] == [1, 6]
    # What the client shows; the score and chunk id stay server side.
    assert citations[0] == {
        "n": 1,
        "doc_slug": "leave-policy",
        "title": "Leave Policy",
        "version": 2,
        "section": "Section 1",
        "page": 1,
        "text": "Excerpt 1.",
    }


def test_a_reply_without_markers_cites_nothing() -> None:
    assert resolve_citations("You have 8 days left.", SOURCES) == ("You have 8 days left.", [])


def test_sources_come_from_every_search_in_the_turn() -> None:
    first, second = [source(1), source(2)], [source(2), source(6)]

    assert sources_from_artifacts([first, [], second]).keys() == {1, 2, 6}
