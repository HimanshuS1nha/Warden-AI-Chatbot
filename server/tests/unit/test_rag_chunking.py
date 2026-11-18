import pytest

from tests.rag_fakes import WhitespaceTokenCounter
from warden.models import AccessLevel
from warden.rag.chunking import Chunk, DocumentInfo, chunk_blocks
from warden.rag.parsing import BlockKind, ParsedBlock

DOC = DocumentInfo("leave-policy", "Leave Policy", 2, AccessLevel.MANAGER)
COUNTER = WhitespaceTokenCounter()


def words(prefix: str, n: int) -> str:
    """`n` one-word sentences, "a1. a2. a3. ...", so every word is a token
    and every token ends a sentence."""
    return " ".join(f"{prefix}{i}." for i in range(1, n + 1))


def chunk(blocks: list[ParsedBlock], target: int = 50, overlap: int = 10) -> list[Chunk]:
    return chunk_blocks(blocks, DOC, COUNTER, target_tokens=target, overlap_tokens=overlap)


def test_every_chunk_carries_document_metadata() -> None:
    blocks = [
        ParsedBlock("Intro text.", 1, ()),
        ParsedBlock("Twelve days.", 2, ("3. Entitlement", "3.1 Joining")),
    ]
    chunks = chunk(blocks)
    assert [(c.chunk_index, c.section, c.page) for c in chunks] == [
        (0, "", 1),
        (1, "3. Entitlement > 3.1 Joining", 2),
    ]
    first = chunks[0]
    assert (first.doc_slug, first.title, first.version, first.access_level) == (
        "leave-policy",
        "Leave Policy",
        2,
        AccessLevel.MANAGER,
    )
    assert first.chunk_id == "leave-policy:2:0"
    assert first.token_count == 2


def test_embedding_text_includes_title_and_section() -> None:
    chunks = chunk([ParsedBlock("Twelve days.", 1, ("3. Entitlement",))])
    assert chunks[0].embedding_text == "Leave Policy\n3. Entitlement\n\nTwelve days."


def test_chunks_never_cross_a_heading() -> None:
    blocks = [
        ParsedBlock("Short one.", 1, ("1. Purpose",)),
        ParsedBlock("Short two.", 1, ("2. Scope",)),
    ]
    assert [c.text for c in chunk(blocks)] == ["Short one.", "Short two."]


def test_small_blocks_in_a_section_share_a_chunk() -> None:
    blocks = [ParsedBlock("First.", 1, ("1",)), ParsedBlock("- Second.", 1, ("1",))]
    assert [c.text for c in chunk(blocks)] == ["First.\n\n- Second."]


def test_long_section_splits_with_overlap_at_sentence_boundaries() -> None:
    blocks = [ParsedBlock(words("a", 30), 1, ("1",)), ParsedBlock(words("b", 30), 2, ("1",))]
    chunks = chunk(blocks, target=50, overlap=10)

    assert [c.text for c in chunks] == [
        words("a", 30),
        # The last 10 sentences of the first chunk, then the next block.
        " ".join(f"a{i}." for i in range(21, 31)) + "\n\n" + words("b", 30),
    ]
    assert all(c.token_count <= 50 for c in chunks)
    # The second chunk's overlap comes from page 1, so that's where it starts.
    assert [c.page for c in chunks] == [1, 1]


def test_overlap_is_shorter_when_sentences_are_long() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(1, 16)) + "."
    blocks = [
        ParsedBlock(f"{long_sentence} {long_sentence} {long_sentence}", 1, ("1",)),
        ParsedBlock(words("b", 30), 1, ("1",)),
    ]
    second = chunk(blocks, target=50, overlap=10)[1]
    # No whole sentence fits in 10 tokens, so the overlap is the last 10 words.
    assert second.text.startswith(" ".join(f"w{i}" for i in range(6, 16)) + ".\n\nb1.")


def test_oversized_paragraph_is_split_by_sentences() -> None:
    chunks = chunk([ParsedBlock(words("a", 100), 1, ("1",))], target=50, overlap=10)
    assert all(c.token_count <= 50 for c in chunks)
    texts = [c.text for c in chunks]
    assert texts[0] == words("a", 40)
    # Each later chunk opens with the previous chunk's last 10 sentences.
    assert texts[1].startswith("a31. a32.")
    # Nothing is lost: every sentence appears somewhere.
    joined = " ".join(texts)
    assert all(f"a{i}." in joined.split() for i in range(1, 101))


def test_sentence_longer_than_a_chunk_is_split_by_tokens() -> None:
    text = " ".join(f"w{i}" for i in range(1, 121))
    chunks = chunk([ParsedBlock(text, 1, ("1",))], target=50, overlap=10)
    assert all(c.token_count <= 50 for c in chunks)
    assert chunks[0].text.split()[0] == "w1"
    assert chunks[-1].text.split()[-1] == "w120"


def test_table_is_never_split_from_the_text_around_it_mid_row() -> None:
    table = "| Leave | Days |\n|---|---|\n| Casual | 12 |\n| Sick | 10 |"
    blocks = [
        ParsedBlock(words("a", 40), 1, ("3",)),
        ParsedBlock(table, 1, ("3",), BlockKind.TABLE),
        ParsedBlock(words("b", 5), 1, ("3",)),
    ]
    chunks = chunk(blocks, target=50, overlap=10)
    tables = [c for c in chunks if "| Casual | 12 |" in c.text]
    assert len(tables) == 1
    assert table in tables[0].text


def test_table_is_never_used_as_overlap() -> None:
    table = "| Leave | Days |\n|---|---|\n" + "\n".join(f"| t{i} | {i} |" for i in range(8))
    blocks = [
        ParsedBlock(table, 1, ("3",), BlockKind.TABLE),
        ParsedBlock(words("b", 45), 1, ("3",)),
    ]
    chunks = chunk(blocks, target=50, overlap=10)
    assert chunks[0].text == table
    assert chunks[1].text == words("b", 45)


def test_large_table_splits_between_rows_with_the_header_repeated() -> None:
    header = "| Expense | Limit |\n|---|---|"
    rows = [f"| item{i} | INR {i},000 per night |" for i in range(20)]
    table = header + "\n" + "\n".join(rows)
    chunks = chunk([ParsedBlock(table, 1, ("4",), BlockKind.TABLE)], target=50, overlap=10)

    assert len(chunks) > 1
    seen_rows: list[str] = []
    for c in chunks:
        lines = c.text.split("\n")
        assert "\n".join(lines[:2]) == header
        seen_rows.extend(lines[2:])
    assert seen_rows == rows


def test_rejects_overlap_not_smaller_than_target() -> None:
    with pytest.raises(ValueError):
        chunk([ParsedBlock("x", 1, ())], target=10, overlap=10)
