"""Split cleaned blocks into chunks for embedding and retrieval.

Chunks never cross a heading: blocks are grouped by heading path first, then
each section is packed into chunks of about `target_tokens`. A section that
needs more than one chunk repeats the end of the previous chunk (up to
`overlap_tokens`, in whole sentences where possible) at the start of the next
one, so a fact near a boundary appears whole in at least one chunk.

Tables are never cut mid-row and never used as overlap. A table larger than a
chunk is split between rows, with the header row repeated in each part.

Token counts come from a general-purpose tokenizer (GPT-2's), not the
embedding model's. They only size the chunks, so being close is enough.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby, pairwise
from typing import Protocol

from tokenizers import Tokenizer

from warden.models import AccessLevel
from warden.rag.parsing import BlockKind, ParsedBlock

TARGET_TOKENS = 500
OVERLAP_TOKENS = 80
TOKENIZER_NAME = "gpt2"
SECTION_SEPARATOR = " > "

_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")


class TokenCounter(Protocol):
    def offsets(self, text: str) -> list[tuple[int, int]]:
        """(start, end) character offsets of each token in `text`."""
        ...


class HFTokenCounter:
    """Counts tokens with a Hugging Face tokenizer, loaded on first use: the
    load checks the Hub for updates, which a run with nothing to chunk skips."""

    def __init__(self, name: str = TOKENIZER_NAME) -> None:
        self._name = name
        self._tokenizer: Tokenizer | None = None

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if self._tokenizer is None:
            # Downloaded from the Hub once, then read from its cache.
            self._tokenizer = Tokenizer.from_pretrained(self._name)
        return list(self._tokenizer.encode(text, add_special_tokens=False).offsets)


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    slug: str
    title: str
    version: int
    access_level: AccessLevel


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_slug: str
    title: str
    version: int
    access_level: AccessLevel
    chunk_index: int
    # Heading path joined with " > ". Empty for text before the first heading.
    section: str
    # Page the chunk starts on. None for documents without pages.
    page: int | None
    text: str
    token_count: int

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_slug}:{self.version}:{self.chunk_index}"

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the text with its document title and section,
        so a chunk that never names its topic can still match a query."""
        header = f"{self.title}\n{self.section}" if self.section else self.title
        return f"{header}\n\n{self.text}"


@dataclass(frozen=True, slots=True)
class _Unit:
    """A piece of a section that goes into a chunk whole."""

    text: str
    tokens: int
    page: int | None
    # Units from the same block join with a space, others with a blank line.
    block: int
    is_table: bool


def chunk_blocks(
    blocks: Sequence[ParsedBlock],
    document: DocumentInfo,
    counter: TokenCounter,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError("overlap_tokens must be at least 0 and less than target_tokens")

    chunks: list[Chunk] = []
    numbered = list(enumerate(blocks))
    for heading_path, group in groupby(numbered, key=lambda item: item[1].heading_path):
        units: list[_Unit] = []
        for block_number, block in group:
            if block.kind is BlockKind.TABLE:
                units.extend(_table_units(block, block_number, counter, target_tokens))
            else:
                # Leave room for the overlap so a chunk stays within target.
                limit = target_tokens - overlap_tokens
                units.extend(_text_units(block, block_number, counter, limit))
        section = SECTION_SEPARATOR.join(heading_path)
        for pieces in _pack(units, counter, target_tokens, overlap_tokens):
            text = _join(pieces)
            chunks.append(
                Chunk(
                    doc_slug=document.slug,
                    title=document.title,
                    version=document.version,
                    access_level=document.access_level,
                    chunk_index=len(chunks),
                    section=section,
                    page=next((u.page for u in pieces if u.page is not None), None),
                    text=text,
                    token_count=len(counter.offsets(text)),
                )
            )
    return chunks


def _pack(
    units: list[_Unit], counter: TokenCounter, target: int, overlap: int
) -> list[list[_Unit]]:
    packed: list[list[_Unit]] = []
    current: list[_Unit] = []
    size = 0
    for unit in units:
        if current and size + unit.tokens > target:
            packed.append(current)
            current = []
            size = 0
            tail = _overlap(packed[-1][-1], counter, overlap)
            if tail is not None and tail.tokens + unit.tokens <= target:
                current.append(tail)
                size = tail.tokens
        current.append(unit)
        size += unit.tokens
    if current:
        packed.append(current)
    return packed


def _overlap(unit: _Unit, counter: TokenCounter, limit: int) -> _Unit | None:
    """The end of `unit` to repeat at the start of the next chunk: as many
    whole trailing sentences as fit in `limit` tokens, or the last `limit`
    tokens from a word boundary when even one sentence is too long."""
    if unit.is_table or limit == 0:
        return None
    sentences = _sentences(unit.text)
    taken: list[str] = []
    tokens = 0
    for sentence in reversed(sentences):
        count = len(counter.offsets(sentence))
        if tokens + count > limit:
            break
        taken.insert(0, sentence)
        tokens += count
    text = " ".join(taken) if taken else _token_tail(unit.text, counter, limit)
    if not text:
        return None
    return _Unit(text, len(counter.offsets(text)), unit.page, unit.block, is_table=False)


def _token_tail(text: str, counter: TokenCounter, limit: int) -> str:
    offsets = counter.offsets(text)
    if len(offsets) <= limit:
        return text
    start = offsets[-limit][0]
    # Move forward to the next word so the overlap doesn't start mid-word.
    while start < len(text) and not text[start - 1].isspace():
        start += 1
    return text[start:].strip()


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s]


def _text_units(
    block: ParsedBlock, block_number: int, counter: TokenCounter, limit: int
) -> list[_Unit]:
    tokens = len(counter.offsets(block.text))
    if tokens <= limit:
        return [_Unit(block.text, tokens, block.page, block_number, is_table=False)]

    pieces: list[str] = []
    for sentence in _sentences(block.text):
        if len(counter.offsets(sentence)) > limit:
            pieces.extend(_hard_split(sentence, counter, limit))
        else:
            pieces.append(sentence)

    units: list[_Unit] = []
    current: list[str] = []
    for piece in pieces:
        candidate = " ".join([*current, piece])
        if current and len(counter.offsets(candidate)) > limit:
            units.append(_text_unit(" ".join(current), block, block_number, counter))
            current = [piece]
        else:
            current.append(piece)
    if current:
        units.append(_text_unit(" ".join(current), block, block_number, counter))
    return units


def _text_unit(text: str, block: ParsedBlock, block_number: int, counter: TokenCounter) -> _Unit:
    return _Unit(text, len(counter.offsets(text)), block.page, block_number, is_table=False)


def _hard_split(text: str, counter: TokenCounter, limit: int) -> list[str]:
    """Split text with no usable sentence breaks into pieces of at most
    `limit` tokens, at token boundaries."""
    offsets = counter.offsets(text)
    pieces = []
    for start in range(0, len(offsets), limit):
        window = offsets[start : start + limit]
        pieces.append(text[window[0][0] : window[-1][1]].strip())
    return [piece for piece in pieces if piece]


def _table_units(
    block: ParsedBlock, block_number: int, counter: TokenCounter, limit: int
) -> list[_Unit]:
    tokens = len(counter.offsets(block.text))
    lines = block.text.split("\n")
    if tokens <= limit or len(lines) <= 3:
        return [_Unit(block.text, tokens, block.page, block_number, is_table=True)]

    header, rows = lines[:2], lines[2:]
    units: list[_Unit] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join([*header, *current, row])
        if current and len(counter.offsets(candidate)) > limit:
            units.append(_table_unit([*header, *current], block, block_number, counter))
            current = [row]
        else:
            current.append(row)
    units.append(_table_unit([*header, *current], block, block_number, counter))
    return units


def _table_unit(
    lines: list[str], block: ParsedBlock, block_number: int, counter: TokenCounter
) -> _Unit:
    text = "\n".join(lines)
    return _Unit(text, len(counter.offsets(text)), block.page, block_number, is_table=True)


def _join(units: list[_Unit]) -> str:
    text = units[0].text
    for previous, unit in pairwise(units):
        separator = " " if unit.block == previous.block and not unit.is_table else "\n\n"
        text += separator + unit.text
    return text
