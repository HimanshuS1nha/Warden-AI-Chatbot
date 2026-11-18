"""Clean parsed blocks before chunking.

- Normalize whitespace: odd Unicode spaces become plain spaces, soft hyphens
  and zero-width characters go, runs of spaces collapse.
- Drop running headers and footers. A PDF block in the page margin is page
  furniture (document name, page number, classification). A short block
  elsewhere that matches one, like a cover page banner that repeats the
  header, goes too.
- Join wrapped lines. A line ending in a hyphen joins without one when the
  unhyphenated word appears elsewhere in the document ("medi-" + "cal"), and
  keeps it otherwise ("ride-" + "hailing").

Tables keep their line breaks, since each line is a row. A table that
continues on the next page (same section, same header row, nothing in
between) is merged back into one table.
"""

import dataclasses
import re
from collections.abc import Collection, Iterable, Sequence

from warden.rag.parsing import BlockKind, ParsedBlock

# A body block longer than this is real content even if it repeats a header.
FURNITURE_MAX_WORDS = 8

_SPACES = re.compile(r"[ \t\f\v\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200d\u2060\ufeff]")
_BLANK_LINES = re.compile(r"\n{3,}")
_LINE_END_HYPHEN = re.compile(r"(\w+)-\n(\w+)")
_WORD = re.compile(r"[^\W\d_]+")


def normalize_whitespace(text: str) -> str:
    text = _INVISIBLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def word_vocabulary(texts: Iterable[str]) -> set[str]:
    """Lowercase words in `texts`, for deciding how to join hyphenated breaks."""
    return {word.lower() for text in texts for word in _WORD.findall(text)}


def fix_hyphenation(text: str, vocabulary: Collection[str]) -> str:
    """Join words broken across lines with a hyphen. The hyphen stays unless
    the joined word without it is in `vocabulary`."""

    def join(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if (left + right).lower() in vocabulary:
            return left + right
        return f"{left}-{right}"

    return _LINE_END_HYPHEN.sub(join, text)


def unwrap_lines(text: str) -> str:
    """Join a paragraph's wrapped lines with spaces."""
    return " ".join(line for line in text.split("\n") if line)


def furniture_key(text: str) -> str:
    """Compare headers and footers ignoring spacing and numbers, so "Page 1 of
    4" matches "Page 2 of 4" and "W A R D E N" matches "WA R D E N". Case
    counts: a footer's "INTERNAL" is furniture, a cover page's "Internal" isn't."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", text))


def clean_blocks(blocks: Sequence[ParsedBlock]) -> list[ParsedBlock]:
    normalized = [
        dataclasses.replace(block, text=normalize_whitespace(block.text)) for block in blocks
    ]
    furniture = {furniture_key(block.text) for block in normalized if block.in_margin}

    def is_furniture(block: ParsedBlock) -> bool:
        if block.in_margin:
            return True
        return (
            block.kind is BlockKind.TEXT
            and len(block.text.split()) <= FURNITURE_MAX_WORDS
            and furniture_key(block.text) in furniture
        )

    kept = [block for block in normalized if block.text and not is_furniture(block)]
    vocabulary = word_vocabulary(block.text for block in kept)

    cleaned: list[ParsedBlock] = []
    for block in kept:
        if block.kind is BlockKind.TEXT:
            text = unwrap_lines(fix_hyphenation(block.text, vocabulary))
            block = dataclasses.replace(block, text=text)
        cleaned.append(block)
    return merge_split_tables(cleaned)


def merge_split_tables(blocks: Sequence[ParsedBlock]) -> list[ParsedBlock]:
    """Join a table with the next block when that block is the same table
    continued on a new page: adjacent, same section and same header row. The
    repeated header and separator rows are dropped."""
    merged: list[ParsedBlock] = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.kind is BlockKind.TABLE
            and block.kind is BlockKind.TABLE
            and previous.heading_path == block.heading_path
            and previous.text.split("\n", 1)[0] == block.text.split("\n", 1)[0]
        ):
            rows = block.text.split("\n")[2:]
            merged[-1] = dataclasses.replace(previous, text="\n".join([previous.text, *rows]))
        else:
            merged.append(block)
    return merged
