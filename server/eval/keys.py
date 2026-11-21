"""Dataset labels to chunk IDs (task 10.2).

Questions are labelled with the sections that answer them, as
`slug:version:section` (the heading path, as in a chunk's `section`), not
with chunk indexes. Changing the chunk size or overlap renumbers chunks but
keeps sections, so the labels stay valid; a section split into several
chunks maps to all of them, and finding any one counts.

A label that matches no chunk means a document or its headings changed, and
the dataset needs fixing. That raises instead of quietly scoring zero.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from warden.rag.chunking import Chunk


class UnknownKey(Exception):
    pass


def section_key(chunk: Chunk) -> str:
    return f"{chunk.doc_slug}:{chunk.version}:{chunk.section}"


def chunk_ids_by_key(chunks: Iterable[Chunk]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        index[section_key(chunk)].add(chunk.chunk_id)
    return dict(index)


def resolve(keys: Sequence[str], index: dict[str, set[str]]) -> list[set[str]]:
    """The chunk IDs for each key, in order. Raises UnknownKey for a key no
    chunk has."""
    unknown = [key for key in keys if key not in index]
    if unknown:
        raise UnknownKey(f"No chunk has the section {', '.join(map(repr, unknown))}")
    return [index[key] for key in keys]
