from dataclasses import dataclass
from typing import Protocol

from warden.models import AccessLevel, Role

# Which documents each role may retrieve. Employees see documents for
# everyone, managers also see manager documents, HR sees everything.
ROLE_ACCESS_LEVELS: dict[Role, frozenset[AccessLevel]] = {
    Role.EMPLOYEE: frozenset({AccessLevel.ALL}),
    Role.MANAGER: frozenset({AccessLevel.ALL, AccessLevel.MANAGER}),
    Role.HR: frozenset({AccessLevel.ALL, AccessLevel.MANAGER, AccessLevel.HR}),
}


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Limits a search. Built by the backend from the session, never from
    model output."""

    access_levels: frozenset[AccessLevel]
    # Only search these documents. None searches all of them.
    doc_slugs: frozenset[str] | None = None

    @classmethod
    def for_role(cls, role: Role) -> "SearchFilters":
        return cls(access_levels=ROLE_ACCESS_LEVELS[role])


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_slug: str
    title: str
    version: int
    access_level: AccessLevel
    section: str
    page: int | None
    text: str
    # Higher is more relevant. The scale depends on the retriever (cosine
    # similarity for dense search, a rerank score later).
    score: float


class Retriever(Protocol):
    async def search(self, query: str, k: int, filters: SearchFilters) -> list[RetrievedChunk]:
        """The `k` most relevant chunks that pass `filters`, best first. Only
        chunks from each document's active version are searched."""
        ...
