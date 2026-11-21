"""The Qdrant collection that policy search runs against.

One point per chunk of each document's active version, with two named
vectors and the chunk's metadata as payload:

- `dense`: the embedding model's vector, compared by cosine similarity.
- `bm25`: BM25 term frequencies. The IDF modifier makes Qdrant weight rare
  terms itself, from the collection's current contents.

Point IDs are `uuid5` of the chunk ID (`slug:version:chunk_index`), so
upserting the same chunk again overwrites it instead of adding a copy.

Postgres is the source of truth for documents. Everything in the collection
can be rebuilt from it with `uv run python -m warden.rag.ingest --rebuild`.
"""

import uuid
from collections.abc import Collection, Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from warden.core.config import Settings
from warden.rag.chunking import Chunk
from warden.rag.embeddings import Vectors
from warden.rag.sparse import Bm25Encoder, SparseEncoder

DENSE = "dense"
SPARSE = "bm25"

# Fixed forever: changing it gives every chunk a new point ID.
_POINT_NAMESPACE = uuid.UUID("6c1d8a52-9f0b-4a39-8f3e-2b7f61c0d4a7")

# Payload fields that searches filter on. Qdrant filters during the search
# when they're indexed, instead of scanning every candidate.
PAYLOAD_INDEXES = {
    "doc_slug": PayloadSchemaType.KEYWORD,
    "version": PayloadSchemaType.INTEGER,
    "access_level": PayloadSchemaType.KEYWORD,
    "is_active": PayloadSchemaType.BOOL,
}

_UPSERT_BATCH = 64


class PolicyIndexError(Exception):
    pass


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_slug": chunk.doc_slug,
        "title": chunk.title,
        "version": chunk.version,
        "access_level": chunk.access_level.value,
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "page": chunk.page,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "is_active": True,
    }


class PolicyIndex:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        *,
        dimensions: int,
        sparse: SparseEncoder,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dimensions = dimensions
        self.sparse = sparse

    async def ensure_collection(self) -> None:
        """Create the collection if it's missing. Fails if it exists with
        dense vectors of another size, which means the embedding model
        changed and the collection needs rebuilding."""
        try:
            if not await self.client.collection_exists(self.collection):
                await self._create()
                return
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise PolicyIndexError(_unreachable(exc)) from exc
        await self._check_dimensions()

    async def verify(self) -> int:
        """Check the collection is ready to search and return its point
        count. Fails if it's missing, empty, or built for another model."""
        try:
            exists = await self.client.collection_exists(self.collection)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise PolicyIndexError(_unreachable(exc)) from exc
        if not exists:
            raise PolicyIndexError(
                f"Qdrant has no {self.collection!r} collection. "
                "Run `uv run python -m warden.rag.ingest`."
            )
        await self._check_dimensions()
        count = await self.count()
        if count == 0:
            raise PolicyIndexError(
                f"Collection {self.collection!r} is empty. "
                "Run `uv run python -m warden.rag.ingest`."
            )
        return count

    async def _check_dimensions(self) -> None:
        try:
            info = await self.client.get_collection(self.collection)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise PolicyIndexError(_unreachable(exc)) from exc
        vectors = info.config.params.vectors
        size = vectors[DENSE].size if isinstance(vectors, dict) and DENSE in vectors else None
        if size != self.dimensions:
            raise PolicyIndexError(
                f"Collection {self.collection!r} has {DENSE!r} vectors of size {size}, but the "
                f"embedding model gives {self.dimensions}. Run ingest with --rebuild."
            )

    async def recreate_collection(self) -> None:
        """Drop the collection and create it empty."""
        try:
            await self.client.delete_collection(self.collection)
            await self._create()
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise PolicyIndexError(_unreachable(exc)) from exc

    async def _create(self) -> None:
        await self.client.create_collection(
            self.collection,
            vectors_config={DENSE: VectorParams(size=self.dimensions, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
        )
        for field, schema in PAYLOAD_INDEXES.items():
            await self.client.create_payload_index(self.collection, field, schema)

    async def upsert(self, chunks: Sequence[Chunk], dense: Vectors) -> list[str]:
        """Write the chunks as active points and return their IDs."""
        if len(chunks) != len(dense):
            raise ValueError(f"{len(chunks)} chunks but {len(dense)} vectors")
        sparse = await self.sparse.encode_documents([c.embedding_text for c in chunks])
        points = [
            PointStruct(
                id=point_id(chunk.chunk_id),
                vector={DENSE: vector.tolist(), SPARSE: sparse_vector},
                payload=chunk_payload(chunk),
            )
            for chunk, vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            await self.client.upsert(
                self.collection, points[start : start + _UPSERT_BATCH], wait=True
            )
        return [str(point.id) for point in points]

    async def retire_stale(self, doc_slug: str, keep: Collection[str]) -> int:
        """Remove a document's points other than `keep`: older versions and
        chunks the current version no longer has. Returns how many.

        They're marked inactive first, which searches filter out, so a
        failed delete never leaves them searchable."""
        stale = Filter(
            must=[FieldCondition(key="doc_slug", match=MatchValue(value=doc_slug))],
            must_not=[HasIdCondition(has_id=list(keep))],
        )
        count = (await self.client.count(self.collection, count_filter=stale, exact=True)).count
        if count:
            await self.client.set_payload(
                self.collection,
                {"is_active": False},
                points=FilterSelector(filter=stale),
                wait=True,
            )
            await self.client.delete(
                self.collection, points_selector=FilterSelector(filter=stale), wait=True
            )
        return count

    async def count(self, doc_slug: str | None = None) -> int:
        count_filter = (
            None
            if doc_slug is None
            else Filter(must=[FieldCondition(key="doc_slug", match=MatchValue(value=doc_slug))])
        )
        result = await self.client.count(self.collection, count_filter=count_filter, exact=True)
        return result.count


def build_index(
    settings: Settings, client: AsyncQdrantClient, sparse: SparseEncoder | None = None
) -> PolicyIndex:
    """The index the settings describe. BM25 defaults to fastembed's, loaded
    on first use."""
    return PolicyIndex(
        client,
        settings.qdrant_collection,
        dimensions=settings.embedding_dimensions,
        sparse=sparse or Bm25Encoder(settings.models_dir),
    )


def _unreachable(exc: Exception) -> str:
    return (
        f"Can't reach Qdrant: {exc}. Is it running? "
        "(docker compose -f docker-compose.dev.yml up -d)"
    )
