import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from tests.rag_fakes import DIMENSIONS, FakeSparseEncoder
from warden.rag.index import DENSE, SPARSE, PolicyIndex, PolicyIndexError, point_id


def test_point_id_is_a_stable_uuid_per_chunk() -> None:
    assert point_id("leave-policy:2:0") == point_id("leave-policy:2:0")
    assert point_id("leave-policy:2:0") != point_id("leave-policy:2:1")
    assert uuid.UUID(point_id("leave-policy:2:0")).version == 5


async def test_collection_has_dense_and_bm25_vectors(policy_index: PolicyIndex) -> None:
    info = await policy_index.client.get_collection(policy_index.collection)

    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[DENSE].size == DIMENSIONS
    sparse = info.config.params.sparse_vectors
    assert sparse is not None
    assert sparse[SPARSE].modifier == "idf"


async def test_ensure_collection_keeps_an_existing_one(policy_index: PolicyIndex) -> None:
    await policy_index.ensure_collection()

    assert await policy_index.client.collection_exists(policy_index.collection)


async def test_ensure_collection_refuses_another_vector_size(policy_index: PolicyIndex) -> None:
    other_model = PolicyIndex(
        policy_index.client,
        policy_index.collection,
        dimensions=DIMENSIONS * 2,
        sparse=FakeSparseEncoder(),
    )

    with pytest.raises(PolicyIndexError, match="--rebuild"):
        await other_model.ensure_collection()


async def test_unreachable_qdrant_is_a_policy_index_error() -> None:
    client = AsyncQdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    index = PolicyIndex(client, "x", dimensions=DIMENSIONS, sparse=FakeSparseEncoder())

    with pytest.raises(PolicyIndexError, match="Is it running"):
        await index.ensure_collection()
    await client.close()
