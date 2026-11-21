"""Print the top chunks for a few questions, to eyeball retrieval quality.

    uv run python -m warden.scripts.try_retrieval
    uv run python -m warden.scripts.try_retrieval "Can I carry forward casual leave?"

Uses the RETRIEVER setting (`RETRIEVER=dense uv run ...` to compare), without
its score threshold, so off-topic questions show the scores the threshold
has to beat. Searches as HR, so every document is in scope. Needs ingest to
have run and an embedding API key (each question is one call, cached after
the first run).

Throwaway: delete once `eval/retrieval.py` (task 10.5) measures this properly.
"""

import asyncio
import sys
import textwrap

from warden.core.config import get_settings
from warden.models import Role
from warden.rag.embeddings import Embedder, build_embedding_client
from warden.rag.retrievers import SearchFilters
from warden.rag.retrievers.factory import open_retriever

QUESTIONS = [
    "How many casual leaves do I get in a year?",
    "What's the carry-forward rule for earned leave?",
    "Can I backdate sick leave?",
    "What is the hotel limit per night in Mumbai?",
    "How many days a week can I work from home?",
    "What is the notice period if I resign?",
    "How long is maternity leave?",
    "Does health insurance cover my parents?",
    "Can I accept a gift from a vendor?",
    "What should a manager check before approving leave?",
    # Off-topic: nothing in the corpus answers these.
    "How do I bake sourdough bread?",
    "Who won the football world cup?",
    "What is the capital of Australia?",
]


async def main(questions: list[str]) -> None:
    settings = get_settings()
    client = build_embedding_client(settings)
    embedder = Embedder(client, settings.cache_dir, batch_size=settings.embedding_batch_size)
    filters = SearchFilters.for_role(Role.HR)
    try:
        async with open_retriever(settings, embedder, gated=False) as retriever:
            print(f"Retriever: {settings.retriever}")
            for question in questions:
                print(f"\n=== {question}")
                for rank, chunk in enumerate(await retriever.search(question, 3, filters), 1):
                    page = f", page {chunk.page}" if chunk.page is not None else ""
                    print(f"[{rank}] {chunk.score:.3f}  {chunk.title} v{chunk.version}{page}")
                    print(f"    {chunk.section}")
                    preview = textwrap.shorten(chunk.text, width=240, placeholder=" ...")
                    print(textwrap.indent(textwrap.fill(preview, width=96), "    "))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or QUESTIONS))
