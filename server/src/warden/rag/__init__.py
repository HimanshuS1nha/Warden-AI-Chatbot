"""Retrieval-augmented generation over the company policy documents.

Ingestion runs parse -> clean -> chunk -> embed, then writes the chunks where
a retriever can search them. See `warden.rag.ingest`.
"""
