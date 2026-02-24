#!/usr/bin/env python3
"""
Backfill doc_chunks for all existing documents.

Usage (inside container):
    docker exec mesh-api python /app/scripts/reindex_chunks.py

Loads all documents, chunks each, batch-embeds, stores in doc_chunks.
Also updates doc_embeddings with first-chunk embedding for consistency.
"""
import asyncio
import logging
import os
import sys
import time

# Ensure mesh package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncpg
from mesh.chunker import chunk_document
from mesh.embeddings import get_embedding_service
from mesh.config import get_database_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 32  # texts per embedding batch


async def main():
    logger.info("Starting chunk reindex...")
    start = time.time()

    # Connect to database
    pool = await asyncpg.create_pool(get_database_url(), min_size=2, max_size=5)

    # Ensure doc_chunks table exists
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id SERIAL PRIMARY KEY,
                guid VARCHAR(32) NOT NULL REFERENCES documents(guid) ON DELETE CASCADE,
                chunk_index SMALLINT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding vector(768),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(guid, chunk_index)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS doc_chunks_guid_idx ON doc_chunks(guid)
        """)

    # Load all documents
    async with pool.acquire() as conn:
        docs = await conn.fetch(
            "SELECT guid, content FROM documents ORDER BY created_at"
        )
    logger.info(f"Loaded {len(docs)} documents")

    # Initialize embedding service
    embedding_service = await get_embedding_service()

    total_chunks = 0
    processed = 0

    for doc in docs:
        guid = doc['guid']
        content = doc['content'] or ""

        # Chunk the document
        chunks = chunk_document(content)
        if not chunks:
            logger.warning(f"  {guid}: empty content, skipping")
            continue

        # Batch-embed all chunks
        embeddings = embedding_service.generate_embeddings_batch(chunks)

        # Store chunks
        async with pool.acquire() as conn:
            # Clear old chunks for this doc
            await conn.execute("DELETE FROM doc_chunks WHERE guid = $1", guid)

            for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                emb_str = '[' + ','.join(map(str, emb)) + ']'
                await conn.execute(
                    """INSERT INTO doc_chunks (guid, chunk_index, chunk_text, embedding)
                       VALUES ($1, $2, $3, $4::vector)""",
                    guid, idx, chunk_text, emb_str
                )

            # Update doc_embeddings with first chunk embedding
            first_emb_str = '[' + ','.join(map(str, embeddings[0])) + ']'
            await conn.execute(
                """INSERT INTO doc_embeddings (guid, embedding)
                   VALUES ($1, $2::vector)
                   ON CONFLICT (guid) DO UPDATE SET embedding = EXCLUDED.embedding""",
                guid, first_emb_str
            )

        total_chunks += len(chunks)
        processed += 1

        if processed % 50 == 0:
            logger.info(f"  Processed {processed}/{len(docs)} docs, {total_chunks} chunks so far...")

    # Build HNSW index (recreate to ensure optimal build)
    logger.info("Rebuilding HNSW index on doc_chunks...")
    async with pool.acquire() as conn:
        await conn.execute("DROP INDEX IF EXISTS doc_chunks_hnsw_idx")
        await conn.execute("""
            CREATE INDEX doc_chunks_hnsw_idx
            ON doc_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

    elapsed = time.time() - start
    logger.info(
        f"Reindex complete: {processed} docs, {total_chunks} chunks, "
        f"{elapsed:.1f}s elapsed"
    )

    # Print summary
    async with pool.acquire() as conn:
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM doc_chunks")
        doc_count = await conn.fetchval("SELECT COUNT(DISTINCT guid) FROM doc_chunks")
    logger.info(f"Final: {chunk_count} chunks across {doc_count} documents")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
