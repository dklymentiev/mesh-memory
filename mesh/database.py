#!/usr/bin/env python3
"""
Database connection and table management
"""
import asyncpg
import logging

logger = logging.getLogger(__name__)

async def create_tables(pool: asyncpg.Pool) -> None:
    """Create database tables if they don't exist"""
    async with pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                guid VARCHAR(32) PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash VARCHAR(32),
                filename VARCHAR(500),
                source VARCHAR(100) DEFAULT 'api',
                tags TEXT[] DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Add new columns if they don't exist (migration for existing deployments)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32)
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS filename VARCHAR(500)
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS source VARCHAR(100) DEFAULT 'api'
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()
        """)
        await conn.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT DEFAULT NULL
        """)

        # Create index on content_hash for fast lookup
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS documents_content_hash_idx
            ON documents(content_hash)
        """)
        
        # Create doc_embeddings table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_embeddings (
                guid VARCHAR(32) PRIMARY KEY REFERENCES documents(guid) ON DELETE CASCADE,
                embedding vector(768)
            )
        """)
        
        # Create HNSW index for vector similarity search.
        # HNSW provides better recall than IVFFlat with no tuning required.
        # Drop legacy IVFFlat index if it exists from older deployments.
        await conn.execute("""
            DROP INDEX IF EXISTS doc_embeddings_embedding_idx
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS doc_embeddings_hnsw_idx
            ON doc_embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        # Create document_metadata table for structured metadata
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id SERIAL PRIMARY KEY,
                guid VARCHAR(32) REFERENCES documents(guid) ON DELETE CASCADE,
                doc_type VARCHAR(50),
                metadata JSONB DEFAULT '{}',
                extracted_at TIMESTAMPTZ DEFAULT NOW(),
                extractor_version VARCHAR(20),
                UNIQUE(guid)
            )
        """)

        # Create indexes for metadata queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS document_metadata_guid_idx
            ON document_metadata(guid)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS document_metadata_doc_type_idx
            ON document_metadata(doc_type)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS document_metadata_metadata_idx
            ON document_metadata USING GIN (metadata)
        """)

        # Create category_centroids table for AI categorizer
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS category_centroids (
                category_id VARCHAR(64) PRIMARY KEY,
                parent_id VARCHAR(64),
                name VARCHAR(200) NOT NULL,
                description TEXT,
                centroid vector(768),
                doc_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS category_centroids_hnsw_idx
            ON category_centroids USING hnsw (centroid vector_cosine_ops)
        """)

        logger.info("Database tables created successfully")
