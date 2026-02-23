#!/usr/bin/env python3
"""
Database CRUD operations for documents and embeddings
"""
import asyncpg
import hashlib
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import logging

from .models import DocumentCreateRequest, DocumentResponse, MetadataRequest, MetadataResponse

logger = logging.getLogger(__name__)


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash of content"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

class DocumentCRUD:
    """Database operations for documents"""
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def create_document(self, guid: str, request: DocumentCreateRequest) -> DocumentResponse:
        """Create a new document in the database"""
        now = datetime.now(timezone.utc)
        created_at = request.created_at or now
        updated_at = request.updated_at or now
        tags = request.tags or []
        content_hash = compute_content_hash(request.content)
        filename = request.filename
        source = request.source or "api"

        async with self.pool.acquire() as conn:
            try:
                # Dedup: check if document with same content already exists
                existing = await conn.fetchrow(
                    "SELECT guid FROM documents WHERE content_hash = $1 LIMIT 1",
                    content_hash
                )
                if existing:
                    logger.info(f"Dedup: content_hash {content_hash} already exists as {existing['guid']}, skipping")
                    # Return existing document info
                    row = await conn.fetchrow(
                        "SELECT * FROM documents WHERE guid = $1", existing['guid']
                    )
                    return DocumentResponse(
                        guid=row['guid'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        filename=row.get('filename'),
                        source=row.get('source', 'api'),
                        tags=list(row['tags']) if row['tags'] else [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        directory="inbox"
                    )

                # Insert document
                await conn.execute(
                    """
                    INSERT INTO documents (guid, content, content_hash, filename, source, tags, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    guid, request.content, content_hash, filename, source, tags, created_at, updated_at
                )

                return DocumentResponse(
                    guid=guid,
                    content=request.content,
                    content_hash=content_hash,
                    filename=filename,
                    source=source,
                    tags=tags,
                    created_at=created_at,
                    updated_at=updated_at,
                    directory="inbox"
                )

            except asyncpg.UniqueViolationError:
                logger.error(f"Document with GUID {guid} already exists")
                raise ValueError(f"Document with GUID {guid} already exists")
            except Exception as e:
                logger.error(f"Failed to create document {guid}: {e}")
                raise

    async def upsert_document(self, guid: str, request: DocumentCreateRequest) -> DocumentResponse:
        """Create or update a document by GUID"""
        now = datetime.now(timezone.utc)
        updated_at = request.updated_at or now
        tags = request.tags or []
        content_hash = compute_content_hash(request.content)
        filename = request.filename
        source = request.source or "api"

        created_at = request.created_at or now

        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO documents (guid, content, content_hash, filename, source, tags, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (guid) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        filename = EXCLUDED.filename,
                        source = EXCLUDED.source,
                        tags = EXCLUDED.tags,
                        updated_at = EXCLUDED.updated_at,
                        created_at = COALESCE($7, documents.created_at)
                    RETURNING created_at
                    """,
                    guid, request.content, content_hash, filename, source, tags, created_at, updated_at
                )
                logger.info(f"Upserted document {guid}")

                return DocumentResponse(
                    guid=guid,
                    content=request.content,
                    content_hash=content_hash,
                    filename=filename,
                    source=source,
                    tags=tags,
                    created_at=row['created_at'],
                    updated_at=updated_at,
                    directory="inbox"
                )

            except Exception as e:
                logger.error(f"Failed to upsert document {guid}: {e}")
                raise

    async def get_document_by_guid(self, guid: str) -> Optional[DocumentResponse]:
        """Get a document by its GUID"""
        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT guid, content, content_hash, filename, source, tags, created_at, updated_at
                    FROM documents
                    WHERE guid = $1
                    """,
                    guid
                )

                if not row:
                    return None

                return DocumentResponse(
                    guid=row['guid'],
                    content=row['content'],
                    content_hash=row['content_hash'],
                    filename=row['filename'],
                    source=row['source'],
                    tags=row['tags'] or [],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    directory="inbox"
                )

            except Exception as e:
                logger.error(f"Failed to get document {guid}: {e}")
                raise
    
    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[List[str]] = None
    ) -> List[DocumentResponse]:
        """Search documents by keyword (case-insensitive ILIKE).

        Used as a fallback/complement to semantic search for exact term matching.
        """
        async with self.pool.acquire() as conn:
            try:
                # Escape ILIKE special characters to prevent wildcard injection
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                pattern = f"%{escaped}%"

                if tags:
                    rows = await conn.fetch(
                        """
                        SELECT guid, content, content_hash, filename, source, tags, created_at, updated_at
                        FROM documents
                        WHERE content ILIKE $1
                          AND tags @> $3::text[]
                        ORDER BY updated_at DESC
                        LIMIT $2
                        """,
                        pattern, limit, tags
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT guid, content, content_hash, filename, source, tags, created_at, updated_at
                        FROM documents
                        WHERE content ILIKE $1
                        ORDER BY updated_at DESC
                        LIMIT $2
                        """,
                        pattern, limit
                    )

                return [
                    DocumentResponse(
                        guid=row['guid'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        filename=row['filename'],
                        source=row['source'],
                        tags=row['tags'] or [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        directory="inbox"
                    )
                    for row in rows
                ]

            except Exception as e:
                logger.error(f"Failed to keyword search: {e}")
                raise


class EmbeddingCRUD:
    """Database operations for embeddings"""
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def store_embedding(self, guid: str, embedding: List[float]) -> None:
        """Store an embedding for a document"""
        async with self.pool.acquire() as conn:
            try:
                # Convert list to vector format for PostgreSQL
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                await conn.execute(
                    """
                    INSERT INTO doc_embeddings (guid, embedding)
                    VALUES ($1, $2::vector)
                    ON CONFLICT (guid) DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    guid, embedding_str
                )
                
            except Exception as e:
                logger.error(f"Failed to store embedding for {guid}: {e}")
                raise
    
    async def search_similar_documents(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar documents using cosine similarity.

        Args:
            tags: If provided, only return documents containing ALL of these tags
                  (PostgreSQL array @> operator).
        """
        async with self.pool.acquire() as conn:
            try:
                query_vector = '[' + ','.join(map(str, query_embedding)) + ']'

                if tags:
                    rows = await conn.fetch(
                        """
                        SELECT e.guid,
                               1 - (e.embedding <=> $1::vector) AS similarity_score
                        FROM doc_embeddings e
                        JOIN documents d ON d.guid = e.guid
                        WHERE 1 - (e.embedding <=> $1::vector) >= $2
                          AND d.tags @> $4::text[]
                        ORDER BY similarity_score DESC
                        LIMIT $3
                        """,
                        query_vector, similarity_threshold, limit, tags
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT e.guid,
                               1 - (e.embedding <=> $1::vector) AS similarity_score
                        FROM doc_embeddings e
                        WHERE 1 - (e.embedding <=> $1::vector) >= $2
                        ORDER BY similarity_score DESC
                        LIMIT $3
                        """,
                        query_vector, similarity_threshold, limit
                    )

                return [(row['guid'], float(row['similarity_score'])) for row in rows]

            except Exception as e:
                logger.error(f"Failed to search similar documents: {e}")
                raise
    


class MetadataCRUD:
    """Database operations for document metadata"""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_metadata(self, guid: str) -> Optional[MetadataResponse]:
        """Get metadata for a document"""
        import json
        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT guid, doc_type, metadata, extracted_at, extractor_version
                    FROM document_metadata
                    WHERE guid = $1
                    """,
                    guid
                )

                if not row:
                    return None

                # Parse JSONB if it's a string
                metadata = row['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                return MetadataResponse(
                    guid=row['guid'],
                    doc_type=row['doc_type'],
                    metadata=metadata or {},
                    extracted_at=row['extracted_at'],
                    extractor_version=row['extractor_version']
                )

            except Exception as e:
                logger.error(f"Failed to get metadata for {guid}: {e}")
                raise

    async def upsert_metadata(self, guid: str, request: MetadataRequest) -> MetadataResponse:
        """Create or update metadata for a document"""
        now = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            try:
                # Check if document exists
                doc_exists = await conn.fetchval(
                    "SELECT 1 FROM documents WHERE guid = $1", guid
                )
                if not doc_exists:
                    raise ValueError(f"Document {guid} not found")

                # Upsert metadata
                import json
                metadata_json = json.dumps(request.metadata)

                await conn.execute(
                    """
                    INSERT INTO document_metadata (guid, doc_type, metadata, extracted_at, extractor_version)
                    VALUES ($1, $2, $3::jsonb, $4, $5)
                    ON CONFLICT (guid) DO UPDATE SET
                        doc_type = EXCLUDED.doc_type,
                        metadata = EXCLUDED.metadata,
                        extracted_at = EXCLUDED.extracted_at,
                        extractor_version = EXCLUDED.extractor_version
                    """,
                    guid, request.doc_type, metadata_json, now, request.extractor_version
                )

                return MetadataResponse(
                    guid=guid,
                    doc_type=request.doc_type,
                    metadata=request.metadata,
                    extracted_at=now,
                    extractor_version=request.extractor_version
                )

            except ValueError:
                raise
            except Exception as e:
                logger.error(f"Failed to upsert metadata for {guid}: {e}")
                raise

    async def delete_metadata(self, guid: str) -> bool:
        """Delete metadata for a document"""
        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute(
                    "DELETE FROM document_metadata WHERE guid = $1",
                    guid
                )
                return result == "DELETE 1"

            except Exception as e:
                logger.error(f"Failed to delete metadata for {guid}: {e}")
                raise

    async def list_by_type(self, doc_type: str, limit: int = 100) -> List[MetadataResponse]:
        """List all metadata of a specific type"""
        import json
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT guid, doc_type, metadata, extracted_at, extractor_version
                    FROM document_metadata
                    WHERE doc_type = $1
                    ORDER BY extracted_at DESC
                    LIMIT $2
                    """,
                    doc_type, limit
                )

                results = []
                for row in rows:
                    metadata = row['metadata']
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    results.append(MetadataResponse(
                        guid=row['guid'],
                        doc_type=row['doc_type'],
                        metadata=metadata or {},
                        extracted_at=row['extracted_at'],
                        extractor_version=row['extractor_version']
                    ))
                return results

            except Exception as e:
                logger.error(f"Failed to list metadata by type {doc_type}: {e}")
                raise

