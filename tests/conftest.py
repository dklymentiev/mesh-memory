"""
Pytest configuration and shared fixtures for Mesh API tests.

Strategy:
  1. Stub sys.modules for heavy ML deps (sentence_transformers, umap, sklearn)
     before main.py is ever imported, so no GPU / model download is needed.
  2. Patch asyncpg.create_pool so no real PostgreSQL is required.
  3. Replace DocumentCRUD / EmbeddingCRUD / MetadataCRUD with in-memory fakes.
  4. The fake asyncpg connection also handles direct conn.execute/fetch calls
     that some endpoints make bypassing the CRUD layer (DELETE, GET /tags, etc.)
  5. Expose an httpx AsyncClient pointing at the FastAPI app.
"""

import asyncio
import hashlib
import sys
import unittest.mock as mock
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies BEFORE importing anything from the project.
# ---------------------------------------------------------------------------

def _stub_module(name: str, **attrs):
    if name not in sys.modules:
        m = mock.MagicMock()
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
    return sys.modules[name]


_stub_module("sentence_transformers")
_stub_module("umap")
_stub_module("umap.umap_")
_stub_module("sklearn")
_stub_module("sklearn.preprocessing")
_stub_module("sklearn.cluster")
_stub_module("sklearn.decomposition")


# ---------------------------------------------------------------------------
# In-memory document store
# ---------------------------------------------------------------------------

class InMemoryStore:
    """Mimics the PostgreSQL documents + doc_embeddings tables."""

    def __init__(self):
        self.documents: Dict[str, dict] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, dict] = {}

    def reset(self):
        self.documents.clear()
        self.embeddings.clear()
        self.metadata.clear()

    def add_document(self, guid: str, content: str, tags: List[str],
                     filename: Optional[str] = None,
                     source: str = "api") -> dict:
        now = datetime.now(timezone.utc)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        doc = {
            "guid": guid, "content": content, "content_hash": content_hash,
            "filename": filename, "source": source, "tags": tags,
            "created_at": now, "updated_at": now, "directory": "inbox",
        }
        self.documents[guid] = doc
        return doc

    def get_document(self, guid: str) -> Optional[dict]:
        return self.documents.get(guid)

    def remove_document(self, guid: str) -> int:
        """Remove document by GUID; return 1 if deleted, 0 if not found."""
        if guid in self.documents:
            del self.documents[guid]
            return 1
        return 0

    def list_documents(self, limit: int = 10, offset: int = 0,
                       tag: Optional[str] = None) -> List[dict]:
        docs = list(self.documents.values())
        if tag:
            docs = [d for d in docs if tag in d["tags"]]
        docs.sort(key=lambda d: d["updated_at"], reverse=True)
        return docs[offset: offset + limit]

    def keyword_search(self, query: str, limit: int = 10,
                       tags: Optional[List[str]] = None) -> List[dict]:
        results = []
        for doc in self.documents.values():
            if query.lower() in doc["content"].lower():
                if tags is None or all(t in doc["tags"] for t in tags):
                    results.append(doc)
        return results[:limit]

    def count(self) -> int:
        return len(self.documents)

    def all_tags(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for doc in self.documents.values():
            for t in doc["tags"]:
                counts[t] = counts.get(t, 0) + 1
        return counts


_store = InMemoryStore()


# ---------------------------------------------------------------------------
# asyncpg pool / connection fakes
# ---------------------------------------------------------------------------

def _row(**kw):
    """Create a dict-like object that supports both row['key'] and row.attr access."""
    r = {}
    r.update(kw)

    class _Row(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    return _Row(kw)


def _make_fake_pool():
    conn = mock.MagicMock()
    conn.__aenter__ = mock.AsyncMock(return_value=conn)
    conn.__aexit__ = mock.AsyncMock(return_value=False)

    async def _execute(sql, *args):
        s = sql.strip().upper()
        if s.startswith("DELETE FROM DOCUMENTS WHERE GUID"):
            n = _store.remove_document(args[0])
            return f"DELETE {n}"
        if s.startswith("DELETE FROM DOCUMENTS"):
            return "DELETE 0"
        if s.startswith("UPDATE DOCUMENTS"):
            # Used by PATCH endpoint
            return "UPDATE 1"
        return None

    conn.execute = _execute

    async def _fetchval(sql, *args):
        s = sql.strip().upper()
        if "COUNT(*) FROM DOCUMENTS" in s:
            return _store.count()
        if "COUNT(*) FROM DOC_EMBEDDINGS" in s:
            return len(_store.embeddings)
        if "COUNT(*) FROM DOCUMENT_METADATA" in s:
            return len(_store.metadata)
        if "COUNT(DISTINCT" in s:
            return len(_store.all_tags())
        if "MAX(CREATED_AT)" in s:
            if _store.documents:
                return max(d["created_at"] for d in _store.documents.values())
            return None
        if "SELECT 1" in s:
            return 1
        return None

    conn.fetchval = _fetchval

    async def _fetchrow(sql, *args):
        s = sql.strip().upper()
        if "CONTENT_HASH" in s and "FROM DOCUMENTS" in s and args:
            # GET /by-hash/{hash} and PATCH /by-hash lookup
            for doc in _store.documents.values():
                if doc.get("content_hash") == args[0]:
                    return _row(**doc)
            return None
        if "FROM DOCUMENTS WHERE GUID" in s and args:
            doc = _store.get_document(args[0])
            return _row(**doc) if doc else None
        if "FROM DOCUMENTS" in s and "TYPE:README" in s:
            return None  # No README doc in test store
        return None

    conn.fetchrow = _fetchrow

    async def _fetch(sql, *args):
        s = sql.strip().upper()
        if "GROUP BY WORKSPACE_ID" in s:
            # GET /stats workspace summary
            now = datetime.now(timezone.utc)
            return [_row(workspace_id="default", doc_count=_store.count(),
                         last_activity=now)]
        if "UNNEST(TAGS)" in s and ("MAX(UPDATED_AT)" in s or "LAST_ACTIVITY" in s
                                      or "MAX(D.UPDATED_AT)" in s or "COUNT(*)" in s):
            # GET /activity endpoint
            prefix_arg = None
            if "LIKE" in s and args:
                prefix_arg = args[0].rstrip("%") if args else None
            tag_counts = _store.all_tags()
            rows = []
            now = datetime.now(timezone.utc)
            for tag, count in tag_counts.items():
                if prefix_arg is None or tag.startswith(prefix_arg):
                    rows.append(_row(tag=tag, doc_count=count,
                                     last_activity=now, first_activity=now))
            limit_val = int(args[-1]) if args else 20
            return rows[:limit_val]
        if "UNNEST(TAGS)" in s:
            # GET /tags endpoint - build tag + doc_count rows
            prefix_arg = None
            if "LIKE $1" in s or "WHERE TAG LIKE" in s:
                if args:
                    prefix_arg = args[0].rstrip("%") if args else None
            tag_counts = _store.all_tags()
            rows = []
            for tag, count in tag_counts.items():
                if prefix_arg is None or tag.startswith(prefix_arg):
                    rows.append(_row(tag=tag, doc_count=count))
            return rows
        if "TRIM(CONTENT)" in s or "LENGTH(TRIM" in s:
            # GET /short endpoint
            max_len = int(args[0]) if args else 10
            limit_val = int(args[1]) if len(args) > 1 else 50
            rows = []
            for doc in _store.documents.values():
                if len(doc["content"].strip()) < max_len:
                    rows.append(_row(
                        guid=doc["guid"], content=doc["content"],
                        tags=doc["tags"], created_at=doc["created_at"],
                        content_length=len(doc["content"].strip())
                    ))
            return rows[:limit_val]
        if "ANY($1)" in s and "FROM DOCUMENTS" in s and args:
            # Batch fetch by GUID list (used by /search N+1 fix)
            guid_list = args[0] if isinstance(args[0], list) else [args[0]]
            rows = []
            for guid in guid_list:
                doc = _store.get_document(guid)
                if doc:
                    rows.append(_row(**doc))
            return rows
        if "FROM DOCUMENTS" in s:
            limit = int(args[0]) if args else 10
            offset = int(args[1]) if len(args) > 1 else 0
            docs = _store.list_documents(limit=limit, offset=offset)
            return [_row(**d) for d in docs]
        if "FROM DOC_EMBEDDINGS" in s:
            return []
        if "FROM DOCUMENT_METADATA" in s:
            return []
        return []

    conn.fetch = _fetch

    pool = mock.MagicMock()
    pool.acquire = mock.MagicMock()
    pool.acquire.return_value.__aenter__ = mock.AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = mock.AsyncMock(return_value=False)
    return pool


# ---------------------------------------------------------------------------
# Fake EmbeddingService
# ---------------------------------------------------------------------------

class FakeEmbeddingService:
    model = True  # truthy -> health check passes
    model_name = "fake-model-for-tests"

    def generate_embedding(self, text: str) -> List[float]:
        return [0.0] * 768

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 768 for _ in texts]


_fake_emb_svc = FakeEmbeddingService()


async def _get_fake_embedding_service():
    return _fake_emb_svc


# ---------------------------------------------------------------------------
# Fake CRUD classes
# ---------------------------------------------------------------------------

from mesh.models import (
    DocumentCreateRequest, DocumentResponse,
    MetadataRequest, MetadataResponse,
    SearchResult, SearchResponse,
)


def _doc_response(doc: dict) -> DocumentResponse:
    return DocumentResponse(
        guid=doc["guid"], content=doc["content"],
        content_hash=doc.get("content_hash"), filename=doc.get("filename"),
        source=doc.get("source", "api"), tags=doc.get("tags", []),
        created_at=doc["created_at"], updated_at=doc.get("updated_at"),
        directory="inbox",
    )


class FakeDocumentCRUD:
    def __init__(self, pool): self.pool = pool

    def with_auth(self, auth):
        """Return self -- fakes don't need auth scoping."""
        return self

    async def create_document(self, guid: str, req: DocumentCreateRequest,
                              workspace_id: str = "default") -> DocumentResponse:
        doc = _store.add_document(
            guid=guid, content=req.content, tags=req.tags or [],
            filename=req.filename, source=req.source or "api",
        )
        return _doc_response(doc)

    async def upsert_document(self, guid: str, req: DocumentCreateRequest,
                              workspace_id: str = "default") -> DocumentResponse:
        if guid in _store.documents:
            old = _store.documents[guid]
            old["content"] = req.content
            old["tags"] = req.tags or []
            old["updated_at"] = datetime.now(timezone.utc)
            old["content_hash"] = hashlib.md5(req.content.encode()).hexdigest()
            return _doc_response(old)
        return await self.create_document(guid, req)

    async def get_document_by_guid(self, guid: str) -> Optional[DocumentResponse]:
        doc = _store.get_document(guid)
        return _doc_response(doc) if doc else None

    async def keyword_search(self, query: str, limit=10, tags=None) -> List[DocumentResponse]:
        return [_doc_response(d) for d in _store.keyword_search(query, limit, tags)]



class FakeEmbeddingCRUD:
    def __init__(self, pool): self.pool = pool

    def with_auth(self, auth):
        """Return self -- fakes don't need auth scoping."""
        return self

    async def store_embedding(self, guid: str, emb: List[float]) -> None:
        _store.embeddings[guid] = emb

    async def search_similar_documents(
        self, query_embedding, limit=10, similarity_threshold=0.0, tags=None,
        date_from=None, date_to=None,
    ) -> List[Tuple[str, float]]:
        results = []
        for guid, doc in _store.documents.items():
            if tags and not all(t in doc["tags"] for t in tags):
                continue
            results.append((guid, 0.9))
        return results[:limit]

    async def search_similar_chunks(
        self, query_embedding, limit=10, similarity_threshold=0.0, tags=None,
        date_from=None, date_to=None,
    ) -> List[Tuple[str, float]]:
        return await self.search_similar_documents(
            query_embedding, limit, similarity_threshold, tags,
            date_from, date_to,
        )

    async def has_chunks(self) -> bool:
        return False



class FakeMetadataCRUD:
    def __init__(self, pool): self.pool = pool

    def with_auth(self, auth):
        """Return self -- fakes don't need auth scoping."""
        return self

    async def get_metadata(self, guid: str): return _store.metadata.get(guid)

    async def upsert_metadata(self, guid: str, req: MetadataRequest) -> MetadataResponse:
        now = datetime.now(timezone.utc)
        m = MetadataResponse(guid=guid, doc_type=req.doc_type, metadata=req.metadata,
                             extracted_at=now, extractor_version=req.extractor_version)
        _store.metadata[guid] = m
        return m

    async def delete_metadata(self, guid: str) -> bool:
        if guid in _store.metadata:
            del _store.metadata[guid]
            return True
        return False

    async def list_by_type(self, doc_type: str, limit=100):
        return [v for v in _store.metadata.values()
                if isinstance(v, MetadataResponse) and v.doc_type == doc_type][:limit]



# ---------------------------------------------------------------------------
# Build the patched FastAPI application (once, at module load time)
# ---------------------------------------------------------------------------

_fake_pool = _make_fake_pool()

async def _fake_create_pool(*args, **kwargs): return _fake_pool
async def _fake_create_tables(pool): pass

_fake_doc_crud = FakeDocumentCRUD(_fake_pool)
_fake_emb_crud = FakeEmbeddingCRUD(_fake_pool)
_fake_meta_crud = FakeMetadataCRUD(_fake_pool)

with (
    mock.patch("asyncpg.create_pool", new=_fake_create_pool),
    mock.patch("mesh.database.create_tables", new=_fake_create_tables),
    mock.patch("mesh.embeddings.get_embedding_service", new=_get_fake_embedding_service),
    mock.patch("mesh.crud.DocumentCRUD", return_value=_fake_doc_crud),
    mock.patch("mesh.crud.EmbeddingCRUD", return_value=_fake_emb_crud),
    mock.patch("mesh.crud.MetadataCRUD", return_value=_fake_meta_crud),
):
    from mesh import main as _mesh_main

# Set module-level globals that the lifespan coroutine would normally set
_mesh_main.document_crud = _fake_doc_crud
_mesh_main.embedding_crud = _fake_emb_crud
_mesh_main.metadata_crud = _fake_meta_crud
_mesh_main.db_pool = _fake_pool
_mesh_main.migration_pool = _fake_pool
_mesh_main.embedding_queue = asyncio.Queue()

_app = _mesh_main.app


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_store():
    """Wipe the in-memory store and rate limiters before every test."""
    _store.reset()
    # Reset rate limiters to prevent cross-test 429 errors
    _mesh_main._search_limiter._hits.clear()
    _mesh_main._embed_limiter._hits.clear()
    _mesh_main._heavy_limiter._hits.clear()
    yield
    _store.reset()


@pytest.fixture(scope="session")
def app():
    return _app


@pytest_asyncio.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
