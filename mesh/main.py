#!/usr/bin/env python3
"""
Main FastAPI application for Mesh document management system
"""
import json
import os
import time
import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status, Query, Request, Depends
from mesh import __version__
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import ipaddress
import asyncpg
import asyncio
from contextlib import asynccontextmanager
import logging

from .models import (
    DocumentCreateRequest,
    DocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    MetadataRequest,
    MetadataResponse
)
from .categorizer.config import is_categorizer_enabled
from .config import (
    get_database_url, get_similarity_threshold, get_ip_whitelist,
    get_cors_origins, is_debug_enabled, get_rate_limit_search,
    get_rate_limit_embed, is_auth_required, get_api_keys,
    get_db_pool_min_size, get_db_pool_max_size, get_log_level,
    get_max_content_length, verify_baked_model
)
from .database import create_tables
from .embeddings import get_embedding_service
from .crud import DocumentCRUD, EmbeddingCRUD, MetadataCRUD
from .utils import generate_document_guid, validate_document_guid, normalize_tags
from .tag_schema import TagSchema

# Configure logging
logging.basicConfig(level=getattr(logging, get_log_level(), logging.INFO))
logger = logging.getLogger(__name__)

# IP Whitelist configuration (loaded from environment)
def get_allowed_networks():
    """Parse IP whitelist from config into network objects"""
    whitelist = get_ip_whitelist()
    if not whitelist:
        return None  # No whitelist = allow all
    networks = []
    for cidr in whitelist:
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError as e:
            logger.warning(f"Invalid IP range in whitelist: {cidr} - {e}")
    return networks if networks else None

ALLOWED_NETWORKS = get_allowed_networks()

# Tag schema (loaded from mesh.yaml)
tag_schema = TagSchema()


class RateLimiter:
    """Simple in-memory sliding window rate limiter per IP."""

    _PRUNE_INTERVAL = 300  # Prune stale keys every 5 minutes

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, collections.deque] = {}
        self._last_prune = time.monotonic()

    def _prune_stale(self, now: float) -> None:
        """Remove keys that have no hits within the window."""
        if now - self._last_prune < self._PRUNE_INTERVAL:
            return
        cutoff = now - self.window
        stale = [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
        self._last_prune = now

    def is_allowed(self, key: str) -> bool:
        if self.max_requests <= 0:
            return True
        now = time.monotonic()
        self._prune_stale(now)
        q = self._hits.setdefault(key, collections.deque())
        # Drop timestamps outside the window
        while q and q[0] <= now - self.window:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


_search_limiter = RateLimiter(get_rate_limit_search())
_embed_limiter = RateLimiter(get_rate_limit_embed())
_heavy_limiter = RateLimiter(10)  # 10 req/min for CPU-intensive endpoints


def _get_client_ip(request: Request) -> str:
    from .config import trust_proxy_headers
    if trust_proxy_headers():
        ip = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Real-IP")
        if ip:
            return ip
    return request.client.host if request.client else "127.0.0.1"


async def check_search_rate_limit(request: Request):
    """Dependency: enforce rate limit on search endpoints."""
    if not _search_limiter.is_allowed(_get_client_ip(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


async def check_embed_rate_limit(request: Request):
    """Dependency: enforce rate limit on embed endpoints."""
    if not _embed_limiter.is_allowed(_get_client_ip(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


async def check_heavy_rate_limit(request: Request):
    """Dependency: enforce rate limit on CPU-intensive endpoints."""
    if not _heavy_limiter.is_allowed(_get_client_ip(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """Restrict access to allowed IP ranges, reading real IP from Cloudflare headers"""

    async def dispatch(self, request: Request, call_next):
        # If no whitelist configured, allow all
        if ALLOWED_NETWORKS is None:
            return await call_next(request)

        # Get real client IP
        # Only trust proxy headers when explicitly configured (prevents header spoofing)
        from .config import trust_proxy_headers
        client_ip = None
        if trust_proxy_headers():
            client_ip = request.headers.get("CF-Connecting-IP")
            if not client_ip:
                client_ip = request.headers.get("X-Real-IP")
        if not client_ip:
            client_ip = request.client.host if request.client else "127.0.0.1"

        # Allow health checks from anywhere (for Docker healthcheck)
        if request.url.path == "/health":
            return await call_next(request)

        # Check if IP is allowed
        try:
            ip = ipaddress.ip_address(client_ip)
            allowed = any(ip in network for network in ALLOWED_NETWORKS)
        except ValueError:
            allowed = False

        if not allowed:
            logger.warning(f"Blocked request from {client_ip} to {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden"}
            )

        return await call_next(request)


# API Key authentication
_AUTH_REQUIRED = is_auth_required()
_API_KEYS = set(get_api_keys())

if _AUTH_REQUIRED and not _API_KEYS:
    raise RuntimeError(
        "AUTH_REQUIRED is true but API_KEYS is empty. "
        "Set API_KEYS environment variable or disable AUTH_REQUIRED."
    )

# Paths exempt from authentication (health probes, public help)
_AUTH_EXEMPT_PATHS = {"/health", "/help"}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require X-API-Key header when AUTH_REQUIRED=true"""

    async def dispatch(self, request: Request, call_next):
        if not _AUTH_REQUIRED:
            return await call_next(request)

        if request.url.path in _AUTH_EXEMPT_PATHS or request.url.path.startswith("/ui"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header"}
            )

        if api_key not in _API_KEYS:
            logger.warning(f"Invalid API key attempt from {_get_client_ip(request)}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"}
            )

        return await call_next(request)


# Global database pool and services
db_pool = None
document_crud = None
embedding_crud = None
metadata_crud = None

# Background embedding queue
embedding_queue: asyncio.Queue = None
embedding_worker_tasks = []
NUM_WORKERS = 3  # Parallel embedding workers

async def embedding_worker():
    """Background worker that processes embedding queue in thread pool"""
    global embedding_queue
    logger.info("Embedding worker started")
    loop = asyncio.get_running_loop()

    while True:
        try:
            # Wait for document guid from queue
            guid, content = await embedding_queue.get()

            try:
                # Generate embedding in thread pool (non-blocking)
                embedding_service = await get_embedding_service()
                embedding = await loop.run_in_executor(
                    None, embedding_service.generate_embedding, content
                )
                await embedding_crud.store_embedding(guid, embedding)
                _visualize_cache.clear()  # invalidate visualize cache
                logger.info(f"Indexed {guid} (queue: {embedding_queue.qsize()})")

                # Auto-infer tags from nearest neighbors
                if tag_schema.infer_enabled:
                    try:
                        await _infer_tags_from_neighbors(guid)
                    except Exception as infer_err:
                        logger.warning(f"Tag inference failed for {guid}: {infer_err}")

                # AI categorizer (if enabled and taxonomy exists)
                if is_categorizer_enabled():
                    try:
                        from .categorizer import classify as categorizer_classify
                        await categorizer_classify(guid, content, embedding, db_pool)
                    except Exception as cat_err:
                        logger.warning(f"Categorizer failed for {guid}: {cat_err}")
            except Exception as e:
                logger.error(f"Embedding failed for {guid}: {e}")
            finally:
                embedding_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Embedding worker stopped")
            break
        except Exception as e:
            logger.error(f"Embedding worker error: {e}")
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global db_pool, document_crud, embedding_crud, metadata_crud, embedding_queue, embedding_worker_tasks

    try:
        # Startup
        logger.info("Starting up Mesh API...")

        # Verify baked model matches runtime config
        verify_baked_model()

        # Initialize database
        db_pool = await asyncpg.create_pool(
            get_database_url(),
            min_size=get_db_pool_min_size(),
            max_size=get_db_pool_max_size()
        )
        await create_tables(db_pool)

        # Auto-seed demo data if ENVIRONMENT=demo and DB is empty
        if os.getenv("ENVIRONMENT") == "demo":
            async with db_pool.acquire() as conn:
                count = await conn.fetchval("SELECT count(*) FROM documents")
            if count == 0:
                seed_path = Path("/app/examples/demo-seed.jsonl")
                if seed_path.exists():
                    logger.info("Demo mode: seeding from demo-seed.jsonl...")
                    seeded = 0
                    with open(seed_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            doc = json.loads(line)
                            guid = generate_document_guid()
                            content = doc.get("content", "")
                            tags = normalize_tags(doc.get("tags", []))
                            source = doc.get("source", "demo")
                            created_at = doc.get("created_at")
                            if created_at and isinstance(created_at, str):
                                try:
                                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                except (ValueError, TypeError):
                                    created_at = None
                            async with db_pool.acquire() as conn:
                                await conn.execute(
                                    """INSERT INTO documents (guid, content, tags, source, created_at)
                                       VALUES ($1, $2, $3, $4, COALESCE($5, now()))""",
                                    guid, content, tags, source, created_at
                                )
                            seeded += 1
                    logger.info(f"Demo mode: seeded {seeded} documents")
                else:
                    logger.info("Demo mode: no seed file found at /app/examples/demo-seed.jsonl")

        # Initialize CRUD services
        document_crud = DocumentCRUD(db_pool)
        embedding_crud = EmbeddingCRUD(db_pool)
        metadata_crud = MetadataCRUD(db_pool)

        # Initialize embedding service
        embedding_service = await get_embedding_service()

        # Start background embedding workers
        embedding_queue = asyncio.Queue()
        for i in range(NUM_WORKERS):
            task = asyncio.create_task(embedding_worker())
            embedding_worker_tasks.append(task)
        logger.info(f"Started {NUM_WORKERS} embedding workers")

        # Requeue pending documents (no embeddings yet)
        async with db_pool.acquire() as conn:
            pending = await conn.fetch("""
                SELECT d.guid, d.content FROM documents d
                LEFT JOIN doc_embeddings e ON d.guid = e.guid
                WHERE e.guid IS NULL
            """)
            for row in pending:
                await embedding_queue.put((row['guid'], row['content'][:4000]))
            if pending:
                logger.info(f"Requeued {len(pending)} pending documents for indexing")

        # Initialize AI categorizer (if enabled)
        if is_categorizer_enabled():
            try:
                from .categorizer import init as categorizer_init
                await categorizer_init(db_pool)
                logger.info("AI categorizer enabled")
            except Exception as e:
                logger.warning(f"Categorizer init failed (continuing without): {e}")

        logger.info("Mesh API startup complete")

        # Warm up visualize cache in background
        async def _warmup_visualize():
            await asyncio.sleep(5)  # let workers start first
            try:
                import numpy as np
                from sklearn.decomposition import PCA
                async with db_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        WITH ranked AS (
                            SELECT d.guid, d.content AS preview, d.tags, d.created_at, e.embedding,
                                   ROW_NUMBER() OVER (ORDER BY d.created_at) as rn,
                                   COUNT(*) OVER () as total
                            FROM documents d
                            JOIN doc_embeddings e ON d.guid = e.guid
                        )
                        SELECT guid, preview, tags, created_at, embedding
                        FROM ranked
                        WHERE rn % GREATEST(1, total / 10000) = 0 OR total <= 10000
                        ORDER BY created_at
                        LIMIT 10000
                    """)
                    if len(rows) >= 5:
                        embeddings = []
                        documents = []
                        for row in rows:
                            emb = row['embedding']
                            if emb is not None:
                                if isinstance(emb, str):
                                    emb = np.array([float(x) for x in emb.strip('[]').split(',') if x.strip()])
                                elif isinstance(emb, (list, tuple)):
                                    emb = np.array(emb)
                                embeddings.append(emb)
                                doc_type = "unknown"
                                project = None
                                for t in (row['tags'] or []):
                                    if t.startswith('type:'): doc_type = t.split(':')[1]
                                    elif t.startswith('guid:'): project = t.split(':')[1]
                                documents.append({
                                    "guid": row['guid'], "type": doc_type, "project": project,
                                    "preview": row['preview'] or "",
                                    "tags": list(row['tags']) if row['tags'] else [],
                                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                                })
                        if len(embeddings) >= 5:
                            arr = np.array(embeddings)
                            coords = PCA(n_components=2).fit_transform(arr)
                            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                            x_range = x_max - x_min if x_max != x_min else 1
                            y_range = y_max - y_min if y_max != y_min else 1
                            for i, doc in enumerate(documents):
                                doc["x"] = float((coords[i, 0] - x_min) / x_range)
                                doc["y"] = float((coords[i, 1] - y_min) / y_range)
                                doc["cluster"] = -1
                            result = {"count": len(documents), "documents": documents}
                            _visualize_cache["10000:None:uniform"] = {"data": result, "ts": time.time()}
                            logger.info(f"Visualize cache warmed: {len(documents)} docs")
            except Exception as e:
                logger.warning(f"Visualize warmup failed: {e}")

        asyncio.create_task(_warmup_visualize())

        yield

    finally:
        # Shutdown
        logger.info("Shutting down Mesh API...")
        for task in embedding_worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if db_pool:
            await db_pool.close()
        logger.info("Mesh API shutdown complete")


async def _infer_tags_from_neighbors(guid: str):
    """Infer tags from nearest neighbors after embedding is stored."""
    n = tag_schema.infer_neighbors
    threshold = tag_schema.infer_threshold
    min_agree = tag_schema.infer_min_agreement
    allowed_prefixes = tag_schema.infer_prefixes

    if not allowed_prefixes:
        logger.debug(f"Inference skipped for {guid}: no infer prefixes configured")
        return

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tags FROM documents WHERE guid = $1", guid)
        if not row:
            return
        current_tags = set(row['tags'] or [])
        current_prefixes = {t.split(':')[0] + ':' for t in current_tags if ':' in t}

        neighbors = await conn.fetch(
            "SELECT d.guid, d.tags, "
            "1 - (e.embedding <=> (SELECT embedding FROM doc_embeddings WHERE guid = $1)) AS sim "
            "FROM documents d "
            "JOIN doc_embeddings e ON d.guid = e.guid "
            "WHERE d.guid != $1 "
            "AND 1 - (e.embedding <=> (SELECT embedding FROM doc_embeddings WHERE guid = $1)) >= $2 "
            "ORDER BY sim DESC "
            "LIMIT $3",
            guid, threshold, n
        )

        logger.info(f"Inference {guid}: {len(neighbors)} neighbors (threshold={threshold}), prefixes={allowed_prefixes}")

        if len(neighbors) < min_agree:
            logger.info(f"Inference {guid}: not enough neighbors ({len(neighbors)} < {min_agree})")
            return

        from collections import Counter
        tag_counts = Counter()
        for nb in neighbors:
            nb_tags = nb['tags'] or []
            for tag in nb_tags:
                if ':' in tag:
                    prefix = tag.split(':')[0] + ':'
                    if prefix in allowed_prefixes:
                        tag_counts[tag] += 1

        logger.info(f"Inference {guid}: tag_counts={dict(tag_counts.most_common(10))}")

        new_tags = []
        for tag, count in tag_counts.items():
            if count >= min_agree and tag not in current_tags:
                prefix = tag.split(':')[0] + ':'
                if prefix not in current_prefixes:
                    new_tags.append(tag)

        if new_tags:
            updated_tags = list(current_tags) + new_tags
            await conn.execute(
                "UPDATE documents SET tags = $1 WHERE guid = $2",
                updated_tags, guid
            )
            logger.info(f"Auto-tagged {guid}: +{new_tags}")
        else:
            logger.info(f"Inference {guid}: no new tags to add (candidates with agreement: {[(t,c) for t,c in tag_counts.items() if c>=min_agree]})")



app = FastAPI(
    title="Mesh Document Management API",
    description="Document management with automatic classification and semantic search",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware (origins from config)
_cors_origins = get_cors_origins()
if "*" in _cors_origins:
    logger.warning(
        "CORS_ORIGINS contains '*' — allow_credentials is disabled. "
        "Browsers reject credentials with wildcard origins per CORS spec. "
        "Set explicit origins to enable credentials."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Starlette executes middleware in LIFO order of add_middleware calls.
# Register auth FIRST, then IP whitelist LAST so IP check runs first at request time.
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(IPWhitelistMiddleware)

# Mount categorizer router (if enabled)
if is_categorizer_enabled():
    from .categorizer.router import router as categorizer_router
    app.include_router(categorizer_router)

# ---- On-demand document summarization ----
@app.post("/summarize/{guid}", dependencies=[Depends(check_heavy_rate_limit)])
async def summarize_document(guid: str):
    """Generate a 2-3 sentence summary for a document using LLM. Cached in DB."""
    from .categorizer.config import get_llm_api_url, get_llm_api_key, get_llm_model, get_llm_timeout, is_llm_available
    import httpx

    if not is_llm_available():
        raise HTTPException(status_code=503, detail="LLM not configured")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content, summary FROM documents WHERE guid = $1", guid
        )
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        # Return cached summary if exists
        if row['summary']:
            return {"guid": guid, "summary": row['summary'], "cached": True}

        content = (row['content'] or "")[:3000]
        prompt = (
            "Write a concise summary of this document in 2-3 sentences. "
            "Focus on the key facts and purpose. Reply with ONLY the summary text, no formatting.\n\n"
            f"Document:\n{content}"
        )

        try:
            async with httpx.AsyncClient(timeout=get_llm_timeout()) as client:
                resp = await client.post(
                    get_llm_api_url().rstrip("/") + "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {get_llm_api_key()}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": get_llm_model(),
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                summary = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Summarize LLM call failed: {e}")
            raise HTTPException(status_code=502, detail="LLM call failed")

        # Store in DB
        await conn.execute(
            "UPDATE documents SET summary = $1 WHERE guid = $2", summary, guid
        )

        return {"guid": guid, "summary": summary, "cached": False}

# Serve built-in web UI (search + map) from /ui/ path
# Add no-cache headers for JS/CSS so Cloudflare doesn't serve stale modules
from pathlib import Path as _Path

@app.middleware("http")
async def ui_no_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/ui/") and (path.endswith(".js") or path.endswith(".css")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        response.headers["CDN-Cache-Control"] = "no-cache"
    return response

_ui_dir = _Path(__file__).resolve().parent.parent / "ui"
if _ui_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Hide internal error details in production mode"""
    detail = exc.detail
    if exc.status_code >= 500 and not is_debug_enabled():
        detail = "Internal server error"
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})

@app.put("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED,
         dependencies=[Depends(check_heavy_rate_limit)])
async def create_document(request: DocumentCreateRequest):
    """Create a new document with automatic GUID generation and async embedding"""
    try:
        # Validate input length
        content_stripped = request.content.strip()
        if len(content_stripped) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content too short ({len(content_stripped)} chars). Minimum 10 characters required."
            )
        max_len = get_max_content_length()
        if len(request.content) > max_len:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content too long ({len(request.content)} chars). Maximum {max_len} characters."
            )

        # Normalize tags
        normalized_tags = normalize_tags(request.tags)

        # Apply schema defaults (date, source) if not already present
        normalized_tags = tag_schema.apply_defaults(normalized_tags, source=request.source)
        request.tags = normalized_tags

        # Advisory validation
        warnings = tag_schema.validate(normalized_tags)
        for w in warnings:
            logger.info(f"Tag validation: {w}")

        # Generate GUID
        guid = generate_document_guid()

        # Create document in database
        document = await document_crud.create_document(guid, request)

        # Queue embedding generation (async, non-blocking)
        await embedding_queue.put((guid, request.content[:4000]))

        logger.info(f"Created document {guid}, queued for embedding")
        return document

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error in create_document: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document data"
        )
    except Exception as e:
        logger.error(f"Failed to create document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create document"
        )

@app.put("/doc/{guid}", response_model=DocumentResponse,
         dependencies=[Depends(check_heavy_rate_limit)])
async def upsert_document(guid: str, request: DocumentCreateRequest):
    """Create or update a document with a specific GUID (upsert)"""
    try:
        # Validate input length
        content_stripped = request.content.strip()
        if len(content_stripped) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content too short ({len(content_stripped)} chars). Minimum 10 characters required."
            )
        max_len = get_max_content_length()
        if len(request.content) > max_len:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content too long ({len(request.content)} chars). Maximum {max_len} characters."
            )

        # Normalize tags
        normalized_tags = normalize_tags(request.tags)

        # Apply schema defaults (date, source) if not already present
        normalized_tags = tag_schema.apply_defaults(normalized_tags, source=request.source)
        request.tags = normalized_tags

        # Upsert document in database
        document = await document_crud.upsert_document(guid, request)

        # Queue embedding generation (async, non-blocking)
        await embedding_queue.put((guid, request.content[:4000]))

        logger.info(f"Upserted document {guid}, queued for embedding")
        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upsert document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upsert document"
        )

@app.get("/")
async def root_endpoint(
    limit: int = Query(default=10, ge=1, le=100, description="Maximum documents (default 10)"),
    offset: int = Query(default=0, ge=0, description="Skip documents"),
    guid: Optional[str] = Query(default=None, description="Filter by project GUID"),
    project: Optional[str] = Query(default=None, description="Filter by project name"),
    type: Optional[str] = Query(default=None, description="Filter by document type (worklog, note, etc)"),
    source: Optional[str] = Query(default=None, description="Filter by source (api, cli, voice, chat)"),
    tag: Optional[List[str]] = Query(default=None, description="Filter by tag(s). Multiple tags use AND logic."),
    list_all: bool = Query(default=False, alias="all", description="List all documents (no filter, sorted by date desc)")
):
    """Root endpoint: returns README from memory if available, otherwise help. With params, returns documents."""
    # If no parameters provided (just browsing), return README from memory or fallback help
    has_filters = guid or project or type or source or tag or list_all
    if not has_filters:
        # Try to fetch the latest README from memory
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    readme_doc = await conn.fetchrow("""
                        SELECT guid, content, created_at
                        FROM documents
                        WHERE 'type:readme' = ANY(tags) AND 'topic:mesh-api' = ANY(tags)
                        ORDER BY created_at DESC
                        LIMIT 1
                    """)

                if readme_doc:
                    return {
                        "service": "Mesh - Document Management & Semantic Search API",
                        "readme": readme_doc['content'],
                        "last_updated": readme_doc['created_at'].isoformat() if readme_doc['created_at'] else None,
                        "source": "from memory (type:readme, topic:mesh-api)",
                        "note": "To update this README, create a new document in memory with tags: type:readme, topic:mesh-api"
                    }
            except Exception as e:
                logger.error(f"Failed to load README from memory: {e}")

        # Fallback: return built-in help
        return {
            "service": "Mesh - Document Management & Semantic Search API",
            "version": __version__,
            "quick_start": {
                "create_doc": "curl -X PUT $API_URL/ -H 'Content-Type: application/json' -d '{\"content\":\"text\",\"tags\":[\"tag1\",\"tag2\"]}'",
                "search": "curl -X POST $API_URL/search -H 'Content-Type: application/json' -d '{\"query\":\"find this\",\"limit\":5}'",
                "list": "curl '$API_URL/?limit=10'",
                "list_by_tag": "curl '$API_URL/?tag=type:worklog&limit=10'",
                "health": "curl $API_URL/health",
                "stats": "curl $API_URL/stats"
            },
            "endpoints": {
                "PUT /": {
                    "description": "Create document with auto-GUID and async embedding",
                    "body": {
                        "content": "string (required) - document text",
                        "tags": "array (optional) - tags for document",
                        "source": "string (optional) - where doc came from (api, cli, voice, etc)"
                    },
                    "returns": "document with guid, created_at, etc"
                },
                "GET /": {
                    "description": "Root help OR list documents (depends on parameters)",
                    "params": {
                        "limit": "1-100 - how many docs to return",
                        "offset": "0+ - skip this many docs",
                        "tag": "filter by exact tag (e.g. 'type:worklog' or 'guid:abc123')"
                    },
                    "note": "No params = this help. With params = documents."
                },
                "POST /search": {
                    "description": "Semantic search - finds similar documents by meaning",
                    "body": {
                        "query": "string (required) - what to search for",
                        "limit": "number (optional, default 10) - max results"
                    },
                    "returns": "list of documents sorted by relevance score"
                },
                "GET /{guid}": {
                    "description": "Get one document by GUID",
                    "example": "GET /doc_12345678"
                },
                "GET /health": {
                    "description": "Health check - is API alive and DB connected"
                },
                "GET /stats": {
                    "description": "Statistics - document count, embedding queue, tags, etc"
                },
                "GET /activity": {
                    "description": "Activity by tags with last update times",
                    "params": {
                        "prefix": "filter by tag prefix (e.g. 'guid:', 'type:')",
                        "limit": "max tags to return"
                    }
                },
                "PUT /bulk": {
                    "description": "Create multiple documents at once",
                    "body": "array of {content, tags, source}"
                },
                "GET /by-hash/{hash}": {
                    "description": "Find document by MD5 content hash"
                },
                "PATCH /by-hash": {
                    "description": "Update document by MD5 hash"
                },
                "GET /help": {
                    "description": "Full API documentation (this help)"
                }
            },
            "curl_examples": {
                "create": "curl -X PUT $API_URL/ -H 'Content-Type: application/json' -d '{\"content\":\"My note\",\"tags\":[\"type:note\",\"date:2025-12-30\"]}'",
                "search": "curl -X POST $API_URL/search -H 'Content-Type: application/json' -d '{\"query\":\"PostgreSQL optimization\",\"limit\":3}'",
                "list_recent": "curl '$API_URL/?limit=5'",
                "list_worklog": "curl '$API_URL/?tag=type:worklog&limit=10'",
                "list_project": "curl '$API_URL/?tag=guid:myproject&limit=10'",
                "get_doc": "curl $API_URL/doc_12345678",
                "health_check": "curl $API_URL/health"
            },
            "tag_conventions": {
                "type": "worklog, note, decision, research, artifact, task, rfc, discussion, idea",
                "date": "YYYY-MM-DD format (2025-12-30)",
                "guid": "project GUID for filtering (guid:abc123def4)",
                "status": "draft, proposed, approved, rejected, active, completed, archived",
                "stage": "ideation, conception, design, implementation, testing",
                "agent": "ai-mic, conductor, team-deliberation, team-generic, team-soft, team-soft-2",
                "source": "where document came from: api, cli, voice, chat, web, etc"
            },
            "common_queries": {
                "find_today_worklogs": "GET /?tag=type:worklog&tag=date:2025-12-30",
                "search_topic": "POST /search with query in JSON body",
                "find_project_docs": "GET /?tag=guid:PROJECT_GUID",
                "list_by_source": "GET /?tag=source:voice",
                "recent_notes": "GET /?tag=type:note&limit=10"
            },
            "response_example": {
                "created_doc": {
                    "guid": "doc_a1b2c3d4",
                    "content": "document text here",
                    "tags": ["type:note", "date:2025-12-30"],
                    "created_at": "2025-12-30T12:00:00Z",
                    "source": "api"
                }
            },
            "docs": "/docs (interactive Swagger UI)"
        }

    # With parameters - return documents
    try:
        # Build filter tags from new parameters
        filter_tags = []

        if guid:
            filter_tags.append(f"guid:{guid}")
        if project:
            filter_tags.append(f"project:{project}")
        if type:
            filter_tags.append(f"type:{type}")
        if source:
            filter_tags.append(f"source:{source}")
        if tag:
            # Support multiple tag= params (AND logic)
            for t in tag:
                filter_tags.append(t)

        # Fetch documents
        async with db_pool.acquire() as conn:
            if filter_tags:
                # Filter by ALL tags (AND logic)
                query = """
                    SELECT guid, content, content_hash, filename, source,
                           tags, created_at, updated_at
                    FROM documents
                    WHERE """

                # Each filter tag must be present
                for i, filter_tag in enumerate(filter_tags):
                    if i > 0:
                        query += " AND "
                    query += f"${ i + 1 }::text = ANY(tags)"

                query += f"""
                    ORDER BY created_at DESC
                    LIMIT ${len(filter_tags) + 1} OFFSET ${len(filter_tags) + 2}
                """

                rows = await conn.fetch(query, *filter_tags, limit, offset)
            else:
                # No filters - return recent documents
                rows = await conn.fetch("""
                    SELECT guid, content, content_hash, filename, source,
                           tags, created_at, updated_at
                    FROM documents
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset)

            documents = [
                DocumentResponse(
                    guid=row['guid'],
                    content=row['content'],
                    content_hash=row['content_hash'],
                    filename=row['filename'],
                    source=row['source'],
                    tags=list(row['tags']) if row['tags'] else [],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    directory="inbox"
                )
                for row in rows
            ]

        return documents

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

        # Check embedding service
        embedding_service = await get_embedding_service()
        if not embedding_service.model:
            return {"status": "unhealthy", "service": "mesh-api", "error": "embedding service not ready"}

        return {
            "status": "healthy",
            "service": "mesh-api",
            "database": "connected",
            "embeddings": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "mesh-api",
            "error": "database unavailable"
        }

@app.get("/help")
async def help_endpoint():
    """API documentation and help"""
    return {
        "service": "Mesh - Document Management & Semantic Search",
        "version": __version__,
        "endpoints": {
            "PUT /": "Create document with auto-GUID and async embedding",
            "PUT /doc/{guid}": "Create or update document with specific GUID",
            "GET /": "List documents with pagination and tag filtering",
            "POST /search": "Semantic search with query embedding",
            "GET /{guid}": "Get document by GUID",
            "DELETE /{guid}": "Delete document by GUID",
            "PATCH /{guid}": "Update document tags or content",
            "GET /health": "Health check (database + embeddings)",
            "GET /stats": "API statistics (doc count, queue size, etc)",
            "GET /activity": "Activity by tag prefix with last update times",
            "PUT /bulk": "Bulk create documents",
            "GET /by-hash/{content_hash}": "Find document by MD5 hash",
            "PATCH /by-hash": "Update document by hash",
            "POST /embed": "Generate embedding vector for text",
            "POST /embed/batch": "Generate embeddings for multiple texts",
            "GET/PUT/DELETE /meta/{guid}": "Document metadata CRUD",
            "GET /versions/{guid}": "Version chain via embedding similarity",
            "GET /short": "Find documents with short content (for cleanup)",
            "GET /tags": "List unique tags with document counts",
            "GET /schema": "Tag schema configuration",
            "GET /visualize": "2D document map (UMAP + HDBSCAN clustering)",
            "GET /help": "This help endpoint"
        },
        "examples": {
            "create": {
                "method": "PUT",
                "path": "/",
                "body": {"content": "document text", "tags": ["tag1", "tag2"], "source": "api"}
            },
            "search": {
                "method": "POST",
                "path": "/search",
                "body": {"query": "what to find", "limit": 10}
            },
            "list": {
                "method": "GET",
                "path": "/?limit=10&offset=0&tag=type:worklog"
            },
            "get": {
                "method": "GET",
                "path": "/{guid}"
            }
        },
        "docs": "/docs"
    }

# Server-side query embedding cache (same text -> same vector, deterministic)
_query_embedding_cache: dict = {}  # text -> (embedding, timestamp)
_QUERY_CACHE_TTL = 3600  # 1 hour
_QUERY_CACHE_MAX = 500   # max entries


def _get_cached_embedding(text: str):
    """Return cached embedding or None."""
    entry = _query_embedding_cache.get(text)
    if entry and (time.time() - entry[1]) < _QUERY_CACHE_TTL:
        return entry[0]
    return None


def _put_cached_embedding(text: str, embedding):
    """Store embedding in cache, evict oldest if full."""
    if len(_query_embedding_cache) >= _QUERY_CACHE_MAX:
        oldest_key = min(_query_embedding_cache, key=lambda k: _query_embedding_cache[k][1])
        del _query_embedding_cache[oldest_key]
    _query_embedding_cache[text] = (embedding, time.time())


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(check_search_rate_limit)])
async def search_documents(request: SearchRequest):
    """Hybrid search: semantic similarity + keyword fallback.

    Runs semantic search first, then checks if the query text appears
    literally in any results.  If not, a keyword (ILIKE) search is
    performed and its hits are appended so that exact product names,
    acronyms, etc. are always surfaced.
    """
    try:
        # Validate query
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )

        limit = request.limit or 10

        # Generate embedding for query (cached to avoid 20s CPU inference)
        query_embedding = _get_cached_embedding(request.query)
        if query_embedding is None:
            embedding_service = await get_embedding_service()
            loop = asyncio.get_running_loop()
            query_embedding = await loop.run_in_executor(
                None, embedding_service.generate_embedding, request.query
            )
            _put_cached_embedding(request.query, query_embedding)

        # Semantic search (optionally filtered by tags)
        similar_docs = await embedding_crud.search_similar_documents(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=get_similarity_threshold(),
            tags=request.tags
        )

        # Build results from semantic search (batch fetch to avoid N+1)
        results = []
        seen_guids = set()
        if similar_docs:
            score_map = {guid: score for guid, score in similar_docs}
            guids = list(score_map.keys())
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT guid, content, content_hash, filename, source,
                              tags, created_at, updated_at
                       FROM documents WHERE guid = ANY($1)""",
                    guids
                )
            for row in rows:
                g = row['guid']
                results.append(SearchResult(
                    guid=g,
                    content=row['content'],
                    tags=list(row['tags']) if row['tags'] else [],
                    created_at=row['created_at'],
                    similarity_score=round(score_map[g], 3),
                    directory="inbox"
                ))
                seen_guids.add(g)
            # Preserve similarity order
            results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Check if any semantic result contains the query literally
        query_lower = request.query.lower()
        has_exact_match = any(query_lower in r.content.lower() for r in results)

        # If no exact match found, run keyword search and prepend hits
        if not has_exact_match and len(query_lower) >= 3:
            keyword_results = await document_crud.keyword_search(
                query=request.query,
                limit=limit,
                tags=request.tags
            )
            keyword_hits = []
            for doc in keyword_results:
                if doc.guid not in seen_guids:
                    keyword_hits.append(SearchResult(
                        guid=doc.guid,
                        content=doc.content,
                        tags=doc.tags,
                        created_at=doc.created_at,
                        similarity_score=0.5,
                        directory=doc.directory
                    ))
                    seen_guids.add(doc.guid)
            # Put keyword matches first -- exact text hits are more relevant
            results = keyword_hits + results

        return SearchResponse(
            results=results,
            total_count=len(results),
            query=request.query
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search documents"
        )

@app.get("/browse")
async def browse_all_documents(
    preview: int = Query(default=300, ge=50, le=1000, description="Preview length"),
):
    """Return all documents with short previews for client-side search."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT guid,
                       LEFT(content, $1) AS preview,
                       tags,
                       created_at
                FROM documents
                ORDER BY created_at DESC
            """, preview)

        docs = []
        for row in rows:
            doc_type = "unknown"
            for t in (row['tags'] or []):
                if t.startswith('type:'):
                    doc_type = t[5:]
                    break
            docs.append({
                "g": row['guid'],
                "p": row['preview'] or "",
                "t": list(row['tags']) if row['tags'] else [],
                "y": doc_type,
                "d": row['created_at'].isoformat() if row['created_at'] else None
            })

        return {"count": len(docs), "docs": docs}
    except Exception as e:
        logger.error(f"Browse failed: {e}")
        raise HTTPException(status_code=500, detail="Browse failed")


@app.get("/keyword", dependencies=[Depends(check_search_rate_limit)])
async def keyword_search(
    q: str = Query(..., min_length=1, description="Keyword to search for"),
    limit: int = Query(default=20, ge=1, le=100),
    tag: Optional[str] = Query(default=None, description="Filter by tag")
):
    """Fast keyword search (ILIKE) -- no embedding, instant results."""
    try:
        tags = [tag] if tag else None
        docs = await document_crud.keyword_search(query=q, limit=limit, tags=tags)
        return {
            "results": [
                {
                    "guid": d.guid,
                    "content": d.content,
                    "tags": d.tags,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "similarity_score": None,
                    "directory": d.directory
                }
                for d in docs
            ],
            "total_count": len(docs),
            "query": q,
            "mode": "keyword"
        }
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        raise HTTPException(status_code=500, detail="Keyword search failed")


@app.post("/embed", dependencies=[Depends(check_embed_rate_limit)])
async def generate_embedding(request: dict):
    """Generate embedding vector for given text. Returns 768-dim vector."""
    text = request.get("text", "")
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="text field is required"
        )
    try:
        service = await get_embedding_service()
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, service.generate_embedding, text)
        return {"embedding": embedding, "dimensions": len(embedding), "model": service.model_name}
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embedding"
        )


@app.post("/embed/batch", dependencies=[Depends(check_embed_rate_limit)])
async def generate_embeddings_batch(request: dict):
    """Generate embeddings for multiple texts. Max 100 texts per request."""
    texts = request.get("texts", [])
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="texts array is required"
        )
    if len(texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Max 100 texts per batch"
        )
    try:
        service = await get_embedding_service()
        loop = asyncio.get_running_loop()
        valid_texts = [t for t in texts if t.strip()]
        embeddings = await loop.run_in_executor(None, service.generate_embeddings_batch, valid_texts)
        return {"embeddings": embeddings, "dimensions": len(embeddings[0]) if embeddings else 0, "count": len(embeddings), "model": service.model_name}
    except Exception as e:
        logger.error(f"Failed to generate embeddings batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings"
        )


@app.get("/stats")
async def get_stats():
    """Get API statistics including queue status"""
    try:
        async with db_pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            emb_count = await conn.fetchval("SELECT COUNT(*) FROM doc_embeddings")

            # Count unique projects from tags like "guid:*"
            project_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT tag) FROM (
                    SELECT UNNEST(tags) as tag FROM documents
                ) t WHERE tag LIKE 'guid:%'
            """)

            # Count all unique tags
            tag_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT tag) FROM (
                    SELECT UNNEST(tags) as tag FROM documents
                ) t
            """)

            # Last update
            last_update = await conn.fetchval(
                "SELECT MAX(created_at) FROM documents"
            )

        return {
            "documents": doc_count,
            "indexed": emb_count,
            "pending": doc_count - emb_count,
            "projects": project_count or 0,
            "tags": tag_count or 0,
            "queue_size": embedding_queue.qsize() if embedding_queue else 0,
            "last_update": last_update.isoformat() if last_update else None
        }
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stats"
        )

@app.get("/short")
async def get_short_documents(
    max_length: int = Query(default=10, ge=1, le=100, description="Maximum content length"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum documents to return")
):
    """Find documents with short content (for cleanup)"""
    try:
        if not db_pool:
            return {"error": "Database not available"}

        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT guid, content, tags, created_at, LENGTH(TRIM(content)) as content_length
                FROM documents
                WHERE LENGTH(TRIM(content)) < $1
                ORDER BY content_length ASC, created_at DESC
                LIMIT $2
            """, max_length, limit)

        return [
            {
                "guid": row["guid"],
                "content": row["content"],
                "tags": row["tags"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "content_length": row["content_length"]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Short docs query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query short documents"
        )

@app.get("/tags")
async def get_tags(
    prefix: Optional[str] = Query(default=None, description="Filter tags by prefix (e.g. 'guid:', 'type:', 'source:')"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum tags to return")
):
    """Get list of all unique tags with document counts"""
    try:
        async with db_pool.acquire() as conn:
            if prefix:
                # Filter by prefix
                rows = await conn.fetch("""
                    SELECT
                        tag,
                        COUNT(*) as doc_count
                    FROM (
                        SELECT UNNEST(tags) as tag FROM documents
                    ) t
                    WHERE tag LIKE $1
                    GROUP BY tag
                    ORDER BY doc_count DESC, tag ASC
                    LIMIT $2
                """, f"{prefix}%", limit)
            else:
                # All tags
                rows = await conn.fetch("""
                    SELECT
                        tag,
                        COUNT(*) as doc_count
                    FROM (
                        SELECT UNNEST(tags) as tag FROM documents
                    ) t
                    GROUP BY tag
                    ORDER BY doc_count DESC, tag ASC
                    LIMIT $1
                """, limit)

            return {
                "tags": [
                    {
                        "tag": row['tag'],
                        "count": row['doc_count']
                    }
                    for row in rows
                ],
                "total": len(rows),
                "filter": {
                    "prefix": prefix or "all",
                    "limit": limit
                },
                "examples": {
                    "all_tags": "GET /tags",
                    "projects": "GET /tags?prefix=guid:",
                    "document_types": "GET /tags?prefix=type:",
                    "sources": "GET /tags?prefix=source:",
                    "statuses": "GET /tags?prefix=status:"
                }
            }
    except Exception as e:
        logger.error(f"Tags failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tags"
        )

@app.get("/activity")
async def get_activity(
    prefix: Optional[str] = Query(default=None, description="Tag prefix filter (e.g., 'guid:', 'type:', 'project:')"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum tags to return")
):
    """Get activity by tags - shows last update date for each tag"""
    try:
        async with db_pool.acquire() as conn:
            if prefix:
                # Filter by tag prefix
                rows = await conn.fetch("""
                    SELECT
                        tag,
                        COUNT(*) as doc_count,
                        MAX(updated_at) as last_activity,
                        MIN(created_at) as first_activity
                    FROM (
                        SELECT UNNEST(tags) as tag, created_at, updated_at FROM documents
                    ) t
                    WHERE tag LIKE $1
                    GROUP BY tag
                    ORDER BY last_activity DESC
                    LIMIT $2
                """, f"{prefix}%", limit)
            else:
                # All tags
                rows = await conn.fetch("""
                    SELECT
                        tag,
                        COUNT(*) as doc_count,
                        MAX(updated_at) as last_activity,
                        MIN(created_at) as first_activity
                    FROM (
                        SELECT UNNEST(tags) as tag, created_at, updated_at FROM documents
                    ) t
                    GROUP BY tag
                    ORDER BY last_activity DESC
                    LIMIT $1
                """, limit)

            return [
                {
                    "tag": row['tag'],
                    "docs": row['doc_count'],
                    "last": row['last_activity'].isoformat() if row['last_activity'] else None,
                    "first": row['first_activity'].isoformat() if row['first_activity'] else None
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Activity failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve activity"
        )

MAX_BULK_DOCUMENTS = 100

@app.put("/bulk", dependencies=[Depends(check_heavy_rate_limit)])
async def bulk_create_documents(documents: List[DocumentCreateRequest]):
    """Bulk create documents - saves immediately, embeddings queued. Max 100 per request."""
    if len(documents) > MAX_BULK_DOCUMENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many documents ({len(documents)}). Maximum {MAX_BULK_DOCUMENTS} per request."
        )
    created = 0
    skipped = 0
    for doc in documents:
        try:
            # Skip documents with content < 10 chars
            if len(doc.content.strip()) < 10:
                skipped += 1
                continue

            normalized_tags = normalize_tags(doc.tags)
            doc.tags = normalized_tags
            guid = generate_document_guid()

            await document_crud.create_document(guid, doc)
            await embedding_queue.put((guid, doc.content[:4000]))
            created += 1
        except Exception as e:
            logger.error(f"Bulk create error: {e}")

    logger.info(f"Bulk created {created} documents, skipped {skipped}")
    return {"created": created, "skipped": skipped, "queue_size": embedding_queue.qsize()}

@app.get("/by-hash/{content_hash}", dependencies=[Depends(check_heavy_rate_limit)])
async def get_document_by_hash(content_hash: str):
    """Find document by content MD5 hash"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT guid, content_hash, created_at, updated_at FROM documents WHERE content_hash = $1",
                content_hash
            )
            if not row:
                return {"found": False}
            return {
                "found": True,
                "guid": row['guid'],
                "content_hash": row['content_hash'],
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
            }
    except Exception as e:
        logger.error(f"Hash lookup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to look up document by hash"
        )

from pydantic import BaseModel

class HashUpdateRequest(BaseModel):
    content_hash: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    content: Optional[str] = None  # For updating content

class DocumentUpdateRequest(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    add_tags: Optional[List[str]] = None  # Add to existing tags
    remove_tags: Optional[List[str]] = None  # Remove from existing tags

@app.patch("/by-hash", dependencies=[Depends(check_heavy_rate_limit)])
async def update_document_by_hash(request: HashUpdateRequest):
    """Update document dates or content by MD5 hash"""
    try:
        async with db_pool.acquire() as conn:
            # Find document
            row = await conn.fetchrow(
                "SELECT guid FROM documents WHERE content_hash = $1",
                request.content_hash
            )
            if not row:
                return {"updated": False, "error": "Document not found"}

            guid = row['guid']
            updates = []
            params = []
            param_idx = 1

            if request.created_at:
                updates.append(f"created_at = ${param_idx}")
                params.append(request.created_at)
                param_idx += 1

            if request.updated_at:
                updates.append(f"updated_at = ${param_idx}")
                params.append(request.updated_at)
                param_idx += 1

            if request.content:
                import hashlib
                new_hash = hashlib.md5(request.content.encode('utf-8')).hexdigest()
                updates.append(f"content = ${param_idx}")
                params.append(request.content)
                param_idx += 1
                updates.append(f"content_hash = ${param_idx}")
                params.append(new_hash)
                param_idx += 1

            if not updates:
                return {"updated": False, "error": "No fields to update"}

            params.append(request.content_hash)
            query = f"UPDATE documents SET {', '.join(updates)} WHERE content_hash = ${param_idx}"
            await conn.execute(query, *params)

            return {"updated": True, "guid": guid}
    except Exception as e:
        logger.error(f"Hash update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document by hash"
        )

@app.delete("/{guid}", dependencies=[Depends(check_heavy_rate_limit)])
async def delete_document(guid: str):
    """Delete a document by its GUID"""
    try:
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GUID format: {guid}. Expected format: doc_{{8_hex_chars}}"
            )

        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with db_pool.acquire() as conn:
            result = await conn.execute("DELETE FROM documents WHERE guid = $1", guid)
            # result format: "DELETE N" where N is number of rows deleted
            deleted_count = int(result.split()[-1])

        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with GUID {guid} not found"
            )

        logger.info(f"Deleted document: {guid}")
        return {"deleted": True, "guid": guid}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@app.patch("/{guid}", dependencies=[Depends(check_heavy_rate_limit)])
async def update_document(guid: str, request: DocumentUpdateRequest):
    """Update a document by GUID - supports content and tag modifications"""
    try:
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GUID format: {guid}"
            )

        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with db_pool.acquire() as conn:
            # Get current document
            row = await conn.fetchrow(
                "SELECT content, tags FROM documents WHERE guid = $1", guid
            )
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {guid} not found"
                )

            updates = []
            params = []
            param_idx = 1

            # Update content if provided
            if request.content is not None:
                import hashlib
                new_hash = hashlib.md5(request.content.encode('utf-8')).hexdigest()
                updates.append(f"content = ${param_idx}")
                params.append(request.content)
                param_idx += 1
                updates.append(f"content_hash = ${param_idx}")
                params.append(new_hash)
                param_idx += 1

            # Handle tags
            current_tags = list(row['tags']) if row['tags'] else []

            if request.tags is not None:
                # Replace all tags
                current_tags = request.tags
            else:
                # Add/remove individual tags
                if request.add_tags:
                    for tag in request.add_tags:
                        if tag not in current_tags:
                            current_tags.append(tag)
                if request.remove_tags:
                    current_tags = [t for t in current_tags if t not in request.remove_tags]

            if request.tags is not None or request.add_tags or request.remove_tags:
                updates.append(f"tags = ${param_idx}")
                params.append(current_tags)
                param_idx += 1

            # Always update updated_at
            updates.append(f"updated_at = ${param_idx}")
            params.append(datetime.now(timezone.utc))
            param_idx += 1

            if len(updates) == 1:  # Only updated_at
                return {"updated": False, "message": "No changes requested"}

            params.append(guid)
            query = f"UPDATE documents SET {', '.join(updates)} WHERE guid = ${param_idx}"
            await conn.execute(query, *params)

            logger.info(f"Updated document: {guid}")
            return {"updated": True, "guid": guid, "tags": current_tags}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document"
        )


# ============== METADATA ENDPOINTS ==============

@app.get("/meta/{guid}", response_model=MetadataResponse, dependencies=[Depends(check_search_rate_limit)])
async def get_metadata(guid: str):
    """Get metadata for a document"""
    try:
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GUID format: {guid}"
            )

        metadata = await metadata_crud.get_metadata(guid)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metadata found for document {guid}"
            )

        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata for {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metadata"
        )


@app.put("/meta/{guid}", response_model=MetadataResponse, dependencies=[Depends(check_search_rate_limit)])
async def upsert_metadata(guid: str, request: MetadataRequest):
    """Create or update metadata for a document"""
    try:
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GUID format: {guid}"
            )

        metadata = await metadata_crud.upsert_metadata(guid, request)
        logger.info(f"Upserted metadata for {guid}: type={request.doc_type}")
        return metadata

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {guid} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upsert metadata for {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save metadata"
        )


@app.delete("/meta/{guid}", dependencies=[Depends(check_search_rate_limit)])
async def delete_metadata(guid: str):
    """Delete metadata for a document"""
    try:
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GUID format: {guid}"
            )

        deleted = await metadata_crud.delete_metadata(guid)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metadata found for document {guid}"
            )

        return {"deleted": True, "guid": guid}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete metadata for {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete metadata"
        )


@app.get("/meta", response_model=list, dependencies=[Depends(check_search_rate_limit)])
async def list_metadata(
    type: Optional[str] = Query(default=None, description="Filter by doc_type"),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """List metadata records, optionally filtered by type"""
    try:
        if type:
            results = await metadata_crud.list_by_type(type, limit)
        else:
            # List all - get from database directly
            import json
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT guid, doc_type, metadata, extracted_at, extractor_version
                    FROM document_metadata
                    ORDER BY extracted_at DESC
                    LIMIT $1
                    """,
                    limit
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
        logger.error(f"Failed to list metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list metadata"
        )


# ============== VISUALIZATION ENDPOINT ==============

# In-memory cache for visualize results (TTL-based)
_visualize_cache = {}  # key -> {"data": ..., "ts": time.time()}
_VISUALIZE_CACHE_TTL = 300  # 5 minutes

@app.get("/visualize", dependencies=[Depends(check_heavy_rate_limit)])
async def visualize_documents(
    limit: int = Query(default=2000, ge=10, le=10000, description="Max documents to visualize"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    sample: Optional[str] = Query(default=None, description="Sampling: 'uniform' for even time distribution"),
    nocache: bool = Query(default=False, description="Skip cache")
):
    """
    Get 2D coordinates for document visualization using PCA dimensionality reduction.
    Returns documents with x,y coordinates for scatter plot visualization.
    Results cached for 5 minutes.
    """
    try:
        # Check cache
        cache_key = f"{limit}:{tag}:{sample}"
        if not nocache and cache_key in _visualize_cache:
            cached = _visualize_cache[cache_key]
            if time.time() - cached["ts"] < _VISUALIZE_CACHE_TTL:
                logger.info(f"Visualize cache hit: {cache_key}")
                return cached["data"]

        # Build query based on filters
        async with db_pool.acquire() as conn:
            if sample == "uniform":
                # Uniform sampling across time: use ntile to get evenly spaced docs
                if tag:
                    rows = await conn.fetch("""
                        WITH ranked AS (
                            SELECT d.guid, d.content AS preview, d.tags, d.created_at, e.embedding,
                                   ROW_NUMBER() OVER (ORDER BY d.created_at) as rn,
                                   COUNT(*) OVER () as total
                            FROM documents d
                            JOIN doc_embeddings e ON d.guid = e.guid
                            WHERE $1::text = ANY(d.tags)
                        )
                        SELECT guid, preview, tags, created_at, embedding
                        FROM ranked
                        WHERE rn % GREATEST(1, total / $2) = 0 OR total <= $2
                        ORDER BY created_at
                        LIMIT $2
                    """, tag, limit)
                else:
                    rows = await conn.fetch("""
                        WITH ranked AS (
                            SELECT d.guid, d.content AS preview, d.tags, d.created_at, e.embedding,
                                   ROW_NUMBER() OVER (ORDER BY d.created_at) as rn,
                                   COUNT(*) OVER () as total
                            FROM documents d
                            JOIN doc_embeddings e ON d.guid = e.guid
                        )
                        SELECT guid, preview, tags, created_at, embedding
                        FROM ranked
                        WHERE rn % GREATEST(1, total / $1) = 0 OR total <= $1
                        ORDER BY created_at
                        LIMIT $1
                    """, limit)
            elif tag:
                rows = await conn.fetch("""
                    SELECT d.guid, d.content AS preview, d.tags, d.created_at, e.embedding
                    FROM documents d
                    JOIN doc_embeddings e ON d.guid = e.guid
                    WHERE $1::text = ANY(d.tags)
                    ORDER BY d.created_at DESC
                    LIMIT $2
                """, tag, limit)
            else:
                rows = await conn.fetch("""
                    SELECT d.guid, d.content AS preview, d.tags, d.created_at, e.embedding
                    FROM documents d
                    JOIN doc_embeddings e ON d.guid = e.guid
                    ORDER BY d.created_at DESC
                    LIMIT $1
                """, limit)

        if len(rows) < 5:
            return {
                "error": "Not enough documents with embeddings (minimum 5 required)",
                "count": len(rows),
                "documents": []
            }

        # Extract embeddings and metadata
        import numpy as np
        embeddings = []
        documents = []

        for row in rows:
            emb = row['embedding']
            if emb is not None:
                # Handle string representation of array
                if isinstance(emb, str):
                    emb = np.array([float(x) for x in emb.strip('[]').split(',') if x.strip()])
                elif isinstance(emb, (list, tuple)):
                    emb = np.array(emb)
                embeddings.append(emb)

                # Extract document type from tags
                doc_type = "unknown"
                project = None
                for t in (row['tags'] or []):
                    if t.startswith('type:'):
                        doc_type = t.split(':')[1]
                    elif t.startswith('guid:'):
                        project = t.split(':')[1]

                documents.append({
                    "guid": row['guid'],
                    "type": doc_type,
                    "project": project,
                    "preview": row['preview'] or "",
                    "tags": list(row['tags']) if row['tags'] else [],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })

        if len(embeddings) < 5:
            return {
                "error": "Not enough valid embeddings",
                "count": len(embeddings),
                "documents": []
            }

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # PCA dimensionality reduction (fast, <1s even for 10K docs)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings_array)

        clusters = [-1] * len(documents)

        # Normalize coordinates to 0-1 range
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1

        # Build response
        result_docs = []
        for i, doc in enumerate(documents):
            doc["x"] = float((coords[i, 0] - x_min) / x_range)
            doc["y"] = float((coords[i, 1] - y_min) / y_range)
            doc["cluster"] = int(clusters[i])
            result_docs.append(doc)

        result = {
            "count": len(result_docs),
            "documents": result_docs
        }

        # Cache the result
        _visualize_cache[cache_key] = {"data": result, "ts": time.time()}
        logger.info(f"Visualize computed and cached: {cache_key} ({len(result_docs)} docs)")

        return result

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Visualization failed"
        )



# ============== VERSION CHAINS ENDPOINT ==============

@app.get("/versions/{guid}", dependencies=[Depends(check_search_rate_limit)])
async def get_version_chain(
    guid: str,
    threshold: float = Query(default=0.85, ge=0.5, le=0.99, description="Similarity threshold"),
    limit: int = Query(default=20, ge=1, le=100, description="Max versions to return"),
    same_project: bool = Query(default=True, description="Only docs from same project (guid: tag)")
):
    """Find chronological version chain of a document via embedding similarity.

    Returns documents similar to the given one, sorted by creation date,
    forming a 'version history' of how the document evolved over time.
    Includes diff info: content length delta, tags added/removed.
    """
    try:
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GUID format: {guid}"
            )

        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with db_pool.acquire() as conn:
            # Get target document info (use subquery for embedding comparison)
            target = await conn.fetchrow("""
                SELECT d.guid, d.tags, d.created_at, LENGTH(d.content) as content_length
                FROM documents d
                JOIN doc_embeddings e ON d.guid = e.guid
                WHERE d.guid = $1
            """, guid)

            if not target:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {guid} not found or not yet indexed"
                )

            target_tags = list(target['tags']) if target['tags'] else []

            # Find project tag for filtering
            project_tag = None
            if same_project:
                for t in target_tags:
                    if t.startswith('guid:'):
                        project_tag = t
                        break

            # Find similar documents — embedding compared via subquery (stays in PostgreSQL)
            if project_tag:
                rows = await conn.fetch("""
                    SELECT d.guid, d.tags, d.created_at, d.updated_at,
                           LEFT(d.content, 150) as preview,
                           LENGTH(d.content) as content_length,
                           1 - (e.embedding <=> (SELECT embedding FROM doc_embeddings WHERE guid = $1)) AS similarity
                    FROM documents d
                    JOIN doc_embeddings e ON d.guid = e.guid
                    WHERE d.guid != $1
                      AND 1 - (e.embedding <=> (SELECT embedding FROM doc_embeddings WHERE guid = $1)) >= $2
                      AND $4::text = ANY(d.tags)
                    ORDER BY d.created_at ASC
                    LIMIT $3
                """, guid, threshold, limit, project_tag)
            else:
                rows = await conn.fetch("""
                    SELECT d.guid, d.tags, d.created_at, d.updated_at,
                           LEFT(d.content, 150) as preview,
                           LENGTH(d.content) as content_length,
                           1 - (e.embedding <=> (SELECT embedding FROM doc_embeddings WHERE guid = $1)) AS similarity
                    FROM documents d
                    JOIN doc_embeddings e ON d.guid = e.guid
                    WHERE d.guid != $1
                      AND 1 - (e.embedding <=> (SELECT embedding FROM doc_embeddings WHERE guid = $1)) >= $2
                    ORDER BY d.created_at ASC
                    LIMIT $3
                """, guid, threshold, limit)

        # Build version chain with diff info
        target_tag_set = set(target_tags)
        target_len = target['content_length'] or 0

        chain = []
        for row in rows:
            row_tags = list(row['tags']) if row['tags'] else []
            row_tag_set = set(row_tags)
            content_len = row['content_length'] or 0

            chain.append({
                "guid": row['guid'],
                "similarity": round(float(row['similarity']), 3),
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
                "preview": row['preview'] or "",
                "content_length": content_len,
                "length_delta": content_len - target_len,
                "tags": row_tags,
                "tags_added": sorted(row_tag_set - target_tag_set),
                "tags_removed": sorted(target_tag_set - row_tag_set)
            })

        return {
            "source": {
                "guid": guid,
                "created_at": target['created_at'].isoformat() if target['created_at'] else None,
                "content_length": target_len,
                "tags": target_tags
            },
            "threshold": threshold,
            "same_project": same_project,
            "project_tag": project_tag,
            "count": len(chain),
            "versions": chain
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Version chain failed for {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get version chain"
        )



# ============== TAG SCHEMA ENDPOINT ==============

@app.get("/schema")
async def get_schema():
    """Get the tag schema configuration.

    Returns configured tag prefixes, allowed values, defaults, and auto-inference rules.
    Used by MCP server, CLI, and other clients to understand the tag taxonomy.
    """
    return tag_schema.to_dict()

# ============== CATCH-ALL ROUTE (must be last) ==============

@app.get("/{guid}", response_model=DocumentResponse)
async def get_document(guid: str):
    """Get a document by its GUID (must be last route to avoid shadowing specific routes)"""
    try:
        # Validate GUID format
        if not validate_document_guid(guid):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Not found"
            )

        # Get document from database
        document = await document_crud.get_document_by_guid(guid)

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with GUID {guid} not found"
            )

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {guid}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document"
        )

if __name__ == "__main__":
    import uvicorn
    from .config import get_api_host, get_api_port
    
    uvicorn.run(
        app,
        host=get_api_host(),
        port=get_api_port(),
        log_level="info"
    )
