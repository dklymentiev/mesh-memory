"""FastAPI router for categorizer endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from . import batch_scan, bootstrap, classify, get_store, init
from .config import is_categorizer_enabled, is_llm_available
from .models import BatchRequest, BootstrapRequest, TaxonomyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/categorizer", tags=["categorizer"])


def _get_pool():
    """Get the database pool from the store."""
    store = get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Categorizer not initialized")
    return store.pool


@router.get("/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy():
    """Get current taxonomy with stats and pending proposals."""
    store = get_store()
    if store is None or store.taxonomy is None:
        return TaxonomyResponse(
            taxonomy={"version": 1, "categories": []},
            stats={"status": "no taxonomy", "hint": "POST /categorizer/bootstrap to generate"},
        )

    pool = _get_pool()

    # Gather stats
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT count(*) FROM documents")
        categorized = await conn.fetchval(
            """
            SELECT count(*) FROM documents
            WHERE EXISTS (SELECT 1 FROM unnest(tags) AS t WHERE t LIKE 'ai-category:%')
            """
        )
        uncategorized_count = await conn.fetchval(
            """
            SELECT count(*) FROM documents
            WHERE EXISTS (SELECT 1 FROM unnest(tags) AS t WHERE t = 'ai-category:uncategorized')
            """
        )
        centroids_count = await conn.fetchval("SELECT count(*) FROM category_centroids")

        # Pending proposals
        proposals_rows = await conn.fetch(
            """
            SELECT guid, content, created_at FROM documents
            WHERE tags @> '{type:taxonomy-proposal}'::text[]
            ORDER BY created_at DESC LIMIT 20
            """
        )

    import json

    proposals = []
    for row in proposals_rows:
        try:
            proposals.append(
                {"guid": row["guid"], "created_at": str(row["created_at"]),
                 **json.loads(row["content"])}
            )
        except (json.JSONDecodeError, TypeError):
            proposals.append({"guid": row["guid"], "raw": row["content"][:200]})

    return TaxonomyResponse(
        taxonomy=store.taxonomy,
        stats={
            "total_documents": total,
            "categorized": categorized,
            "uncategorized": uncategorized_count,
            "centroids": centroids_count,
            "categories": len(store.taxonomy.categories),
            "llm_available": is_llm_available(),
        },
        proposals=proposals,
    )


@router.post("/bootstrap")
async def run_bootstrap(request: BootstrapRequest = BootstrapRequest()):
    """Generate taxonomy from existing documents."""
    pool = _get_pool()

    if request.mode == "designed" and not is_llm_available():
        raise HTTPException(
            status_code=400,
            detail="LLM required for 'designed' mode. Set LLM_API_URL, LLM_API_KEY, LLM_MODEL.",
        )

    if request.use_llm and not is_llm_available():
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. Set LLM_API_URL, LLM_API_KEY, LLM_MODEL.",
        )

    taxonomy = await bootstrap(
        pool,
        use_llm=request.use_llm,
        sample_size=request.sample_size,
        min_cluster_size=request.min_cluster_size,
        mode=request.mode,
    )

    return {
        "status": "ok",
        "categories": len(taxonomy.categories),
        "subcategories": sum(len(c.subcategories) for c in taxonomy.categories),
        "taxonomy": taxonomy.model_dump(),
    }


@router.post("/classify/{guid}")
async def classify_single(guid: str):
    """Classify a single document by GUID."""
    pool = _get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT e.embedding, d.content
            FROM doc_embeddings e
            JOIN documents d ON d.guid = e.guid
            WHERE e.guid = $1
            """,
            guid,
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"Document {guid} not found or has no embedding")

    import numpy as np

    raw = row["embedding"]
    if isinstance(raw, str):
        embedding = np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32).tolist()
    else:
        embedding = list(raw)

    result = await classify(guid, row["content"], embedding, pool)
    if result is None:
        raise HTTPException(status_code=503, detail="No taxonomy loaded. Run bootstrap first.")

    return {"guid": guid, "classification": result.model_dump()}


@router.post("/batch")
async def run_batch(request: BatchRequest = BatchRequest()):
    """Batch classify untagged documents."""
    pool = _get_pool()
    stats = await batch_scan(pool, limit=request.limit, force=request.force)
    return {"status": "ok", "stats": stats}
