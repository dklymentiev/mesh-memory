"""Taxonomy management: load/save from Mesh docs + centroids table."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import asyncpg
import numpy as np

from .models import Category, Subcategory, Taxonomy

logger = logging.getLogger(__name__)

# Tag used to find the taxonomy document in Mesh
TAXONOMY_TAG = "type:taxonomy"
TAXONOMY_TOPIC = "topic:ai-categorizer"


class TaxonomyStore:
    """Manages taxonomy persistence and centroid cache."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        # In-memory cache: category_id -> (centroid_vector, parent_id, name)
        self._centroids: Dict[str, Tuple[np.ndarray, Optional[str], str]] = {}
        self._taxonomy: Optional[Taxonomy] = None

    @property
    def taxonomy(self) -> Optional[Taxonomy]:
        return self._taxonomy

    @property
    def centroids_loaded(self) -> bool:
        return len(self._centroids) > 0

    async def load(self) -> Optional[Taxonomy]:
        """Load taxonomy from Mesh document and centroids from DB."""
        self._taxonomy = await self._load_taxonomy_doc()
        await self._load_centroids()
        if self._taxonomy:
            logger.info(
                f"Taxonomy loaded: {len(self._taxonomy.categories)} categories, "
                f"{len(self._centroids)} centroids cached"
            )
        return self._taxonomy

    async def save(self, taxonomy: Taxonomy) -> str:
        """Save taxonomy as Mesh document and update centroids table."""
        self._taxonomy = taxonomy
        guid = await self._save_taxonomy_doc(taxonomy)
        logger.info(f"Taxonomy saved to document {guid}")
        return guid

    async def save_centroids(
        self, centroids: List[Tuple[str, Optional[str], str, str, List[float], int]]
    ) -> int:
        """Bulk save centroids to category_centroids table.

        Each tuple: (category_id, parent_id, name, description, centroid_vector, doc_count)
        Returns number of rows upserted.
        """
        async with self.pool.acquire() as conn:
            # Clear old centroids and insert new
            await conn.execute("DELETE FROM category_centroids")
            count = 0
            for cat_id, parent_id, name, desc, centroid, doc_count in centroids:
                vec_str = "[" + ",".join(map(str, centroid)) + "]"
                await conn.execute(
                    """
                    INSERT INTO category_centroids
                        (category_id, parent_id, name, description, centroid, doc_count)
                    VALUES ($1, $2, $3, $4, $5::vector, $6)
                    ON CONFLICT (category_id) DO UPDATE SET
                        parent_id = EXCLUDED.parent_id,
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        centroid = EXCLUDED.centroid,
                        doc_count = EXCLUDED.doc_count,
                        updated_at = NOW()
                    """,
                    cat_id, parent_id, name, desc, vec_str, doc_count,
                )
                count += 1

        # Reload cache
        await self._load_centroids()
        logger.info(f"Saved {count} centroids to database")
        return count

    def get_category_centroids(self) -> Dict[str, Tuple[np.ndarray, str]]:
        """Return top-level category centroids: {id: (vector, name)}."""
        return {
            cid: (vec, name)
            for cid, (vec, parent, name) in self._centroids.items()
            if parent is None
        }

    def get_subcategory_centroids(self, parent_id: str) -> Dict[str, Tuple[np.ndarray, str]]:
        """Return subcategory centroids for a given parent category."""
        return {
            cid: (vec, name)
            for cid, (vec, parent, name) in self._centroids.items()
            if parent == parent_id
        }

    # -- Private methods --

    async def _load_taxonomy_doc(self) -> Optional[Taxonomy]:
        """Find and parse the taxonomy document from Mesh."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT content FROM documents
                WHERE tags @> $1::text[]
                ORDER BY updated_at DESC LIMIT 1
                """,
                [TAXONOMY_TAG, TAXONOMY_TOPIC],
            )
            if not row:
                return None
            try:
                data = json.loads(row["content"])
                return Taxonomy(**data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse taxonomy document: {e}")
                return None

    async def _save_taxonomy_doc(self, taxonomy: Taxonomy) -> str:
        """Save taxonomy as a Mesh document (upsert by tag)."""
        import hashlib
        from datetime import datetime, timezone

        content = json.dumps(taxonomy.model_dump(), indent=2, ensure_ascii=False)
        tags = [TAXONOMY_TAG, TAXONOMY_TOPIC, "status:active"]

        async with self.pool.acquire() as conn:
            # Check for existing taxonomy doc
            existing = await conn.fetchrow(
                """
                SELECT guid FROM documents
                WHERE tags @> $1::text[]
                ORDER BY updated_at DESC LIMIT 1
                """,
                [TAXONOMY_TAG, TAXONOMY_TOPIC],
            )

            now = datetime.now(timezone.utc)
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if existing:
                guid = existing["guid"]
                await conn.execute(
                    """
                    UPDATE documents
                    SET content = $1, content_hash = $2, tags = $3, updated_at = $4
                    WHERE guid = $5
                    """,
                    content, content_hash, tags, now, guid,
                )
            else:
                guid = "doc_" + hashlib.md5(
                    f"taxonomy-{now.isoformat()}".encode()
                ).hexdigest()[:8]
                await conn.execute(
                    """
                    INSERT INTO documents (guid, content, content_hash, source, tags, created_at, updated_at)
                    VALUES ($1, $2, $3, 'system', $4, $5, $5)
                    """,
                    guid, content, content_hash, tags, now,
                )
            return guid

    async def _load_centroids(self) -> None:
        """Load all centroids from DB into memory cache."""
        self._centroids.clear()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT category_id, parent_id, name, centroid FROM category_centroids"
            )
            for row in rows:
                raw = row["centroid"]
                if raw is None:
                    continue
                # pgvector returns string "[0.1,0.2,...]" via asyncpg
                if isinstance(raw, str):
                    vec = np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32)
                else:
                    vec = np.array(raw, dtype=np.float32)
                self._centroids[row["category_id"]] = (
                    vec,
                    row["parent_id"],
                    row["name"],
                )
