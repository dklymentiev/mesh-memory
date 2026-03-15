"""
API tests for Mesh document management service.

Covers:
  - GET  /health          health check
  - GET  /help            API help / documentation endpoint
  - PUT  /                create document
  - GET  /?all=true       list documents
  - GET  /{guid}          get document by GUID
  - POST /search          semantic + keyword search
  - GET  /tags            tag listing
  - GET  /stats           statistics
  - DELETE /{guid}        delete document
  - PATCH /{guid}         update document
  - PUT  /doc/{guid}      upsert document with explicit GUID

All tests run against in-memory fakes; no PostgreSQL or GPU required.
"""

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONTENT = "This is a valid test document with enough characters."
SHORT_CONTENT = "too short"


async def _create_doc(client, content: str = VALID_CONTENT,
                      tags: list | None = None) -> dict:
    """Helper: PUT / and return the response."""
    payload: dict = {"content": content}
    if tags is not None:
        payload["tags"] = tags
    return await client.put("/", json=payload)


# ---------------------------------------------------------------------------
# Health check  GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_body_has_status_key(self, client):
        body = (await client.get("/health")).json()
        assert "status" in body

    @pytest.mark.asyncio
    async def test_health_reports_healthy(self, client):
        body = (await client.get("/health")).json()
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_includes_service_name(self, client):
        body = (await client.get("/health")).json()
        assert body.get("service") == "mesh-api"


# ---------------------------------------------------------------------------
# Help endpoint  GET /help
# ---------------------------------------------------------------------------

class TestHelp:
    @pytest.mark.asyncio
    async def test_help_returns_200(self, client):
        resp = await client.get("/help")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_help_lists_endpoints(self, client):
        body = (await client.get("/help")).json()
        assert "endpoints" in body

    @pytest.mark.asyncio
    async def test_help_includes_search_endpoint(self, client):
        body = (await client.get("/help")).json()
        endpoints = body.get("endpoints", {})
        assert any("search" in k.lower() for k in endpoints)


# ---------------------------------------------------------------------------
# Create document  PUT /
# ---------------------------------------------------------------------------

class TestCreateDocument:
    @pytest.mark.asyncio
    async def test_create_returns_201(self, client):
        resp = await _create_doc(client)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_create_returns_guid(self, client):
        body = (await _create_doc(client)).json()
        assert "guid" in body
        # GUID format: doc_{16 hex chars}  ->  20 chars total
        assert body["guid"].startswith("doc_")
        assert len(body["guid"]) == 20

    @pytest.mark.asyncio
    async def test_create_returns_content(self, client):
        body = (await _create_doc(client)).json()
        assert body["content"] == VALID_CONTENT

    @pytest.mark.asyncio
    async def test_create_with_tags(self, client):
        resp = await _create_doc(client, tags=["type:test", "project:mesh"])
        body = resp.json()
        assert resp.status_code == 201
        assert "type:test" in body["tags"]
        assert "project:mesh" in body["tags"]

    @pytest.mark.asyncio
    async def test_create_deduplicates_tags(self, client):
        resp = await _create_doc(client, tags=["foo", "foo", "bar"])
        body = resp.json()
        assert body["tags"].count("foo") == 1

    @pytest.mark.asyncio
    async def test_create_short_content_returns_400(self, client):
        resp = await _create_doc(client, content=SHORT_CONTENT)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_empty_content_returns_error(self, client):
        resp = await client.put("/", json={"content": ""})
        # Pydantic min_length=1 -> 422, or our 400
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_create_missing_content_returns_422(self, client):
        resp = await client.put("/", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_normalises_tags_to_lowercase(self, client):
        resp = await _create_doc(client, tags=["TYPE:WORKLOG", "Project:Mesh"])
        body = resp.json()
        assert "type:worklog" in body["tags"]
        assert "project:mesh" in body["tags"]

    @pytest.mark.asyncio
    async def test_create_returns_directory(self, client):
        body = (await _create_doc(client)).json()
        assert "directory" in body


# ---------------------------------------------------------------------------
# List documents  GET /?all=true
#
# The root GET / with no params returns a help/README dict.
# Pass ?all=true to get a document listing.
# ---------------------------------------------------------------------------

class TestListDocuments:
    @pytest.mark.asyncio
    async def test_list_with_all_flag_returns_200(self, client):
        resp = await client.get("/?all=true")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_returns_created_doc(self, client):
        await _create_doc(client, content="Unique document for listing test here.")
        resp = await client.get("/?all=true")
        body = resp.json()
        # Response may be a list or dict with "documents" key
        docs = body if isinstance(body, list) else body.get("documents", [])
        assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_list_limit_param(self, client):
        for i in range(5):
            await _create_doc(client, content=f"Document number {i} for limit test enough.")
        resp = await client.get("/?all=true&limit=2")
        assert resp.status_code == 200
        body = resp.json()
        docs = body if isinstance(body, list) else body.get("documents", [])
        assert len(docs) <= 2

    @pytest.mark.asyncio
    async def test_list_no_params_returns_help_dict(self, client):
        """GET / with no params returns the README/help response, not a list."""
        resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        # Should be a dict containing service info
        assert isinstance(body, dict)
        assert "service" in body

    @pytest.mark.asyncio
    async def test_list_invalid_limit_returns_422(self, client):
        resp = await client.get("/?limit=0&all=true")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get document by GUID  GET /{guid}
# ---------------------------------------------------------------------------

class TestGetDocument:
    @pytest.mark.asyncio
    async def test_get_existing_document(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        resp = await client.get(f"/{guid}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_returns_correct_guid(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        body = (await client.get(f"/{guid}")).json()
        assert body["guid"] == guid

    @pytest.mark.asyncio
    async def test_get_returns_correct_content(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        body = (await client.get(f"/{guid}")).json()
        assert body["content"] == VALID_CONTENT

    @pytest.mark.asyncio
    async def test_get_nonexistent_guid_returns_404(self, client):
        resp = await client.get("/doc_00000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_invalid_guid_format_returns_4xx(self, client):
        """An invalid GUID format returns 400 (bad request) from the endpoint."""
        resp = await client.get("/not-a-valid-guid")
        # The endpoint validates and returns 400 for invalid format
        assert resp.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# Search  POST /search
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_200(self, client):
        resp = await client.post("/search", json={"query": "test document"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_search_response_structure(self, client):
        body = (await client.post("/search", json={"query": "hello"})).json()
        assert "results" in body
        assert "total_count" in body
        assert "query" in body

    @pytest.mark.asyncio
    async def test_search_echoes_query(self, client):
        query = "unique search query string"
        body = (await client.post("/search", json={"query": query})).json()
        assert body["query"] == query

    @pytest.mark.asyncio
    async def test_search_keyword_fallback(self, client):
        """Keyword fallback should surface exact term matches."""
        content = "The XYZZY protocol is used for special network routing."
        await _create_doc(client, content=content)
        body = (await client.post("/search", json={"query": "XYZZY"})).json()
        contents = [r["content"] for r in body["results"]]
        assert any("XYZZY" in c for c in contents)

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_error(self, client):
        resp = await client.post("/search", json={"query": ""})
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_search_missing_query_returns_422(self, client):
        resp = await client.post("/search", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_with_limit(self, client):
        for i in range(5):
            await _create_doc(
                client,
                content=f"Test document {i} about machine learning and AI research."
            )
        body = (await client.post(
            "/search", json={"query": "machine learning", "limit": 2}
        )).json()
        assert len(body["results"]) <= 2

    @pytest.mark.asyncio
    async def test_search_result_has_similarity_score(self, client):
        await _create_doc(client, content="Document about neural networks and deep learning.")
        body = (await client.post("/search", json={"query": "neural networks"})).json()
        for result in body["results"]:
            assert "similarity_score" in result
            score = result["similarity_score"]
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_search_result_has_required_fields(self, client):
        await _create_doc(client, content="Document with all required response fields test.")
        body = (await client.post("/search", json={"query": "required fields"})).json()
        for result in body["results"]:
            assert "guid" in result
            assert "content" in result
            assert "tags" in result
            assert "created_at" in result
            assert "similarity_score" in result

    @pytest.mark.asyncio
    async def test_search_with_tag_filter(self, client):
        await _create_doc(
            client,
            content="Python programming language tutorial for beginners.",
            tags=["lang:python"]
        )
        await _create_doc(
            client,
            content="JavaScript frontend development guide and tutorial.",
            tags=["lang:javascript"]
        )
        body = (await client.post(
            "/search", json={"query": "programming tutorial", "tags": ["lang:python"]}
        )).json()
        for result in body["results"]:
            assert "lang:python" in result["tags"]

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_docs(self, client):
        body = (await client.post("/search", json={"query": "anything"})).json()
        assert body["total_count"] == 0
        assert body["results"] == []


# ---------------------------------------------------------------------------
# Tags endpoint  GET /tags
# Response shape: {"tags": [{"tag": str, "count": int}, ...], "total": int, ...}
# ---------------------------------------------------------------------------

class TestTags:
    @pytest.mark.asyncio
    async def test_tags_returns_200(self, client):
        resp = await client.get("/tags")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_tags_response_has_tags_key(self, client):
        body = (await client.get("/tags")).json()
        assert "tags" in body

    @pytest.mark.asyncio
    async def test_tags_lists_created_tags(self, client):
        await _create_doc(client, tags=["type:note", "project:alpha"])
        body = (await client.get("/tags")).json()
        # tags is a list of {"tag": ..., "count": ...}
        tag_names = [t["tag"] for t in body["tags"]]
        assert "type:note" in tag_names
        assert "project:alpha" in tag_names

    @pytest.mark.asyncio
    async def test_tags_count_is_accurate(self, client):
        await _create_doc(client, tags=["type:note"])
        await _create_doc(
            client,
            content="Second document for tag count test accuracy here.",
            tags=["type:note"]
        )
        body = (await client.get("/tags")).json()
        note_entry = next((t for t in body["tags"] if t["tag"] == "type:note"), None)
        assert note_entry is not None
        assert note_entry["count"] == 2

    @pytest.mark.asyncio
    async def test_tags_prefix_filter(self, client):
        await _create_doc(client, tags=["type:worklog", "project:beta", "status:active"])
        resp = await client.get("/tags?prefix=type:")
        assert resp.status_code == 200
        body = resp.json()
        tag_names = [t["tag"] for t in body["tags"]]
        # All returned tags must start with "type:"
        for name in tag_names:
            assert name.startswith("type:")

    @pytest.mark.asyncio
    async def test_tags_response_has_total_key(self, client):
        body = (await client.get("/tags")).json()
        assert "total" in body


# ---------------------------------------------------------------------------
# Stats endpoint  GET /stats
# ---------------------------------------------------------------------------

class TestStats:
    @pytest.mark.asyncio
    async def test_stats_returns_200(self, client):
        resp = await client.get("/stats")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stats_has_documents_key(self, client):
        body = (await client.get("/stats")).json()
        assert "documents" in body

    @pytest.mark.asyncio
    async def test_stats_document_count_increases(self, client):
        before = (await client.get("/stats")).json().get("documents", 0)
        await _create_doc(client)
        after = (await client.get("/stats")).json().get("documents", 0)
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_stats_has_indexed_key(self, client):
        body = (await client.get("/stats")).json()
        assert "indexed" in body

    @pytest.mark.asyncio
    async def test_stats_empty_store_shows_zero(self, client):
        body = (await client.get("/stats")).json()
        assert body["documents"] == 0

    @pytest.mark.asyncio
    async def test_stats_has_workspaces_key(self, client):
        body = (await client.get("/stats")).json()
        assert "workspaces" in body
        assert isinstance(body["workspaces"], list)

    @pytest.mark.asyncio
    async def test_stats_workspaces_have_required_fields(self, client):
        await _create_doc(client)
        body = (await client.get("/stats")).json()
        for ws in body["workspaces"]:
            assert "workspace" in ws
            assert "documents" in ws
            assert "last_activity" in ws


# ---------------------------------------------------------------------------
# Upsert with explicit GUID  PUT /doc/{guid}
# ---------------------------------------------------------------------------

class TestUpsertDocument:
    @pytest.mark.asyncio
    async def test_upsert_new_document(self, client):
        resp = await client.put(
            "/doc/doc_aabbccdd",
            json={"content": "Upserted document with explicit GUID value."}
        )
        assert resp.status_code in (200, 201)
        assert resp.json()["guid"] == "doc_aabbccdd"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_document(self, client):
        guid = "doc_11223344"
        await client.put(
            f"/doc/{guid}",
            json={"content": "Original content for upsert update test here."}
        )
        resp = await client.put(
            f"/doc/{guid}",
            json={"content": "Updated content for the same document in store."}
        )
        assert resp.status_code in (200, 201)
        assert resp.json()["content"] == "Updated content for the same document in store."


# ---------------------------------------------------------------------------
# Delete document  DELETE /{guid}
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_existing_document(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        resp = await client.delete(f"/{guid}")
        assert resp.status_code in (200, 204)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        resp = await client.delete("/doc_deadbeef")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removed_from_store(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        del_resp = await client.delete(f"/{guid}")
        assert del_resp.status_code in (200, 204)
        get_resp = await client.get(f"/{guid}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_returns_confirmation(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        body = (await client.delete(f"/{guid}")).json()
        assert body.get("deleted") is True
        assert body.get("guid") == guid


# ---------------------------------------------------------------------------
# Patch / update document  PATCH /{guid}
# ---------------------------------------------------------------------------

class TestPatchDocument:
    @pytest.mark.asyncio
    async def test_patch_updates_tags(self, client):
        guid = (await _create_doc(client, tags=["old:tag"])).json()["guid"]
        resp = await client.patch(f"/{guid}", json={"tags": ["new:tag", "updated:yes"]})
        assert resp.status_code == 200
        body = resp.json()
        assert "new:tag" in body["tags"]

    @pytest.mark.asyncio
    async def test_patch_nonexistent_returns_404(self, client):
        resp = await client.patch("/doc_00000000", json={"tags": ["x"]})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Model / utility unit tests (no HTTP, no fixtures needed)
# ---------------------------------------------------------------------------

class TestUtils:
    def test_generate_guid_format(self):
        from mesh.utils import generate_document_guid, validate_document_guid
        guid = generate_document_guid()
        assert guid.startswith("doc_")
        assert len(guid) == 20
        assert validate_document_guid(guid)

    def test_validate_guid_rejects_bad_format(self):
        from mesh.utils import validate_document_guid
        assert not validate_document_guid("bad-guid")
        assert not validate_document_guid("doc_ZZZZZZZZ")  # non-hex
        assert not validate_document_guid("doc_123")        # too short
        assert not validate_document_guid("")

    def test_normalize_tags_lowercases(self):
        from mesh.utils import normalize_tags
        assert normalize_tags(["FOO", "Bar", "BAZ"]) == ["foo", "bar", "baz"]

    def test_normalize_tags_deduplicates(self):
        from mesh.utils import normalize_tags
        assert normalize_tags(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_normalize_tags_strips_whitespace(self):
        from mesh.utils import normalize_tags
        result = normalize_tags(["  hello  ", "  world"])
        assert "hello" in result
        assert "world" in result

    def test_normalize_tags_removes_empty(self):
        from mesh.utils import normalize_tags
        assert normalize_tags(["", "  ", "valid"]) == ["valid"]

    def test_normalize_tags_none_input(self):
        from mesh.utils import normalize_tags
        assert normalize_tags(None) == []

    def test_normalize_tags_strips_duplicates_after_lowercasing(self):
        from mesh.utils import normalize_tags
        # "FOO" and "foo" should become one entry after normalization
        result = normalize_tags(["FOO", "foo", "bar"])
        assert result.count("foo") == 1

    def test_compute_content_hash(self):
        from mesh.crud import compute_content_hash
        h1 = compute_content_hash("hello world")
        h2 = compute_content_hash("hello world")
        h3 = compute_content_hash("different content")
        assert h1 == h2          # deterministic
        assert h1 != h3          # different content -> different hash
        assert len(h1) == 32     # MD5 hex is 32 chars

    def test_validate_guid_accepts_valid_format(self):
        from mesh.utils import validate_document_guid
        assert validate_document_guid("doc_12345678")
        assert validate_document_guid("doc_abcdef01")
        assert validate_document_guid("doc_00000000")



# ---------------------------------------------------------------------------
# Embed endpoint  POST /embed
# ---------------------------------------------------------------------------

class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_returns_200(self, client):
        resp = await client.post("/embed", json={"text": "hello world"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_embed_returns_embedding_vector(self, client):
        body = (await client.post("/embed", json={"text": "test text"})).json()
        assert "embedding" in body
        assert isinstance(body["embedding"], list)
        assert len(body["embedding"]) == 768

    @pytest.mark.asyncio
    async def test_embed_returns_dimensions(self, client):
        body = (await client.post("/embed", json={"text": "dimensions"})).json()
        assert body["dimensions"] == 768

    @pytest.mark.asyncio
    async def test_embed_returns_model_name(self, client):
        body = (await client.post("/embed", json={"text": "model"})).json()
        assert "model" in body

    @pytest.mark.asyncio
    async def test_embed_empty_text_returns_400(self, client):
        resp = await client.post("/embed", json={"text": ""})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_embed_whitespace_text_returns_400(self, client):
        resp = await client.post("/embed", json={"text": "   "})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_embed_missing_text_returns_400(self, client):
        resp = await client.post("/embed", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Embed batch endpoint  POST /embed/batch
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_200(self, client):
        resp = await client.post("/embed/batch", json={"texts": ["one", "two"]})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_returns_embeddings(self, client):
        body = (await client.post(
            "/embed/batch", json={"texts": ["a", "b", "c"]}
        )).json()
        assert "embeddings" in body
        assert body["count"] == 3
        assert body["dimensions"] == 768

    @pytest.mark.asyncio
    async def test_batch_empty_texts_returns_400(self, client):
        resp = await client.post("/embed/batch", json={"texts": []})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_missing_texts_returns_400(self, client):
        resp = await client.post("/embed/batch", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_over_100_returns_400(self, client):
        resp = await client.post(
            "/embed/batch", json={"texts": ["t"] * 101}
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_filters_whitespace(self, client):
        body = (await client.post(
            "/embed/batch", json={"texts": ["valid", "   ", "also valid"]}
        )).json()
        assert body["count"] == 2


# ---------------------------------------------------------------------------
# Bulk create  PUT /bulk
# ---------------------------------------------------------------------------

class TestBulkCreate:
    @pytest.mark.asyncio
    async def test_bulk_returns_200(self, client):
        docs = [
            {"content": "First bulk document with enough characters here."},
            {"content": "Second bulk document with enough characters here."},
        ]
        resp = await client.put("/bulk", json=docs)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_bulk_creates_documents(self, client):
        docs = [
            {"content": "Bulk doc one is long enough for validation check."},
            {"content": "Bulk doc two is long enough for validation check."},
        ]
        body = (await client.put("/bulk", json=docs)).json()
        assert body["created"] == 2
        assert body["skipped"] == 0

    @pytest.mark.asyncio
    async def test_bulk_skips_short_content(self, client):
        docs = [
            {"content": "Valid document with enough characters for bulk."},
            {"content": "short"},
        ]
        body = (await client.put("/bulk", json=docs)).json()
        assert body["created"] == 1
        assert body["skipped"] == 1

    @pytest.mark.asyncio
    async def test_bulk_empty_list(self, client):
        body = (await client.put("/bulk", json=[])).json()
        assert body["created"] == 0


# ---------------------------------------------------------------------------
# By-hash endpoints  GET /by-hash/{hash}  PATCH /by-hash
# ---------------------------------------------------------------------------

class TestByHash:
    @pytest.mark.asyncio
    async def test_get_by_hash_found(self, client):
        create_resp = await _create_doc(client)
        content_hash = create_resp.json()["content_hash"]
        resp = await client.get(f"/by-hash/{content_hash}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert "guid" in body

    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self, client):
        resp = await client.get("/by-hash/0000000000000000deadbeefdeadbeef")
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is False

    @pytest.mark.asyncio
    async def test_patch_by_hash_not_found(self, client):
        resp = await client.patch("/by-hash", json={
            "content_hash": "0000000000000000deadbeefdeadbeef"
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["updated"] is False

    @pytest.mark.asyncio
    async def test_patch_by_hash_no_fields(self, client):
        create_resp = await _create_doc(client)
        content_hash = create_resp.json()["content_hash"]
        resp = await client.patch("/by-hash", json={
            "content_hash": content_hash
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["updated"] is False


# ---------------------------------------------------------------------------
# Metadata endpoints  /meta
# ---------------------------------------------------------------------------

class TestMetadata:
    @pytest.mark.asyncio
    async def test_put_metadata(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        resp = await client.put(f"/meta/{guid}", json={
            "doc_type": "test",
            "metadata": {"key": "value"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["guid"] == guid
        assert body["doc_type"] == "test"

    @pytest.mark.asyncio
    async def test_get_metadata(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        await client.put(f"/meta/{guid}", json={
            "doc_type": "note",
            "metadata": {"foo": "bar"},
        })
        resp = await client.get(f"/meta/{guid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] == "note"

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self, client):
        resp = await client.get("/meta/doc_00000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_metadata(self, client):
        guid = (await _create_doc(client)).json()["guid"]
        await client.put(f"/meta/{guid}", json={
            "doc_type": "temp",
            "metadata": {},
        })
        resp = await client.delete(f"/meta/{guid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_metadata_not_found(self, client):
        resp = await client.delete("/meta/doc_00000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_metadata(self, client):
        resp = await client.get("/meta")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_put_metadata_invalid_guid(self, client):
        resp = await client.put("/meta/bad-guid", json={
            "doc_type": "test",
            "metadata": {},
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Activity endpoint  GET /activity
# ---------------------------------------------------------------------------

class TestActivity:
    @pytest.mark.asyncio
    async def test_activity_returns_200(self, client):
        resp = await client.get("/activity")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_activity_returns_list(self, client):
        body = (await client.get("/activity")).json()
        assert isinstance(body, list)

    @pytest.mark.asyncio
    async def test_activity_with_prefix(self, client):
        await _create_doc(client, tags=["type:worklog", "guid:proj1"])
        resp = await client.get("/activity?prefix=type:")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_activity_limit(self, client):
        resp = await client.get("/activity?limit=5")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Short documents endpoint  GET /short
# ---------------------------------------------------------------------------

class TestShort:
    @pytest.mark.asyncio
    async def test_short_returns_200(self, client):
        resp = await client.get("/short")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_short_returns_list(self, client):
        body = (await client.get("/short")).json()
        assert isinstance(body, list)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Validate that database schema columns match what CRUD operations expect."""

    def test_create_table_has_all_crud_columns(self):
        """Ensure CREATE TABLE in database.py defines all columns used by CRUD INSERT."""
        import mesh.database as database
        import inspect

        # Get the source of create_tables
        source = inspect.getsource(database.create_tables)

        # Find the documents CREATE TABLE block (use greedy match for nested parens)
        import re
        # Match from CREATE TABLE to the closing triple-quote of the SQL string
        match = re.search(
            r'CREATE TABLE IF NOT EXISTS documents\s*\((.+?)\)\s*\n\s*"""\)',
            source, re.DOTALL | re.IGNORECASE
        )
        assert match, "Could not find CREATE TABLE documents statement"
        schema_text = match.group(1).lower()

        # Columns that CRUD INSERT uses (from crud.py DocumentCRUD.create_document)
        required_columns = [
            "guid", "content", "content_hash", "filename",
            "source", "tags", "created_at", "updated_at"
        ]

        for col in required_columns:
            assert col in schema_text, (
                f"Column '{col}' is used in CRUD INSERT but missing from "
                f"CREATE TABLE documents in database.py"
            )

    def test_migration_adds_missing_columns(self):
        """Ensure ALTER TABLE migrations cover all columns not in original schema."""
        import mesh.database as database
        import inspect

        source = inspect.getsource(database.create_tables)

        # All columns that have ALTER TABLE ADD COLUMN
        migrated_columns = ["content_hash", "filename", "source", "updated_at"]
        for col in migrated_columns:
            assert f"add column if not exists {col}" in source.lower(), (
                f"Migration for column '{col}' missing from database.py"
            )

    def test_env_example_embedding_model_matches_schema(self):
        """Ensure .env.example model produces correct vector dimensions."""
        import os

        env_example = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".env.example"
        )
        if not os.path.exists(env_example):
            pytest.skip(".env.example not found")

        with open(env_example) as f:
            content = f.read()

        # Known model -> dimension mapping
        model_dims = {
            "all-MiniLM-L6-v2": 384,
            "intfloat/multilingual-e5-base": 768,
            "intfloat/multilingual-e5-large": 1024,
        }

        import mesh.database as database
        import inspect, re
        db_source = inspect.getsource(database.create_tables)
        dim_match = re.search(r"vector\((\d+)\)", db_source)
        assert dim_match, "Could not find vector dimension in database.py"
        schema_dim = int(dim_match.group(1))

        # Find EMBEDDING_MODEL line in .env.example
        for line in content.splitlines():
            if line.startswith("EMBEDDING_MODEL="):
                model = line.split("=", 1)[1].strip()
                if model in model_dims:
                    assert model_dims[model] == schema_dim, (
                        f".env.example sets EMBEDDING_MODEL={model} "
                        f"({model_dims[model]}-dim) but schema uses "
                        f"vector({schema_dim})"
                    )
                break

    def test_health_does_not_leak_errors(self, client):
        """Verify /health does not include raw exception strings."""
        # We can't easily trigger a real DB error in test, but verify the
        # unhealthy response structure doesn't use dynamic error strings.
        # The healthy path should return a known structure.
        import pytest
        # This is a compile-time check: read the source and verify no str(e)
        from mesh import main as _main
        import inspect, re
        source = inspect.getsource(_main.health_check)
        # Should NOT contain str(e) in the return dict
        assert 'str(e)' not in source, (
            "/health endpoint should not return str(e) - "
            "this leaks internal error details"
        )


class TestErrorResponses:
    """Verify error responses don't leak internal details."""

    @pytest.mark.asyncio
    async def test_health_unhealthy_generic_message(self, client):
        """Health endpoint returns generic error, not exception details."""
        # With our fake pool, /health returns healthy
        resp = await client.get("/health")
        body = resp.json()
        assert resp.status_code == 200
        # When healthy, no error field
        if "error" in body:
            # If present, must be generic
            assert "postgresql://" not in body["error"].lower()
            assert "password" not in body["error"].lower()

    @pytest.mark.asyncio
    async def test_help_shows_put_not_post(self, client):
        """Help endpoint should show PUT / not POST / for document creation."""
        resp = await client.get("/help")
        body = resp.json()
        endpoints = body.get("endpoints", {})
        assert "PUT /" in endpoints, "Help should show PUT / for document creation"
        assert "POST /" not in endpoints, "Help should not show POST / (route is PUT)"

    @pytest.mark.asyncio
    async def test_version_consistency(self, client):
        """Version in /help should match app version."""
        from mesh import main as _main
        help_resp = await client.get("/help")
        help_version = help_resp.json().get("version")
        assert help_version == _main.app.version, (
            f"/help version ({help_version}) != app version ({_main.app.version})"
        )
