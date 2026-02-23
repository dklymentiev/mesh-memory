# Changelog

All notable changes to Mesh are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **AI Categorizer** (opt-in): automatic document classification using LLM
  - Embedding-based + LLM-based classifiers with configurable thresholds
  - Category taxonomy with subcategories
  - API endpoints: `/categorizer/taxonomy`, `/categorizer/bootstrap`, `/categorizer/batch`
  - 7 new environment variables (`CATEGORIZER_ENABLED`, `LLM_API_URL`, etc.)
  - Gated by `CATEGORIZER_ENABLED=false` (default off)
- **Built-in Web UI** served at `/ui/`
  - Search page (`index.html`): semantic search, card grid, document panel with markdown
  - Map page (`map.html`): Three.js galaxy/timeline visualization with particle clustering
  - Modular ES modules architecture (`js/map/*.js`, 15 modules, no build step)
  - Text search (TX) and semantic search (AI) with live timer and client-side cache
- `/browse` endpoint: all documents with short previews for client-side search
- `/keyword` endpoint: fast ILIKE search without embedding
- `/summarize/{guid}` endpoint: on-demand LLM document summarization with DB caching
- Server-side query embedding cache (1h TTL, 500 entries) to avoid repeated 20s CPU inference
- `api-demo` service in docker-compose.yml for public demo instance
- `scripts/generate_demo.py` and `scripts/enrich_demo_tags.py` dev utilities
- **Demo seed data**: `examples/demo-seed.jsonl` with 1000 pre-generated documents across 15 categories
  - 3 new categories: Design & UX, DevOps & SRE, Career & Management
  - Tags pre-enriched (category, topic, project, status) -- no separate enrichment step needed
  - Auto-seeds on first startup when `ENVIRONMENT=demo` and database is empty
- `scripts/seed_demo.py`: standalone JSONL loader for manual seeding
- `generate_demo.py` now supports `--output` (write JSONL) and `--seed` (deterministic output) flags

### Changed
- docker-compose.yml: domain defaults changed to `example.com` placeholders (set real domains in `.env`)
- `.dockerignore`: added `docker-compose*.yml`, `mcp_server.py`, `scripts/` to reduce build context

### Fixed
- `.gitignore`: added `*.bak` pattern to prevent accidental backup commits

---

## [1.2.1] - 2026-02-21

### Added
- CONTRIBUTING.md with development setup, test instructions, code style guidelines, and PR process
- `pyproject.toml` with project metadata, ruff/pytest/coverage config
- `.editorconfig` for consistent editor settings
- GitHub issue templates (bug report, feature request) and PR template
- `examples/` directory with `basic_usage.py` and `bulk_import.sh`

### Changed
- Reorganized project into `mesh/` Python package (moved from flat files in root)
- Gunicorn entry point: `mesh.main:app` (was `main:app`)
- CI uses `pgvector/pgvector:pg16` image (was `postgres:15`)
- Docker workflow uses GitHub Container Registry (ghcr.io) with native `GITHUB_TOKEN` auth
- `config.py` SIMILARITY_THRESHOLD default aligned to 0.1 (matches docker-compose)
- `.env.example` defaults aligned with docker-compose.yml
- README.md: badges, clone URL, comparison table, complete API endpoints table, examples section
- CONTRIBUTING.md: ruff replaces black, correct paths, updated project structure

### Fixed
- README clone URL and CI badges now point to GitHub (canonical host)
- `mesh/__init__.py` version corrected to 1.2.1 (was 1.0.0)
- `mcp` and `httpx` added to requirements.txt (were missing)
- Broken test imports: `import main` -> `from mesh import main` in test_api.py
- Internal domain removed from CHANGELOG example
- `secrets/` added to .dockerignore to prevent credential leaks in Docker builds

### Removed
- Dead code: `calculate_similarity` from embeddings.py, `check_connection` from database.py
- Dead code: unused config stubs and utility functions

---

## [1.2.0] - 2026-01-10

### Added
- **Metadata Layer**: Separate structured metadata storage per document
  - New `document_metadata` table with JSONB column for flexible schemas
  - `GET /meta/{guid}` - retrieve metadata for a document
  - `PUT /meta/{guid}` - create or replace metadata
  - `DELETE /meta/{guid}` - remove metadata
  - `GET /meta?type={doc_type}` - list metadata filtered by document type
- **Metadata layer**: Configurable origin restrictions for CORS via existing `CORS_ORIGINS` env var (frontend access for metadata endpoints)
- `MetadataRequest` Pydantic model: `doc_type`, `metadata` (dict), `extractor_version`
- `MetadataResponse` Pydantic model: `guid`, `doc_type`, `metadata`, `extracted_at`, `extractor_version`

### Use Cases Enabled
- Store Twitter/X post metadata: author, likes, retweets, source URL
- Store YouTube video metadata: channel, view count, duration, transcript
- Store webpage metadata: title, description, Open Graph image

---

## [1.1.0] - 2026-01-03

### Added
- Content hash (MD5) stored on every document for deduplication
- `GET /by-hash/{hash}` - find a document by its content hash
- `PATCH /by-hash` - update document content addressed by hash
- `PUT /bulk` - create multiple documents in a single request (up to 100)
- `GET /activity` - tag statistics with last-updated timestamps, filterable by tag prefix
- `GET /versions/{guid}` - document version chain (chronological evolution via embedding similarity)
- `GET /schema` - tag schema configuration endpoint
- `GET /visualize` - 2D document map (UMAP + HDBSCAN clustering)
- Tag schema system (`mesh.yaml`) with auto-inference from nearest neighbors
- MCP server (`mcp_server.py`) with 12 tools for Claude Desktop/Code integration
- API key authentication (`AUTH_REQUIRED`, `API_KEYS` env vars)
- Per-IP sliding window rate limiting (search, embed, heavy tiers)
- CORS support via `CORS_ORIGINS` env var
- `TRUST_PROXY_HEADERS` for Cloudflare/Traefik proxy deployments
- Model sentinel file to detect EMBEDDING_MODEL mismatch at startup

---

## [1.0.0] - 2025-12-29

### Added
- Initial release of Mesh document management and semantic search API
- Document CRUD with auto-generated GUIDs (`doc_xxxxxxxx` format)
- Semantic search powered by `intfloat/multilingual-e5-base` sentence-transformers model
- Background embedding worker: documents saved immediately, embeddings generated asynchronously
- Tag-based document filtering (`GET /?tag=type:worklog`)
- Smart root endpoint: returns API help without parameters, document list with parameters
- `PUT /` - create document
- `GET /` - help or document list
- `POST /search` - semantic similarity search with configurable threshold
- `GET /{guid}` - retrieve document by GUID
- `GET /health` - health check for database and embedding service
- `GET /stats` - document count, embedding queue depth, and service statistics
- `GET /help` - full inline API documentation
- IP whitelist middleware with CIDR range support, reads real IP from Cloudflare headers
- PostgreSQL 16 + pgvector backend with connection pooling via asyncpg
- Docker Compose deployment with Traefik reverse proxy integration
- Environment-based configuration via `.env` / `python-dotenv`
