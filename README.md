[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

# Mesh Memory

![Mesh Demo](app/demo.gif)

**Save notes, decisions, worklogs. Find them by meaning, not keywords.**

You write a note "fixed the authentication bug in login flow". Two weeks later you search for "login problems" -- and regular search finds nothing because those words aren't in your note. Mesh finds it, because it understands that "authentication bug" and "login problems" mean the same thing.

```bash
# Save a note
curl -X PUT localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"content": "Fixed authentication bug in login flow", "tags": ["type:worklog"]}'

# Search by meaning -- not exact words
curl -X POST localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "login problems"}'
# -> finds your note, even though "problems" isn't in the text
```

Self-hosted, single Docker container, no external APIs. Uses a local AI model for embeddings -- your data never leaves your server.

| Regular Search | Mesh Memory |
|----------------|-------------|
| "auth bug" finds "auth bug" | "auth bug" finds "login problem" |
| Exact match only | Understands synonyms and context |
| One language | Works across 100+ languages |

## Quick Start

```bash
# Clone and configure
git clone https://github.com/dklymentiev/mesh-memory.git
cd mesh-memory
cp .env.example .env
```

Open `.env` and set a password for PostgreSQL:

```
POSTGRES_PASSWORD=pick_any_password_here
```

That's the only required change. Everything else has sensible defaults.

```bash
# Start
docker compose up -d

# Check it works (takes 2-3 min on first launch)
curl http://localhost:8000/health
# {"status": "healthy", "embeddings": "ready"}
```

First launch downloads PostgreSQL, builds the API image, and bakes in the AI model (~560 MB). Subsequent starts are instant.

```bash
# Save your first document
curl -X PUT http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"content": "Decided to use PostgreSQL for the project", "tags": ["type:decision"]}'

# Search by meaning
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "which database did we choose?"}'
```

It finds your document -- even though the words are completely different.

For a step-by-step walkthrough including Web UI, tagging, and MCP integration, see the **[Getting Started Guide](docs/configuration.md)**.

## How It Works

Mesh has three layers:

```
Layer 1: DOCUMENTS        What you save -- notes, decisions, worklogs, research
Layer 2: EMBEDDINGS       AI model turns text into 768-dim vectors (automatic)
Layer 3: SEARCH           Find by meaning using vector similarity
```

**You save a document** -- plain text with optional tags:

```bash
curl -X PUT localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"content": "Chose Redis for caching because...", "tags": ["type:decision"]}'
```

**Mesh automatically:**
1. Stores the document in PostgreSQL
2. Generates an embedding vector in the background
3. Auto-adds `date:2026-02-23` and `source:api` tags
4. After indexing, infers tags from similar documents (e.g. `type:decision`, `topic:redis`)

**You search by meaning:**

```bash
curl -X POST localhost:8000/search \
  -d '{"query": "what caching solution did we pick?"}'
# -> finds the Redis decision, even with zero word overlap
```

No configuration. No prompt engineering. No API keys for external services. The AI model runs locally inside the container.

## What Can You Do With It

### Personal knowledge base

Save everything you learn, decide, or think about. Find it later by describing what you're looking for in plain language.

### Team memory

Multiple people save worklogs and decisions. Anyone can search across everything by meaning. "How did we handle rate limiting?" finds the relevant design doc even if it never mentions those words.

### AI agent memory

Give your AI agents persistent memory. They save context, decisions, and progress. Later they (or other agents) can search for relevant information.

```bash
# Agent saves its work
mesh add "Implemented retry logic with exponential backoff for API calls" \
  "type:worklog,date:2026-02-23"

# Another agent finds it
mesh search "error handling patterns" 5
```

### MCP integration

Mesh includes an MCP server (12 tools) that works with Claude Desktop, Claude Code, and Cursor. Your AI assistant can search, save, and organize documents directly.

## Demo Mode

The repo ships with 1000 pre-generated documents across 15 categories -- software development, AI/ML, infrastructure, design, DevOps, career, and more. Great for trying out the UI and search.

```bash
# Start the demo instance (auto-seeds 1000 documents)
docker compose up -d api-demo

# Open the galaxy map
open http://localhost:8001/ui/map.html

# Or the search UI
open http://localhost:8001/ui/
```

The demo auto-seeds on first startup when the database is empty. You can also seed manually:

```bash
python scripts/seed_demo.py --api http://localhost:8001
```

## Features

### Semantic search

Search by meaning, not keywords. Ask "login problems" and find "authentication bug". Works across 100+ languages thanks to the [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) model.

### Smart auto-tagging

Three levels of automatic tagging (no AI API calls needed):

| Level | When | Example |
|-------|------|---------|
| **Defaults** | On save | `date:2026-02-23`, `source:api` |
| **Neighbor inference** | After embedding | 7 nearest neighbors share `type:worklog` -> auto-added |
| **Manual** | On save | Your tags always take priority |

Configured in `mesh.yaml`. Custom tags beyond the schema are always accepted.

### Version chains

Track how a document evolved over time. Mesh uses embedding similarity to find related versions automatically -- no manual linking needed.

```bash
curl "localhost:8000/versions/doc_4fa2cae8?threshold=0.85"
```

### Built-in Web UI

Served at `/ui/`:

- **Search page** (`/ui/`) -- semantic search with card grid and markdown rendering
- **Galaxy map** (`/ui/map.html`) -- Three.js visualization with clustering by category, timeline view, and real-time search highlighting

No build step. Native ES modules, baked into the Docker image.

### MCP Server (12 tools)

For Claude Desktop, Claude Code, and Cursor:

| Tool | What it does |
|------|-------------|
| `mesh_search` | Semantic search |
| `mesh_add` | Save document (auto-tags applied) |
| `mesh_update` | Update content or tags |
| `mesh_delete` | Delete by GUID |
| `mesh_get` | Get by GUID |
| `mesh_bytag` | Filter by tags |
| `mesh_versions` | Document version chain |
| `mesh_projects` | List projects with stats |
| `mesh_stats` | Memory statistics |
| `mesh_recent` | Recent documents |
| `mesh_tags` | List tags with counts |
| `mesh_schema` | Show tag schema |

Setup:

```json
{
  "mcpServers": {
    "mesh": {
      "command": "python3",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "MESH_API_URL": "https://your-mesh-instance.com"
      }
    }
  }
}
```

### AI Categorizer (opt-in)

Automatic document classification using an LLM. Disabled by default -- works without any external API keys.

**How documents get organized:**

| Step | What happens | Requires |
|------|-------------|----------|
| 1. Save document | Tags you provide are stored as-is | Nothing |
| 2. Auto-tags | `date:2026-02-23`, `source:api` added automatically | Nothing |
| 3. Neighbor inference | After embedding, tags are inferred from 7 nearest neighbors | 5+ existing documents |
| 4. AI categorization | LLM assigns `category:*` and `subcategory:*` tags | LLM API key (opt-in) |

Steps 1-3 work out of the box. Step 4 requires an OpenAI-compatible API.

**Enable the AI categorizer:**

```env
# .env
CATEGORIZER_ENABLED=true
LLM_API_URL=https://openrouter.ai/api/v1   # or any OpenAI-compatible endpoint
LLM_API_KEY=sk-...
LLM_MODEL=google/gemini-2.0-flash-001      # or gpt-4o-mini, claude-haiku, etc.
```

**Generate a taxonomy from your documents:**

```bash
# LLM analyzes 50 diverse documents and designs 10-15 categories
curl -X POST localhost:8000/categorizer/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"mode": "designed"}'

# View the generated taxonomy
curl localhost:8000/categorizer/taxonomy
```

The bootstrap samples your documents, asks the LLM to propose meaningful categories (like "Infrastructure", "Product Decisions", "Research Notes"), then classifies every document. New documents are classified automatically as they arrive.

**Use your own categories:**

You can also define a custom taxonomy by saving it as a document:

```bash
curl -X PUT localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "{\"version\": 1, \"categories\": [{\"id\": \"engineering\", \"name\": \"Engineering\", \"description\": \"Code, architecture, debugging\"}, {\"id\": \"product\", \"name\": \"Product\", \"description\": \"Features, roadmap, user feedback\"}, {\"id\": \"operations\", \"name\": \"Operations\", \"description\": \"Deployments, monitoring, incidents\"}]}",
    "tags": ["type:taxonomy", "topic:ai-categorizer", "status:active"]
  }'
```

After saving, restart the service or call `POST /categorizer/bootstrap` to apply. The Galaxy Map visualization uses these categories to create clusters.

**Classify existing documents in bulk:**

```bash
# Classify up to 500 untagged documents
curl -X POST localhost:8000/categorizer/batch \
  -H "Content-Type: application/json" \
  -d '{"limit": 500}'
```

### Bulk import

Import up to 100 documents per request:

```bash
# Import from JSONL
python scripts/seed_demo.py --api http://localhost:8000

# Or use the API directly
curl -X PUT localhost:8000/bulk \
  -H "Content-Type: application/json" \
  -d '[{"content": "doc 1", "tags": ["type:note"]}, {"content": "doc 2", "tags": ["type:note"]}]'
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/` | Create document |
| `GET` | `/` | Help (no params) or list documents (with params) |
| `POST` | `/search` | Semantic search |
| `GET` | `/{guid}` | Get document |
| `PATCH` | `/{guid}` | Update document |
| `DELETE` | `/{guid}` | Delete document |
| `PUT` | `/bulk` | Bulk create (up to 100) |
| `GET` | `/versions/{guid}` | Version chain |
| `GET` | `/tags` | All tags with counts |
| `GET` | `/stats` | Statistics |
| `GET` | `/health` | Health check |
| `GET` | `/schema` | Tag schema |
| `GET` | `/browse` | All documents (for client-side search) |
| `GET` | `/keyword` | Fast keyword search (no embedding) |
| `PUT` | `/meta/{guid}` | Document metadata (JSONB) |
| `GET` | `/meta/{guid}` | Get document metadata |
| `DELETE` | `/meta/{guid}` | Delete document metadata |
| `GET` | `/meta?type=` | List metadata by document type |
| `POST` | `/embed` | Generate embedding for text |
| `POST` | `/embed/batch` | Batch embed multiple texts |
| `GET` | `/activity` | Document activity feed |
| `GET` | `/by-hash/{hash}` | Find document by content hash |
| `PATCH` | `/by-hash` | Update document by content hash |
| `POST` | `/summarize/{guid}` | AI-powered document summary |
| `GET` | `/visualize` | Visualization data for galaxy map |
| `GET` | `/help` | Full API documentation |

Full endpoint details: [docs/api-reference.md](docs/api-reference.md)

## Architecture

```
+--------------+  +--------------+  +--------------+
|  CLI / curl  |  |  MCP Server  |  |  Web / Apps  |
+------+-------+  +------+-------+  +------+-------+
       |                 |                 |
       +-----------------+-----------------+
                         |
              +----------v----------+
              |  Mesh API (FastAPI) |
              |  localhost:8000     |
              |                     |
              |  +---------------+  |
              |  |  Tag Schema   |  |  mesh.yaml
              |  |  (defaults,   |  |  auto-tagging
              |  |   inference)  |  |  validation
              |  +---------------+  |
              +----------+----------+
                         |
              +----------v--------------+
              |  PostgreSQL + pgvector  |
              |  +- documents          |
              |  +- doc_embeddings     |  768-dim vectors
              |  +- document_metadata  |  JSONB
              +-----------+------------+
```

## Deployment

### Docker Compose (recommended)

```bash
docker compose up -d
```

### Without Docker

```bash
# You need: Python 3.11+, PostgreSQL 16 with pgvector
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set DATABASE_URL
python -m mesh.main
```

### With Traefik

```bash
echo "TRAEFIK_HOST=mesh.example.com" >> .env
docker compose -f docker-compose.yml -f docker-compose.traefik.yml up -d
```

### Environment Variables

```bash
# Core
DATABASE_URL=postgresql://postgres:password@postgres:5432/meshdb
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# Security
AUTH_REQUIRED=false           # Require X-API-Key header
API_KEYS=key1,key2            # Valid API keys
IP_WHITELIST=                 # CIDR ranges (empty = allow all)
CORS_ORIGINS=                 # Allowed origins

# Rate limiting (requests/min, 0 = unlimited)
RATE_LIMIT_SEARCH=60
RATE_LIMIT_EMBED=30

# AI Categorizer (opt-in)
CATEGORIZER_ENABLED=false
LLM_API_URL=                  # OpenAI-compatible endpoint
LLM_API_KEY=
LLM_MODEL=
```

Full configuration: [docs/configuration.md](docs/configuration.md)

## Tag System

Mesh supports any tags. Recommended types:

| Tag | Description |
|-----|-------------|
| `type:worklog` | Completed work |
| `type:note` | Notes and ideas |
| `type:decision` | Architecture decisions |
| `type:task` | Action items |
| `type:research` | Analysis and findings |
| `type:rfc` | Proposals |
| `status:active` | In progress |
| `status:completed` | Done |
| `date:YYYY-MM-DD` | When created (auto-added) |
| `guid:project-id` | Project marker |

## How Mesh Compares

| | Mesh | Pinecone | Weaviate | ChromaDB |
|--|------|----------|----------|----------|
| Self-hosted | Yes | No (SaaS) | Yes | Yes |
| Database | PostgreSQL + pgvector | Proprietary | Custom | SQLite |
| Auto-tagging | Yes (neighbor inference) | No | No | No |
| MCP integration | Built-in | No | No | No |
| Web UI | Built-in | Dashboard | Console | No |
| Setup | `docker compose up` | Sign up | Helm chart | `pip install` |
| Best for | Knowledge management | Large-scale search | Knowledge graphs | Prototyping |

Mesh is designed for personal and team knowledge management -- not for billion-scale vector search. If you need a "memory" for your projects with smart search, tagging, and simple deployment, Mesh is for you.

## Project Structure

```
mesh/                     # Python package (FastAPI app)
  main.py                 # Application + auto-tagging + version chains
  crud.py                 # Document CRUD + search
  embeddings.py           # Embedding generation
  tag_schema.py           # Tag schema + auto-inference
  mesh.yaml               # Tag configuration
  categorizer/            # AI classification (opt-in)
ui/                       # Built-in web UI (search + galaxy map)
scripts/                  # Dev utilities
  generate_demo.py        # Generate demo data (--output JSONL, --seed)
  seed_demo.py            # Load JSONL into Mesh via API
  enrich_demo_tags.py     # Tag enrichment rules
examples/                 # Usage examples + demo seed data
  demo-seed.jsonl         # 1000 pre-generated documents
mcp_server.py             # MCP server (12 tools)
tests/                    # Test suite
docs/                     # Documentation
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

For security issues, see [SECURITY.md](SECURITY.md).

## License

MIT
