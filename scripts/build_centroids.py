#!/usr/bin/env python3
"""
Build centroids for custom taxonomy based on Papers path mapping.

Maps Papers document paths to our custom categories, computes centroid
embeddings from matching documents, and saves them to the Mesh DB.
Then re-saves the custom taxonomy and runs batch classification.

No numpy dependency -- uses pure Python for vector math.
"""

import hashlib
import json
import subprocess
import sys

import requests

MESH_API = "http://localhost:8000"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_USER = "postgres"
DB_PASS = "6LtsT3IHcV6Z6RF4by6K"
DB_NAME = "meshdb"


# Mapping: papers-path pattern -> (category_id, subcategory_id or None)
PATH_RULES = [
    # Portal CRM
    ("portal/worklog/", "portal", "worklog"),
    ("portal/plans/", "portal", "planning"),
    ("portal/_archive/plans/", "portal", "planning"),
    ("portal/docs/", "portal", "documentation"),
    ("portal/changelog/", "portal", "documentation"),
    ("portal/procedures/", "portal", "documentation"),
    ("portal/crm-manager/", "portal", "documentation"),
    ("portal-deployment", "portal", "deployment"),
    ("portal/products/", "portal", None),
    ("portal/", "portal", None),
    ("data-import/", "portal", None),

    # Websites
    ("seo-audit", "website", "seo"),
    ("seo/", "website", "seo"),
    ("pagespeed/", "website", "performance"),
    ("koval-distillery.com/worklog/", "website", "cms"),
    ("koval-cms/", "website", "cms"),
    ("unified-checkout/", "website", "cms"),
    ("koval-distillery.com/", "website", None),
    ("koval-distillery/", "website", None),
    ("threshandwinnowspirits.com/", "website", None),

    # Infrastructure
    ("monitoring/", "infrastructure", "monitoring"),
    ("reatan/", "infrastructure", "monitoring"),
    ("infrastructure/worklog/", "infrastructure", "setup"),
    ("infrastructure/", "infrastructure", "setup"),
    ("backup", "infrastructure", "backup"),

    # AI & Agents
    ("scriber/reports/", "ai-agents", "scriber"),
    ("scriber/", "ai-agents", "scriber"),
    ("agents/", "ai-agents", "tools"),
    ("mcp-servers/", "ai-agents", "tools"),
    ("ai-gateway/", "ai-agents", "tools"),
    ("ai-chat/", "ai-agents", "tools"),
    ("ai-agents/", "ai-agents", "tools"),
    ("papers/", "ai-agents", "internal-apps"),
    ("planner/", "ai-agents", "internal-apps"),
    ("mashpad/", "ai-agents", "internal-apps"),
    ("mi/", "ai-agents", "internal-apps"),
    ("web-serfer/", "ai-agents", "tools"),

    # Social Analytics
    ("social-analyst/daily/", "social-analytics", "daily-digest"),
    ("social-analytics/reddit/", "social-analytics", "daily-digest"),
    ("social-analytics/x/", "social-analytics", "daily-digest"),
    ("social-analytics/", "social-analytics", "analysis"),
    ("social-analyst/", "social-analytics", "analysis"),

    # Advertising
    ("advertising/worklog/", "advertising", "campaigns"),
    ("advertising/campaigns/", "advertising", "campaigns"),
    ("advertising-research/", "advertising", "strategy"),
    ("advertising/", "advertising", None),

    # Market Research
    ("market-research/background/", "market-research", "industry"),
    ("market-research/intelligence/", "market-research", "industry"),
    ("market-research/market-analysis/", "market-research", "competitors"),
    ("market-research/forecasting/", "market-research", "consumers"),
    ("market-research/resources/", "market-research", None),
    ("market-research/", "market-research", None),

    # Distillery-2
    ("distillery-2/research/", "distillery-2", "research"),
    ("distillery-2/phase", "distillery-2", "research"),
    ("distillery-2/", "distillery-2", None),

    # Social Media Plans
    ("plans/linkedin", "social-media", None),
    ("plans/instagram", "social-media", None),
    ("plans/pinterest", "social-media", None),
    ("plans/tiktok", "social-media", None),
    ("plans/reddit", "social-media", None),
    ("plans/social", "social-media", None),

    # Roadmap
    ("roadmap", "roadmap", None),
    ("dashboard", "roadmap", None),
    ("plans/", "roadmap", None),
]


# Our taxonomy definition
TAXONOMY = {
    "portal": ("Portal CRM", "CRM system development, salary reports, barrels, distributors"),
    "website": ("Websites", "KOVAL and T&W websites, SEO, PageSpeed, CMS"),
    "infrastructure": ("Infrastructure", "Server setup, Docker, databases, backups, monitoring"),
    "ai-agents": ("AI & Agents", "AI agents, MCP, AI Gateway, Papers, Planner"),
    "social-analytics": ("Social Analytics", "Reddit/X monitoring, daily digests"),
    "advertising": ("Advertising", "Google Ads, social media advertising"),
    "market-research": ("Market Research", "Industry research, consumer analysis, competitors"),
    "distillery-2": ("Distillery-2 Project", "Second distillery planning, multi-phase research"),
    "social-media": ("Social Media Plans", "LinkedIn, Instagram, Pinterest, TikTok planning"),
    "roadmap": ("Roadmap & Planning", "IT roadmap, project plans, task tracking"),
}

SUBCATEGORIES = {
    ("portal", "worklog"): ("Worklogs", "Development worklogs and incidents"),
    ("portal", "planning"): ("Plans", "Implementation plans"),
    ("portal", "documentation"): ("Documentation", "System docs, changelogs"),
    ("portal", "deployment"): ("Deployment", "Deployment procedures"),
    ("website", "seo"): ("SEO", "SEO audits, optimization"),
    ("website", "performance"): ("Performance", "PageSpeed reports"),
    ("website", "cms"): ("CMS Development", "CMS engine development"),
    ("website", "content"): ("Content", "Product data sheets"),
    ("infrastructure", "monitoring"): ("Monitoring", "VPS monitoring reports"),
    ("infrastructure", "setup"): ("Setup", "Server configuration, Docker"),
    ("infrastructure", "backup"): ("Backup", "Backup plans and procedures"),
    ("ai-agents", "scriber"): ("Scriber Reports", "Daily scriber reports"),
    ("ai-agents", "tools"): ("Agent Tools", "MCP, AI Gateway, agent config"),
    ("ai-agents", "internal-apps"): ("Internal Apps", "Papers, Planner, Mashpad"),
    ("social-analytics", "daily-digest"): ("Daily Digests", "Daily Reddit/X reports"),
    ("social-analytics", "analysis"): ("Analysis", "Historical analysis, trends"),
    ("advertising", "campaigns"): ("Campaigns", "Active campaigns"),
    ("advertising", "strategy"): ("Strategy", "Platform strategies"),
    ("market-research", "industry"): ("Industry Data", "Craft distillery data"),
    ("market-research", "consumers"): ("Consumer Analysis", "Consumer segments"),
    ("market-research", "competitors"): ("Competitor Analysis", "Competitive landscape"),
    ("distillery-2", "research"): ("Research Phases", "Phase 1-11 outputs"),
    ("distillery-2", "synthesis"): ("Synthesis", "Final synthesis documents"),
}


def psql(query):
    """Execute query via psql and return output."""
    import os
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASS
    result = subprocess.run(
        ["psql", "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER,
         "-d", DB_NAME, "-t", "-A", "-c", query],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        print(f"psql error: {result.stderr[:200]}")
        return ""
    return result.stdout.strip()


def parse_vector(s):
    """Parse a pgvector string '[0.1,0.2,...]' into list of floats."""
    return [float(x) for x in s.strip("[]").split(",")]


def mean_vectors(vectors):
    """Compute mean of list of vectors (pure Python)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    result = [0.0] * dim
    n = len(vectors)
    for vec in vectors:
        for i in range(dim):
            result[i] += vec[i]
    return [x / n for x in result]


def classify_path(path):
    """Map a Papers path to (category_id, subcategory_id)."""
    for pattern, cat, sub in PATH_RULES:
        if pattern in path:
            return cat, sub
    return None, None


def main():
    # Step 1: Get all documents with their papers-path tags and embeddings
    print("Fetching documents with embeddings...")
    rows_json = psql("""
        SELECT json_agg(row_to_json(t))
        FROM (
            SELECT d.guid, d.tags,
                   e.embedding::text as embedding
            FROM documents d
            JOIN doc_embeddings e ON d.guid = e.guid
        ) t
    """)

    if not rows_json or rows_json == "null":
        print("No documents with embeddings found")
        sys.exit(1)

    docs = json.loads(rows_json)
    print(f"Found {len(docs)} documents with embeddings")

    # Step 2: Map each doc to a category based on papers-path tag
    category_embeddings = {}  # cat_id -> list of embedding vectors
    subcategory_embeddings = {}  # (cat_id, sub_id) -> list of embedding vectors

    classified = 0
    unclassified = 0

    for doc in docs:
        tags = doc.get("tags") or []
        path = None
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("papers-path:"):
                path = tag.replace("papers-path:", "")
                break

        if not path:
            unclassified += 1
            continue

        cat_id, sub_id = classify_path(path)
        if not cat_id:
            unclassified += 1
            continue

        # Parse embedding
        vec = parse_vector(doc["embedding"])

        if cat_id not in category_embeddings:
            category_embeddings[cat_id] = []
        category_embeddings[cat_id].append(vec)

        if sub_id:
            key = (cat_id, sub_id)
            if key not in subcategory_embeddings:
                subcategory_embeddings[key] = []
            subcategory_embeddings[key].append(vec)

        classified += 1

    print(f"Classified: {classified}, Unclassified: {unclassified}")
    print()

    # Step 3: Compute centroids
    print("Computing centroids:")
    centroids_data = []

    for cat_id, (name, desc) in TAXONOMY.items():
        if cat_id in category_embeddings:
            centroid = mean_vectors(category_embeddings[cat_id])
            count = len(category_embeddings[cat_id])
            centroids_data.append((cat_id, None, name, desc, centroid, count))
            print(f"  {cat_id}: {count} docs")
        else:
            print(f"  {cat_id}: 0 docs (skipping)")

    for (cat_id, sub_id), (name, desc) in SUBCATEGORIES.items():
        key = (cat_id, sub_id)
        if key in subcategory_embeddings:
            centroid = mean_vectors(subcategory_embeddings[key])
            count = len(subcategory_embeddings[key])
            centroids_data.append((sub_id, cat_id, name, desc, centroid, count))
            print(f"    {cat_id}/{sub_id}: {count} docs")

    # Step 4: Save centroids to DB
    print(f"\nSaving {len(centroids_data)} centroids to database...")
    psql("DELETE FROM category_centroids")

    for cat_id, parent_id, name, desc, centroid, count in centroids_data:
        vec_str = "[" + ",".join(f"{v:.8f}" for v in centroid) + "]"
        parent_val = f"'{parent_id}'" if parent_id else "NULL"
        # Escape single quotes in strings
        name_esc = name.replace("'", "''")
        desc_esc = desc.replace("'", "''")
        psql(f"""
            INSERT INTO category_centroids (category_id, parent_id, name, description, centroid, doc_count)
            VALUES ('{cat_id}', {parent_val}, '{name_esc}', '{desc_esc}', '{vec_str}'::vector, {count})
        """)

    # Step 5: Re-save custom taxonomy document
    print("Saving custom taxonomy document...")
    taxonomy_obj = {
        "version": 1,
        "categories": [
            {
                "id": cat_id,
                "name": name,
                "description": desc,
                "subcategories": [
                    {"id": s, "name": sub_name, "description": sub_desc}
                    for (c, s), (sub_name, sub_desc) in SUBCATEGORIES.items()
                    if c == cat_id
                ]
            }
            for cat_id, (name, desc) in TAXONOMY.items()
        ]
    }
    taxonomy_json = json.dumps(taxonomy_obj)
    content_hash = hashlib.md5(taxonomy_json.encode()).hexdigest()

    # Delete old taxonomy docs
    psql("DELETE FROM documents WHERE tags @> '{type:taxonomy,topic:ai-categorizer}'::text[]")

    # Escape for SQL
    taxonomy_sql = taxonomy_json.replace("'", "''")
    psql(f"""
        INSERT INTO documents (guid, content, content_hash, source, tags, created_at, updated_at)
        VALUES (
            'doc_custom_taxonomy',
            '{taxonomy_sql}',
            '{content_hash}',
            'system',
            '{{type:taxonomy,topic:ai-categorizer,status:active}}',
            NOW(), NOW()
        )
    """)

    # Step 6: Strip old ai-category tags from all documents
    print("Removing old classification tags...")
    psql("""
        UPDATE documents SET tags = array(
            SELECT t FROM unnest(tags) AS t
            WHERE t NOT LIKE 'ai-category:%' AND t NOT LIKE 'ai-subcategory:%'
        )
        WHERE EXISTS (
            SELECT 1 FROM unnest(tags) AS t
            WHERE t LIKE 'ai-category:%' OR t LIKE 'ai-subcategory:%'
        )
    """)

    print("\nDone! Restart mesh-api to load new taxonomy, then batch classify:")
    print("  docker compose restart api && sleep 10")
    print("  curl -X POST localhost:8000/categorizer/batch -H 'Content-Type: application/json' -d '{\"limit\":1000,\"force\":true}'")


if __name__ == "__main__":
    main()
