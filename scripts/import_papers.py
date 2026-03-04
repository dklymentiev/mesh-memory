#!/usr/bin/env python3
"""
Import Papers documents into Mesh Memory.

Reads documents from Papers PostgreSQL DB (via psql), loads markdown content
from files, and bulk-imports into Mesh API with tags derived from path and metadata.
"""

import os
import sys
import json
import time
import subprocess
import requests

# Configuration
PAPERS_FILES_DIR = "/srv/papers/files"
MESH_API_URL = os.environ.get("MESH_API_URL", "http://localhost:8000")
BULK_SIZE = 50  # documents per batch (Mesh limit is 100)
MAX_CONTENT = 49000  # stay under Mesh's 50000 char limit


def get_papers_documents():
    """Fetch all active documents from Papers DB via psql."""
    query = """
    SELECT json_agg(row_to_json(t))
    FROM (
        SELECT guid::text, path, title, doc_type, created_by,
               to_char(created_at, 'YYYY-MM-DD') as created_date,
               tags, summary
        FROM documents
        WHERE deleted_at IS NULL
        ORDER BY created_at
    ) t;
    """
    env = os.environ.copy()
    env["PGPASSWORD"] = "papers_secure_2026"
    result = subprocess.run(
        ["psql", "-h", "192.168.55.205", "-p", "5432", "-U", "papers_app",
         "-d", "koval_papers", "-t", "-A", "-c", query],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        print(f"psql error: {result.stderr}")
        sys.exit(1)

    output = result.stdout.strip()
    if not output or output == "null":
        return []

    return json.loads(output)


def path_to_tags(path):
    """Derive Mesh tags from Papers document path."""
    tags = ["source:papers"]
    parts = path.replace(".md", "").split("/")

    # Extract known path segments
    if "worklog" in parts:
        tags.append("type:worklog")
    elif "plans" in parts or "plan" in path.lower():
        tags.append("type:plan")
    elif "reports" in parts or "report" in path.lower():
        tags.append("type:report")
    elif "daily" in parts:
        tags.append("type:daily-report")
    elif "monitoring" in parts:
        tags.append("type:monitoring")
    else:
        tags.append("type:document")

    # Team/section
    if parts[0] == "teams":
        if len(parts) > 1:
            tags.append(f"team:{parts[1]}")
        # Project from path
        if "development" in parts:
            dev_idx = parts.index("development")
            if dev_idx + 1 < len(parts):
                project = parts[dev_idx + 1]
                if project != "worklog":
                    tags.append(f"project:{project}")
        elif "infrastructure" in parts:
            tags.append("project:infrastructure")
    elif parts[0] == "infrastructure":
        tags.append("project:infrastructure")
    elif parts[0] == "shared":
        tags.append("scope:shared")
    elif parts[0] == "users":
        if len(parts) > 1:
            tags.append(f"author:{parts[1]}")

    return tags


def read_file_content(path):
    """Read markdown file content from Papers files directory."""
    full_path = os.path.join(PAPERS_FILES_DIR, path)
    if not os.path.exists(full_path):
        return None
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > MAX_CONTENT:
            content = content[:MAX_CONTENT]
        return content
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return None


def import_batch(batch, retries=5):
    """Send a batch of documents to Mesh bulk endpoint with retry."""
    for attempt in range(retries):
        try:
            resp = requests.put(
                f"{MESH_API_URL}/bulk",
                json=batch,
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            print(f"  Bulk import error: {e}")
            return None
        except Exception as e:
            print(f"  Bulk import error: {e}")
            return None
    print("  Max retries reached for batch")
    return None


def main():
    print("Fetching documents from Papers DB...")
    docs = get_papers_documents()
    print(f"Found {len(docs)} documents")

    # Clean up test docs first
    print("Cleaning up test documents...")
    try:
        resp = requests.get(f"{MESH_API_URL}/?limit=100&tag=type:test")
        if resp.ok:
            data = resp.json()
            for doc in data.get("documents", []):
                requests.delete(f"{MESH_API_URL}/{doc['guid']}")
                print(f"  Deleted test doc {doc['guid']}")
    except Exception:
        pass

    # Get already-imported papers-guid tags to skip duplicates
    print("Checking for already-imported documents...")
    existing_guids = set()
    try:
        offset = 0
        while True:
            resp = requests.get(
                f"{MESH_API_URL}/?limit=100&offset={offset}&tag=source:papers"
            )
            if not resp.ok:
                break
            data = resp.json()
            documents = data if isinstance(data, list) else data.get("documents", [])
            if not documents:
                break
            for doc in documents:
                for tag in doc.get("tags", []):
                    if tag.startswith("papers-guid:"):
                        existing_guids.add(tag.replace("papers-guid:", ""))
            offset += 100
            if len(documents) < 100:
                break
    except Exception as e:
        print(f"  Warning: could not check existing docs: {e}")
    print(f"Already imported: {len(existing_guids)} documents")

    batch = []
    total_imported = 0
    total_skipped = 0
    total_errors = 0

    for i, doc in enumerate(docs):
        path = doc["path"]
        title = doc.get("title") or ""
        guid = doc["guid"]

        # Skip already imported
        if guid in existing_guids:
            total_skipped += 1
            continue

        # Read file content
        content = read_file_content(path)
        if not content or len(content.strip()) < 10:
            total_skipped += 1
            continue

        # Build content with title prefix for better search
        if title and not content.startswith(f"# {title}"):
            full_content = f"# {title}\n\n{content}"
        else:
            full_content = content

        if len(full_content) > MAX_CONTENT:
            full_content = full_content[:MAX_CONTENT]

        # Build tags
        tags = path_to_tags(path)
        tags.append(f"papers-guid:{guid}")
        tags.append(f"papers-path:{path}")

        # Add date
        if doc.get("created_date"):
            tags.append(f"date:{doc['created_date']}")

        # Add original Papers tags
        if doc.get("tags"):
            for t in doc["tags"]:
                tag = f"papers-tag:{t}"
                if tag not in tags:
                    tags.append(tag)

        # Add created_by
        if doc.get("created_by"):
            if f"author:{doc['created_by']}" not in tags:
                tags.append(f"author:{doc['created_by']}")

        batch.append({
            "content": full_content,
            "tags": tags,
            "source": "papers-import",
        })

        # Send batch when full
        if len(batch) >= BULK_SIZE:
            batch_start = total_imported + total_skipped + total_errors - len(batch) + 1
            print(f"  Importing batch ({len(batch)} docs, ~{i+1}/{len(docs)})...")
            result = import_batch(batch)
            if result:
                total_imported += result.get("created", 0)
                total_skipped += result.get("skipped", 0)
                print(f"    Created: {result.get('created', 0)}, Skipped: {result.get('skipped', 0)}")
            else:
                total_errors += len(batch)
            batch = []
            time.sleep(3)

    # Send remaining
    if batch:
        print(f"  Importing final batch ({len(batch)} docs)...")
        result = import_batch(batch)
        if result:
            total_imported += result.get("created", 0)
            total_skipped += result.get("skipped", 0)
            print(f"    Created: {result.get('created', 0)}, Skipped: {result.get('skipped', 0)}")
        else:
            total_errors += len(batch)

    print(f"\nDone!")
    print(f"  Imported: {total_imported}")
    print(f"  Skipped:  {total_skipped}")
    print(f"  Errors:   {total_errors}")

    # Check Mesh stats
    try:
        resp = requests.get(f"{MESH_API_URL}/stats")
        stats = resp.json()
        print(f"\nMesh stats:")
        print(f"  Total documents: {stats['documents']}")
        print(f"  Indexed: {stats['indexed']}")
        print(f"  Pending: {stats['pending']}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
