# ============================================================
# view_db_and_chroma.py  (FINAL FIXED & ROBUST VERSION)
# Safe diagnostic viewer for Postgres + ChromaDB.
# Works with SQLAlchemy Row, psycopg2, asyncpg-style schemas, etc.
# ============================================================

import os
import sys
from sqlalchemy import create_engine, text, inspect
import chromadb

# ----------------------------
# CONFIG (edit OR set via env)
# ----------------------------
DATABASE_URL = "postgresql://postgres:Mahadeva%40123@localhost/marketing_advantage"
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = "ingested_content"

SAMPLE_LIMIT = 5
RECENT_GCI_LIMIT = 50
BATCH_SIZE = 64

EXPECTED_TABLES = [
    "global_content_index",
    "ingested_content",
    "ingested_file",
    "pending_taxonomy",
    "taxonomy",
    "taxonomy_alias",
    "classification_logs",
    "business_classification",
    "business_taxonomy",
]


def fatal(msg):
    print("\nFATAL ERROR:", msg)
    sys.exit(1)


# ----------------------------------------
# ENGINE
# ----------------------------------------
def get_engine():
    if not DATABASE_URL:
        fatal("DATABASE_URL is not set. Export it or edit script.")
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        fatal(f"SQLAlchemy engine creation failed: {e}")


# ----------------------------------------
# Helpers
# ----------------------------------------
def list_tables(engine):
    insp = inspect(engine)
    tables = []
    for schema in insp.get_schema_names():
        try:
            for t in insp.get_table_names(schema=schema):
                tables.append((schema, t))
        except Exception:
            continue
    return tables


def find_table_schema(engine, table_name):
    insp = inspect(engine)
    for schema in insp.get_schema_names():
        try:
            if table_name in insp.get_table_names(schema=schema):
                return schema
        except Exception:
            pass
    return None


def get_table_columns(engine, schema, table):
    insp = inspect(engine)
    try:
        cols = insp.get_columns(table, schema=schema)
        return [c["name"] for c in cols]
    except Exception:
        return []


# ----------------------------------------
# Robust SQLAlchemy Row → dict conversion
# ----------------------------------------
def row_to_dict(row, engine=None, schema=None, table=None):
    # Try Row._mapping first
    try:
        if hasattr(row, "_mapping"):
            return dict(row._mapping)
    except Exception:
        pass

    # Try dict(row)
    try:
        return dict(row)
    except Exception:
        pass

    # Last fallback: zip with column names
    if engine and schema and table:
        try:
            cols = get_table_columns(engine, schema, table)
            return dict(zip(cols, row))
        except Exception:
            pass

    # Last fallback: stringify
    return {"row": str(row)}


# ----------------------------------------
# Sample table rows
# ----------------------------------------
def sample_table(engine, schema, table, limit=SAMPLE_LIMIT):
    cols = get_table_columns(engine, schema, table)
    order_col = None

    for c in ("created_at", "updated_at", "id"):
        if c in cols:
            order_col = c
            break

    try:
        if order_col:
            q = text(f"SELECT * FROM {schema}.{table} ORDER BY {order_col} DESC LIMIT :lim")
        else:
            q = text(f"SELECT * FROM {schema}.{table} LIMIT :lim")

        with engine.connect() as conn:
            rows = conn.execute(q, {"lim": limit}).fetchall()
    except Exception as e:
        print(f"Error sampling {schema}.{table}: {e}")
        return []

    return [row_to_dict(r, engine, schema, table) for r in rows]


# ----------------------------------------
# Count table
# ----------------------------------------
def count_table(engine, schema, table):
    try:
        q = text(f"SELECT COUNT(*) FROM {schema}.{table}")
        with engine.connect() as conn:
            (count,) = conn.execute(q).fetchone()
        return count
    except Exception as e:
        print(f"Error counting {schema}.{table}: {e}")
        return None


# ----------------------------------------
# Check GCI ↔ Chroma alignment
# ----------------------------------------
def check_gci(engine, schema, table, chroma_collection):
    print("\n--- Checking Global Content Index → Chroma alignment ---")

    cols = get_table_columns(engine, schema, table)
    desired = ["id", "semantic_hash", "business_id", "created_at", "updated_at"]
    available = [c for c in desired if c in cols]

    if not available:
        print(f"No usable columns in {schema}.{table}. Columns: {cols}")
        return

    select_cols = ", ".join(available)
    q = text(
        f"SELECT {select_cols} FROM {schema}.{table} "
        f"ORDER BY {available[-1]} DESC LIMIT :lim"
    )

    with engine.connect() as conn:
        try:
            rows = conn.execute(q, {"lim": RECENT_GCI_LIMIT}).fetchall()
        except Exception as e:
            print("Query failed:", e)
            return

    rows = [row_to_dict(r, engine, schema, table) for r in rows]

    hashes = [r.get("semantic_hash") for r in rows if r.get("semantic_hash")]
    print(f"Fetched {len(rows)} GCI rows | Semantic hashes: {len(hashes)}")

    found = set()

    for i in range(0, len(hashes), BATCH_SIZE):
        batch = hashes[i:i + BATCH_SIZE]
        try:
            resp = chroma_collection.get(ids=batch, include=["ids"])
        except Exception as e:
            print(f"Chroma get failed batch {i}: {e}")
            continue

        ids = resp.get("ids", []) if isinstance(resp, dict) else []
        for _id in ids:
            found.add(_id)

    missing = set(hashes) - found
    print(f"Chroma matched {len(found)} / {len(hashes)} | Missing {len(missing)}")
    if missing:
        print("Missing sample:", list(missing)[:10])

    print("\nSample GCI rows:")
    for r in rows[:5]:
        print(" -", {k: r.get(k) for k in available})


# ----------------------------------------
# MAIN
# ----------------------------------------
def main():
    engine = get_engine()

    print("\n=== Detected DB Tables ===")
    tables = list_tables(engine)
    for schema, table in tables:
        print(f" - {schema}.{table}")

    print("\n=== Checking Expected Tables ===")
    table_schemas = {}

    for t in EXPECTED_TABLES:
        schema = find_table_schema(engine, t)
        if not schema:
            print(f"[MISSING] {t}")
            continue

        table_schemas[t] = schema
        count = count_table(engine, schema, t)
        print(f"[OK] {schema}.{t} (rows={count})")

        sample = sample_table(engine, schema, t, SAMPLE_LIMIT)
        print(f" sample_rows={len(sample)}")
        for row in sample:
            keys = list(row.keys())[:8]
            print("  -", {k: row.get(k) for k in keys})

    # ----------------------------------------
    # Chroma DB
    # ----------------------------------------
    print("\nConnecting to Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)

    try:
        print("Chroma vector count:", collection.count())
    except:
        print("Chroma: count() not supported.")

    # ----------------------------------------
    # Validate GCI ↔ Chroma
    # ----------------------------------------
    if "global_content_index" in table_schemas:
        check_gci(engine, table_schemas["global_content_index"], "global_content_index", collection)

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
