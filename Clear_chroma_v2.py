# =============================================
# clear_chromadb_v2.py — Safe Full Wipe Script
# =============================================

import chromadb
from app.utils.logger import log_info

# =========================================================
# CONFIGURATION
# =========================================================
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "ingested_content"

# =========================================================
# CLEAR CHROMA DB
# =========================================================
def clear_chroma_collection():
    try:
        log_info(f"[ChromaCleanup] Connecting to Chroma at {CHROMA_PATH}...")
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        collections = [col.name for col in client.list_collections()]
        if COLLECTION_NAME in collections:
            log_info(f"[ChromaCleanup] Found collection '{COLLECTION_NAME}'. Deleting...")
            client.delete_collection(COLLECTION_NAME)
            log_info(f"[ChromaCleanup] ✅ Deleted collection '{COLLECTION_NAME}'.")
        else:
            log_info(f"[ChromaCleanup] No collection named '{COLLECTION_NAME}' found.")

        # Recreate fresh empty collection
        client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        log_info(f"[ChromaCleanup] ✅ Recreated empty '{COLLECTION_NAME}' collection.")

    except Exception as e:
        log_info(f"[ERROR] Failed to clear ChromaDB: {e}")


if __name__ == "__main__":
    log_info("[ChromaCleanup] Starting full ChromaDB cleanup...")
    clear_chroma_collection()
    log_info("[ChromaCleanup] ✅ Cleanup complete.")