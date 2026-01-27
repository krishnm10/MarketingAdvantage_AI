import chromadb
import os

CHROMA_PATH = "./chroma_db"

def init_chromadb():
    """Initialize ChromaDB with fresh collection."""
    
    # Create directory
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    try:
        # Create client
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"✅ ChromaDB client created at {CHROMA_PATH}")
        
        # List existing collections
        collections = client.list_collections()
        print(f"📋 Existing collections: {[c.name for c in collections]}")
        
        # Delete any existing collections
        for col in collections:
            print(f"🗑️  Deleting old collection: {col.name}")
            client.delete_collection(col.name)
        
        # Create fresh collection
        collection = client.create_collection(
            name="ingested_content",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ Created collection: {collection.name}")
        print(f"📊 Vector count: {collection.count()}")
        print("\n🎉 ChromaDB initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    init_chromadb()
