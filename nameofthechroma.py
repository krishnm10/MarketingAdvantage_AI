import chromadb
from chromadb.config import Settings
CHROMA_PATH = "./chroma_db"
client = chromadb.Client(
    Settings(
    clien = chromadb.PersistentClient(path=CHROMA_PATH),
        anonymized_telemetry=False,
    )
)

print([c.name for c in clien.list_collections()])
