# Quick test in Python console or create test_hash.py
from app.services.ingestion.media.media_hash_utils import MediaHashComputer

print("✅ MediaHashComputer imported successfully!")
print(f"Image duplicate threshold: {MediaHashComputer.IMAGE_DUPLICATE_THRESHOLD} bits")
