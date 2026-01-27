import asyncio
from app.services.validation.semantic_conflict_engine import run_semantic_conflict_detection

if __name__ == "__main__":
    asyncio.run(run_semantic_conflict_detection(batch_size=50))
