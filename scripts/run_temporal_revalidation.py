import asyncio
from app.services.validation.temporal_revalidation_engine import (
    run_temporal_revalidation
)

if __name__ == "__main__":
    asyncio.run(run_temporal_revalidation(batch_size=50))
