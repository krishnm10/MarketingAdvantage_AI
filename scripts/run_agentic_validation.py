import asyncio
from app.services.validation.agentic_validation_worker import run_agentic_validation

if __name__ == "__main__":
    asyncio.run(run_agentic_validation(batch_size=50))
