from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.di_container import DependencyContainer
from core.logger import configure_uvicorn_logger


@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """Context manager that runs tasks at app start and shutdown."""

    # Run at start
    configure_uvicorn_logger()
    container = DependencyContainer()

    # Yield to app
    yield

    # Run at shutdown
    try:
        await container.shutdown_resources()
    except Exception:
        pass
