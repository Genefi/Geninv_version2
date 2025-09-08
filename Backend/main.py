from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from api.router import v1_router
from core.environment import settings
from core.exceptions import http_exception_handler, validation_exception_handler
from core.lifespan import lifespan_manager


app = FastAPI(
    lifespan=lifespan_manager,
    title="GenInv API",
    summary="Generative Inventory Service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.PARSED_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
