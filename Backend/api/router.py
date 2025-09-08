from fastapi import APIRouter, Depends
from core.auth import verify_access_token
from api.v1.auth import auth_router
from api.v1.users import users_router
from api.v1.health import health_router

v1_router = APIRouter(prefix="/api/v1")


authenticated_v1_router = APIRouter()

# Add routes that need authentication
authenticated_v1_router.include_router(
    users_router,
    dependencies=[Depends(verify_access_token)],
)


v1_router.include_router(auth_router)
v1_router.include_router(authenticated_v1_router)
v1_router.include_router(health_router)

__all__ = ["v1_router"]
