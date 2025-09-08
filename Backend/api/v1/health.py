from fastapi import APIRouter, status
from openai import BaseModel


health_router = APIRouter(tags=["Health"])


class HealthCheck(BaseModel):
    name: str = "GenInv"
    version: str
    description: str = "Generative Inventory Service"
    status: str


@health_router.get("/status", status_code=status.HTTP_200_OK)
async def health_check() -> HealthCheck:
    return HealthCheck(version="0.1.0", status="ok")
