from typing import Any
from pydantic import BaseModel


class BaseResponseDTO(BaseModel):
    data: Any | None = None
    errors: list[Any] | None = None
    message: str
