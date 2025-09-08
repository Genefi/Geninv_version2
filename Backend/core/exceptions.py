from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from fastapi.responses import JSONResponse

from core.logger import app_logger
from model.dto.base import BaseResponseDTO


class GenInvServiceException(Exception):
    def __init__(self, message: str):
        self.message = message


class DAOException(GenInvServiceException): ...


class BadRequestException(HTTPException):
    def __init__(self, message: str = "Bad Request"):
        super().__init__(400, message)


class UnauthorizedException(HTTPException):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(401, message, headers={"WWW-Authenticate": "Bearer"})


class ForbiddenException(HTTPException):
    def __init__(self, message: str = "Forbidden"):
        super().__init__(403, message)


class NotFoundException(HTTPException):
    def __init__(self, message: str = "Not Found"):
        super().__init__(404, message)


class ConflictException(HTTPException):
    def __init__(self, message: str = "Conflict"):
        super().__init__(409, message)


# Exception Handlers
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    app_logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")

    response_dto = BaseResponseDTO(message="An error occurred")

    if isinstance(exc.detail, str):
        response_dto.message = exc.detail
    elif isinstance(exc.detail, list):
        response_dto.errors = exc.detail
    else:
        response_dto.errors = [exc.detail]

    return JSONResponse(
        status_code=exc.status_code,
        content=response_dto.model_dump(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []

    for error in exc.errors():
        location = ".".join([str(loc) for loc in error.get("loc", [])])
        message = error.get("msg", "")

        errors.append({"location": location, "error": message})

    response_dto = BaseResponseDTO(message="Validation Error")

    if errors:
        response_dto.errors = errors

    app_logger.error(f"Validation Error: {errors}")

    return JSONResponse(
        status_code=400,
        content=response_dto.model_dump(),
    )
