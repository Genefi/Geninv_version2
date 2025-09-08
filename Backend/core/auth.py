from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from core.constants import JWT_ALGORITHM
from core.database import SQLDatabase
from core.di_container import DependencyContainer
from core.environment import settings
from core.exceptions import UnauthorizedException
from core.logger import app_logger
from model.dao.users import UserDAO
from model.dto.users import UserDTO

from dependency_injector.wiring import Provide, inject


oauth_scheme = HTTPBearer()
DatabaseDependency = Depends(Provide[DependencyContainer.pg_database])


@inject
async def verify_access_token(
    request: Request,
    token: HTTPAuthorizationCredentials = Security(oauth_scheme),
    pg_database: SQLDatabase = DatabaseDependency,
):
    try:
        claims: dict = jwt.decode(
            token.credentials, settings.ACCESS_TOKEN_SECRET, algorithms=[JWT_ALGORITHM]
        )
        user_email = claims.pop("sub", None)
        if user_email is None:
            app_logger.debug("No user found in sub")
            raise UnauthorizedException()

        user = UserDTO.model_validate(claims)
    except jwt.InvalidTokenError as exc:
        app_logger.debug("Invalid Token")
        app_logger.debug(f"Error: {exc}")
        raise UnauthorizedException()

    user = (
        await UserDAO.filter(
            id=user.id,
            email_verified=True,
            db_resource=pg_database,
        )
    ).first()

    if user is None:
        app_logger.debug("No user found")
        raise UnauthorizedException()
    request.state.user = user.to_dto()
