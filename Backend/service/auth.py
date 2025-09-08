from datetime import UTC, datetime, timedelta
from logging import Logger

import jwt
from core.constants import JWT_ALGORITHM
from core.database import SQLDatabase
from core.environment import settings
from core.exceptions import BadRequestException, UnauthorizedException
from core.logger import app_logger
from core.utils import CryptographyHelper
from model.dao.users import UserDAO
from model.dto.auth import LoginDTO

from model.dto.users import UserDTO


class AuthService:
    def __init__(self, pg_database: SQLDatabase, logger: Logger = app_logger):
        self._db = pg_database
        self._logger = logger

    async def authenticate_user(self, dto: LoginDTO) -> UserDTO:
        user = (await UserDAO.filter(email=dto.email, db_resource=self._db)).first()

        if not user:
            raise BadRequestException("Invalid email")

        if not user.email_verified:
            raise BadRequestException("Your email has not been verified yet.")

        if not CryptographyHelper.verify_password(dto.password, user.password):
            raise BadRequestException("Wrong password.")

        return user.to_dto()

    def generate_access_token(self, user: UserDTO):
        claims = user.model_dump(mode="json")
        claims["sub"] = user.email

        expire_at = datetime.now(UTC) + timedelta(
            seconds=settings.ACCESS_TOKEN_EXPIRES_AFTER_SECS
        )

        claims.update({"exp": expire_at})
        encoded_jwt = jwt.encode(
            claims,
            settings.ACCESS_TOKEN_SECRET,
            algorithm=JWT_ALGORITHM,
        )
        return encoded_jwt

    def generate_refresh_token(self, email: str):
        claims = {"sub": email}

        expire_at = datetime.now(UTC) + timedelta(
            seconds=settings.REFRESH_TOKEN_EXPIRES_AFTER_SECS
        )

        claims.update({"exp": expire_at})
        encoded_jwt = jwt.encode(
            claims,
            settings.REFRESH_TOKEN_SECRET,
            algorithm=JWT_ALGORITHM,
        )

        return encoded_jwt

    async def refresh_access_token(
        self, email: str, generate_refresh_token: bool = False
    ) -> tuple[str, str | None]:
        """Returns a tuple of access token and if specified, refresh token."""

        user = (await UserDAO.filter(email=email, db_resource=self._db)).first()

        if not user:
            raise UnauthorizedException()

        access_token = self.generate_access_token(user.to_dto())
        refresh_token = None

        if generate_refresh_token:
            refresh_token = self.generate_refresh_token(user.email)

        return access_token, refresh_token
