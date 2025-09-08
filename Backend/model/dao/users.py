from datetime import datetime, UTC
from typing import Self
from uuid import UUID

from pydantic import field_serializer
from sqlalchemy import ScalarResult
from sqlmodel import Column, Enum, Field, desc, select
from core.database import SQLDatabase
from model.dao.base import TimestampDAO, UuidDAO
from model.dao.enums import InviteStatus, Role
from model.dto.users import BusinessDTO, UserDTO, UserInviteDTO


class UserDAO(UuidDAO, TimestampDAO, table=True):
    # model config
    __tablename__ = "users"
    __dto_class__ = UserDTO

    first_name: str
    middle_name: str | None = None
    last_name: str
    email: str = Field(nullable=False, unique=True)
    password: str
    business_id: UUID = Field(
        default=None, foreign_key="businesses.id", ondelete="RESTRICT", nullable=True
    )

    role: Role = Field(sa_column=Column(Enum(Role)), default=Role.BUSINESS_USER)
    email_verified: bool = Field(nullable=False, default=False)
    invited_by: UUID = Field(
        default=None, foreign_key="users.id", ondelete="SET NULL", nullable=True
    )

    @classmethod
    async def filter(
        cls,
        *,
        db_resource: SQLDatabase,
        id: UUID | None = None,
        email: str | None = None,
        business_id: UUID | None = None,
        role: Role | None = None,
        email_verified: bool | None = None,
    ) -> ScalarResult[Self]:
        """Filter user entries by id, email, business_id and role_id."""

        async with db_resource.session() as session:
            query = select(UserDAO)

            if id is not None:
                query = query.where(UserDAO.id == id)
            if email is not None:
                query = query.where(UserDAO.email == email)
            if business_id is not None:
                query = query.where(UserDAO.business_id == business_id)
            if role is not None:
                query = query.where(UserDAO.role == role)
            if email_verified is not None:
                query = query.where(UserDAO.email_verified == email_verified)

            return await session.scalars(query)


class UserPasswordResetDAO(UuidDAO, TimestampDAO, table=True):
    # model config
    __tablename__ = "user_password_resets"

    user_id: UUID = Field(default=None, foreign_key="users.id", ondelete="CASCADE")
    expires_at: datetime = Field(nullable=False)
    is_used: bool = Field(nullable=False, default=False)

    @classmethod
    async def filter(
        cls,
        *,
        db_resource: SQLDatabase,
        id: UUID | None = None,
        user_id: UUID | None = None,
        is_used: bool | None = None,
        expires_after: datetime | None = None,
        expires_before: datetime | None = None,
    ) -> ScalarResult[Self]:
        """Filter user entries by id, email, business_id and role_id."""
        async with db_resource.session() as session:
            query = select(UserPasswordResetDAO)
            if id is not None:
                query = query.where(UserPasswordResetDAO.id == id)
            if user_id is not None:
                query = query.where(UserPasswordResetDAO.user_id == user_id)
            if is_used is not None:
                query = query.where(UserPasswordResetDAO.is_used == is_used)
            if expires_after is not None:
                query = query.where(UserPasswordResetDAO.expires_at >= expires_after)
            if expires_before is not None:
                query = query.where(UserPasswordResetDAO.expires_at <= expires_before)

            return await session.scalars(query)


class BusinessDAO(UuidDAO, TimestampDAO, table=True):
    # model config
    __tablename__ = "businesses"
    __dto_class__ = BusinessDTO

    name: str
    industry: str
    owner_id: UUID = Field(foreign_key="users.id", ondelete="CASCADE")

    @classmethod
    async def filter(
        cls,
        *,
        db_resource: SQLDatabase,
        id: UUID | None = None,
        industry: str | None = None,
        owner_id: UUID | None = None,
    ) -> ScalarResult[Self]:
        """Filter business entries by id, industry and owner_id."""
        async with db_resource.session() as session:
            query = select(BusinessDAO)
            if id is not None:
                query = query.where(BusinessDAO.id == id)
            if industry is not None:
                query = query.where(BusinessDAO.industry == industry)
            if owner_id is not None:
                query = query.where(BusinessDAO.owner_id == owner_id)

            return await session.scalars(query)


class UserInviteDAO(UuidDAO, TimestampDAO, table=True):
    # model config
    __tablename__ = "user_invites"
    __dto_class__ = UserInviteDTO

    invitee_email: str
    inviter_id: UUID = Field(foreign_key="users.id", ondelete="CASCADE")
    business_id: UUID = Field(foreign_key="businesses.id", ondelete="CASCADE")
    status: InviteStatus = Field(
        sa_column=Column(Enum(InviteStatus)),
        default=InviteStatus.CREATED,
    )
    expires_at: datetime = Field(nullable=False)

    @field_serializer("expires_at")
    def serialize_dt(self, dt: datetime, _info):
        return dt.replace(tzinfo=UTC).isoformat(timespec="seconds")

    @classmethod
    async def filter(
        cls,
        *,
        db_resource: SQLDatabase,
        id: UUID | None = None,
        inviter_id: UUID | None = None,
        invitee_email: str | None = None,
        status_in: list[InviteStatus] | None = None,
        expires_after: datetime | None = None,
        expires_before: datetime | None = None,
    ) -> ScalarResult[Self]:
        """Filter user invite entries by id, inviter_id, invitee_email, status and expiry range, ordered by recency."""
        async with db_resource.session() as session:
            query = select(UserInviteDAO)
            if id is not None:
                query = query.where(UserInviteDAO.id == id)
            if inviter_id is not None:
                query = query.where(UserInviteDAO.inviter_id == inviter_id)
            if invitee_email is not None:
                query = query.where(UserInviteDAO.invitee_email == invitee_email)
            if status_in is not None:
                query = query.where(UserInviteDAO.status.in_(status_in))
            if expires_after is not None:
                query = query.where(UserInviteDAO.expires_at >= expires_after)
            if expires_before is not None:
                query = query.where(UserInviteDAO.expires_at <= expires_before)

            query = query.order_by(desc(UserInviteDAO.created_at))

            return await session.scalars(query)
