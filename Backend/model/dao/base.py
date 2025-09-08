from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar, Self, Type, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, field_serializer
from sqlalchemy import ScalarResult
from sqlmodel import Field, SQLModel, select

from core.database import SQLDatabase
from core.exceptions import DAOException
from core.utils import naive_utc_now


T = TypeVar("T", bound="BaseModel")


class BaseDAO(SQLModel, ABC):
    __dto_class__: ClassVar[Type[T]]

    @classmethod
    async def filter(cls, **kwargs) -> ScalarResult[Self]:
        raise NotImplementedError(
            f"`filter` method has not been implemented for {cls.__name__}"
        )

    @classmethod
    @abstractmethod
    async def get(cls, id: Any) -> list[Self]: ...

    async def save(self, db_resource: SQLDatabase) -> None:
        async with db_resource.session() as session:
            session.add(self)
            await session.commit()
            await session.refresh(self)

    async def refresh_from_db(self, db_resource: SQLDatabase) -> Self:
        """Returns a refreshed DAO instance"""
        async with db_resource.session() as session:
            return await db_resource.refresh_detached_instance(session, self)

    def to_dto(self) -> T:
        """Convert the DAO instance to a DTO."""

        if self.__dto_class__ is None:
            raise DAOException(f"No DTO class set for {self.__class__.__name__}")

        return self.__dto_class__.model_validate(self.model_dump())

    @classmethod
    def from_dto(cls, dto_obj: T) -> Self:
        """Convert the DTO to a DAO instance."""
        return cls.model_validate(dto_obj, from_attributes=True)


class UuidDAO(BaseDAO):
    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        nullable=False,
    )

    @classmethod
    async def get(
        cls,
        id: UUID,
        db_resource: SQLDatabase,
    ) -> Self:
        """Filter table records by id."""
        async with db_resource.session() as session:
            query = select(cls).where(cls.id == id)
            return (await session.scalars(query)).one()


class IntegerIdDAO(BaseDAO):
    id: int = Field(
        primary_key=True,
        nullable=False,
    )

    @classmethod
    async def get(
        cls,
        id: int,
        db_resource: SQLDatabase,
    ) -> Self:
        """Filter table records by id."""
        async with db_resource.session() as session:
            query = select(cls).where(cls.id == id)
            return (await session.scalars(query)).one_or_none()


class TimestampDAO(BaseDAO):
    created_at: datetime = Field(
        default_factory=naive_utc_now,
        nullable=False,
    )

    updated_at: datetime = Field(
        default_factory=naive_utc_now,
        sa_column_kwargs={"onupdate": naive_utc_now},
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        return dt.replace(tzinfo=UTC).isoformat(timespec="seconds")
