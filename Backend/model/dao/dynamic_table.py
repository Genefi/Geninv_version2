from typing import Self
from uuid import UUID

from sqlalchemy import ScalarResult
from sqlmodel import Column, Field, select
from sqlalchemy.dialects.postgresql import JSONB
from core.database import SQLDatabase
from model.dao.base import TimestampDAO, UuidDAO
from model.dto.dynamic_table import DynamicTableDTO


class DynamicTableDAO(UuidDAO, TimestampDAO, table=True):
    # model config
    __tablename__ = "dynamic_tables"
    __dto_class__ = DynamicTableDTO

    owner_id: UUID = Field(foreign_key="users.id")
    business_id: UUID = Field(foreign_key="businesses.id")
    name: str
    description: str
    upsert_config: dict = Field(sa_column=Column(JSONB, nullable=True))
    relationships: dict = Field(sa_column=Column(JSONB, nullable=True))
    columns: dict = Field(sa_column=Column(JSONB, nullable=True))

    @classmethod
    async def filter(
        cls,
        *,
        db_resource: SQLDatabase,
        id: UUID | None = None,
        owner_id: UUID | None = None,
        name: str = None,
    ) -> ScalarResult[Self]:
        """Filter dynamic tables by id, owner and name."""
        async with db_resource.session() as session:
            query = select(DynamicTableDAO)
            if id is not None:
                query = query.where(DynamicTableDAO.id == id)
            if owner_id is not None:
                query = query.where(DynamicTableDAO.owner_id == owner_id)
            if name is not None:
                query = query.where(DynamicTableDAO.name == name)
            return await session.scalars(query)


class DynamicTableRowDAO(UuidDAO, TimestampDAO, table=True):
    __tablename__ = "dynamic_table_rows"

    table_id: UUID = Field(default=None, foreign_key="dynamic_tables.id")
    data: dict = Field(sa_column=Column(JSONB, nullable=False))

    @classmethod
    async def filter(
        cls,
        *,
        db_resource: SQLDatabase,
        id: UUID | None = None,
        table_id: UUID | None = None,
    ) -> ScalarResult[Self]:
        """Filter dynamic tables entries by id, and table_id."""
        async with db_resource.session() as session:
            query = select(DynamicTableRowDAO)
            if id is not None:
                query = query.where(DynamicTableRowDAO.id == id)
            if table_id is not None:
                query = query.where(DynamicTableRowDAO.table_id == table_id)
            return await session.scalars(query)
