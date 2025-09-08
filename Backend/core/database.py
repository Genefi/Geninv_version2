from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator, Self
from dependency_injector.resources import AsyncResource
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.environment import SQLConfig
from core.logger import app_logger


class SQLDatabase(AsyncResource):
    async def init(
        self, db_config: SQLConfig, logger: logging.Logger = app_logger
    ) -> Self:
        db_url = URL.create(
            drivername=db_config.driver,
            username=db_config.username,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            query=db_config.additional_config,
        )
        self._logger = logger

        self._engine = create_async_engine(db_url, pool_recycle=3600)

        self._session_factory = async_sessionmaker(
            class_=AsyncSession, autocommit=False, autoflush=False, bind=self._engine
        )

        return self

    async def shutdown(self, _: None) -> None:
        self._logger.info("Shutting down...")
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        session: AsyncSession = self._session_factory()
        try:
            yield session
        except Exception as e:
            self._logger.error("An error occurred. Rolling back", exc_info=e)
            await session.rollback()
            raise
        finally:
            self._logger.debug("Closing session")
            await session.close()

    async def refresh_detached_instance(self, session: AsyncSession, instance: object):
        merged_instance = await session.merge(instance)
        await session.refresh(merged_instance)

        return merged_instance
