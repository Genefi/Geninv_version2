from logging import Logger

from dependency_injector import containers, providers
from langchain_openai import ChatOpenAI

from core.environment import settings
from core.database import SQLDatabase
from core.logger import app_logger
from service.auth import AuthService
from service.users import UserService


class DependencyContainer(containers.DeclarativeContainer):
    # Dependency wiring
    wiring_config = containers.WiringConfiguration(
        packages=["service", "api", "model.dao"]
    )

    # Resources/Singletons
    logger: Logger = providers.Object(app_logger)
    pg_database = providers.Resource(
        SQLDatabase,
        db_config=settings.PG_DB_CONFIG,
        logger=logger,
    )

    # Factories
    llm_factory = providers.Factory(
        ChatOpenAI,
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_KEY,
    )

    user_service_factory = providers.Factory(
        UserService,
        pg_database=pg_database,
        logger=logger,
    )

    auth_service_factory = providers.Factory(AuthService, pg_database=pg_database)
