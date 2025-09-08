from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class SQLConfig(BaseModel):
    driver: str
    username: str
    password: str
    host: str
    port: int
    database: str
    additional_config: dict[str, str] | None = {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    ALLOWED_ORIGINS: str = "http://localhost"

    ENVIRONMENT: str = "development"

    OPENAI_KEY: str

    OPENAI_MODEL: str = "gpt-4.1"

    ACCESS_TOKEN_SECRET: str
    REFRESH_TOKEN_SECRET: str

    INVITE_EXPIRES_AFTER_SECS: int = 12 * 60 * 60  # 12 hours
    PASSWORD_RESET_EXPIRES_AFTER_SECS: int = 1 * 60 * 60  # 1 hour
    ACCESS_TOKEN_EXPIRES_AFTER_SECS: int = 10 * 60  # 10 minutes
    REFRESH_TOKEN_EXPIRES_AFTER_SECS: int = 1 * 60 * 60  # 1 hour

    # Database config
    PG_DB_HOST: str = ""
    PG_DB_NAME: str = ""
    PG_DB_PASSWORD: str = ""
    PG_DB_USER: str = ""
    PG_DB_PORT: int = 5432

    @property
    def PG_DB_CONFIG(self) -> SQLConfig:
        sql_driver: str = "postgresql+asyncpg"
        additional_config: dict = {}

        return SQLConfig(
            driver=sql_driver,
            host=self.PG_DB_HOST,
            port=self.PG_DB_PORT,
            database=self.PG_DB_NAME,
            username=self.PG_DB_USER,
            password=self.PG_DB_PASSWORD,
            additional_config=additional_config,
        )

    @property
    def PARSED_ALLOWED_ORIGINS(self):
        return [x.strip() for x in self.ALLOWED_ORIGINS.split(",")]


settings = Settings()

__all__ = ["settings"]
