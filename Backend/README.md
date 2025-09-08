# GenInv (Backend)
## Dependencies
This is project uses the following dependencies:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Dependency Injector](https://python-dependency-injector.ets-labs.org/)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [Pydntic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [SQLModel](https://sqlmodel.tiangolo.com/) (on top of [SQLAlchemy](https://www.sqlalchemy.org/))
- [Alembic](https://alembic.sqlalchemy.org/en/latest/)
- [PyJWT](https://pyjwt.readthedocs.io/en/stable/)
- [Passlib](https://passlib.readthedocs.io/en/stable/) (with bcrypt)
- [AsyncPG](https://magicstack.github.io/asyncpg/current/)
- LangChain
- LangChain (OpenAI)
- LangGraph

## Dependency Management
Dependency management is handled by [Poetry](https://python-poetry.org/).
- Activate Shell (if not done automatically) using `poetry shell`
- Install dependencies using `poetry install`
- Add dependencies using `poetry add dependency-1 dependency-2 ... dependency-n`

## Database Migrations
Database migrations are handled by [Alembic](https://alembic.sqlalchemy.org/en/latest/).
- Create migration from models using `alembic revision -m "Migration message" --autogenerate`
- Create empty migration using `alembic revision -m "Migration message"`
- Apply existing migrations using `alembic upgrade head`
- Un-apply previous migration using `alembic downgrade -1`

## API Server
The project is a [FastAPI](https://fastapi.tiangolo.com/) project
- Start API Server using `uvicorn main:app`
- To auto-reload server (useful during development) use `uvicorn main:app --reload`

#### Docs
Swagger docs are available at `http://[host]:[port]/docs`

