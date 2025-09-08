from uuid import UUID
from pydantic import BaseModel


class DynamicTableDTO(BaseModel):
    id: UUID
    owner_id: UUID
    business_id: UUID
    name: str
    description: str
    upsert_config: dict
    relationships: dict
    columns: dict


class DynamicTableRowDAO(BaseModel):
    id: UUID
    table_id: UUID
    data: dict
