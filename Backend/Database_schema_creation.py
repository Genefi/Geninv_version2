from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
import datetime
import uuid

class DynamicTable(Base):
    __tablename__ = 'dynamictable'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, primary_key=True, nullable=False)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, unique=True, nullable=False)
    upsert_config = Column(JSONB)    # NEW: Upsert behavior rules
    relationships = Column(JSONB)
    columns = Column(JSONB, nullable=False)  # [{name, type}]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    

class DynamicTableRow(Base):
    __tablename__ = 'dynamictablerow'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    table_id = Column(String, ForeignKey('dynamictable.id'))
    data = Column(JSONB, nullable=False)  # {column_name: value}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)