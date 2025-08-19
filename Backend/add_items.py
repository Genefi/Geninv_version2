"""
Complete, runnable module for background upsert into dynamic JSONB tables with linked-table updates.

Instructions:
- Configure your SQLAlchemy `engine` (Postgres recommended for JSONB features).
- Call `start_worker(your_engine)` once at application startup to initialize `SessionLocal` and start the background thread.
- Use `enqueue_rows(...)` or the convenience helpers to queue rows.

This file includes model definitions for `DynamicTable` and `DynamicTableRow`. If you already have your own models, adapt the imports or remove the model block and import your models instead.

Notes:
- Uses PostgreSQL `jsonb` functions for robust JSON field queries. If you use a different DB, adapt `jsonb_field_equals` accordingly.
- Mutations to the `data` JSON are tracked via `MutableDict`.
"""

from __future__ import annotations

import threading
import queue
import uuid
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.tools import tool
from Database_schema_creation import DynamicTable, DynamicTableRow, Base
from langchain.schema import HumanMessage

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from sqlalchemy import (
    Column,
    String,
    DateTime,
    create_engine,
    func,
    cast,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import sessionmaker
from dataclasses import dataclass

key =""# %% 1) define your tool

# %% 2) initialize LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=key)


# -------------------- Pydantic payload --------------------
class DynamicTableRowCreate(BaseModel):
    table_id: str
    data: Dict[str, Any]
    upsert_config: Optional[Dict[str, Any]] = None
    process_links: bool = True
    force_insert: bool = False



# Configure logger
logger = logging.getLogger("dynamictable_upsert")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# SQLAlchemy base + SessionLocal placeholder (set by start_worker)
Base = declarative_base()
SessionLocal = None  # type: ignore
engine = None  # type: ignore

# Local queue and worker control
row_queue: queue.Queue[DynamicTableRowCreate] = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# -------------------- Utility functions --------------------

def to_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal. Raises ValueError on failure."""
    if value is None:
        raise ValueError("value is None")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # Convert float via str to avoid binary floating issues
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            s = value.strip()
            return Decimal(s)
        except InvalidOperation as e:
            raise ValueError(f"cannot convert string to Decimal: {value}") from e
    raise ValueError(f"unsupported numeric type: {type(value)}")


def decimal_to_json_compatible(value: Decimal) -> Any:
    """Convert Decimal to int or float for JSON serializability.
    If Decimal is whole-number, return int; otherwise return float truncated to 12 decimal places.
    """
    if value == value.to_integral_value():
        return int(value)
    quantized = value.quantize(Decimal("1.000000000000"), rounding=ROUND_HALF_UP)
    return float(quantized)




# -------------------- DB helpers --------------------

def jsonb_field_equals(jsonb_column, key: str, value: Any):
    """Return a SQL expression that compares jsonb key text to value (Postgres).

    If you use a non-Postgres DB, replace this helper with an appropriate expression.
    """
    # Use postgres jsonb_extract_path_text to get the text of the key
    return func.jsonb_extract_path_text(jsonb_column, key) == str(value)


def get_table_structure(session, table_id: str) -> Optional[Dict[str, Any]]:
    t = session.query(DynamicTable).filter_by(id=table_id).first()
    if not t:
        return None
    return {
        "table": t,
        "schema": t.schema,
        "upsert_config": t.upsert_config,
        "relationships": t.relationships,
    }


# -------------------- Upsert Implementation --------------------

def upsert_dynamic_row(
    session,
    table_id: str,
    data: Dict[str, Any],
    upsert_config: Optional[Dict[str, Any]] = None,
    force_insert: bool = False,
):
    """Insert or update a DynamicTableRow according to upsert_config.

    Behavior summary:
    - If force_insert: always insert a new row.
    - If upsert_config is None: try to fetch from the DynamicTable; if still None -> insert.
    - If upsert_config specifies unique_fields: find existing row and apply update_fields rules.
    """

    # 1) Force insert shortcut
    if force_insert:
        new_row = DynamicTableRow(id=str(uuid.uuid4()), table_id=table_id, data=data, created_at=datetime.utcnow())
        session.add(new_row)
        session.commit()
        return new_row

    # 2) If no explicit config, attempt to load from table
    if not upsert_config:
        table_info = get_table_structure(session, table_id)
        if table_info:
            upsert_config = table_info.get("upsert_config")

    # 3) Still no config -> simple insert
    if not upsert_config:
        new_row = DynamicTableRow(id=str(uuid.uuid4()), table_id=table_id, data=data, created_at=datetime.utcnow())
        session.add(new_row)
        session.commit()
        return new_row

    # 4) Use upsert config
    unique_fields: List[str] = upsert_config.get("unique_fields", [])

    if not unique_fields:
        # No unique fields means insert new row
        new_row = DynamicTableRow(id=str(uuid.uuid4()), table_id=table_id, data=data, created_at=datetime.utcnow())
        session.add(new_row)
        session.commit()
        return new_row

    # Build query to locate existing row
    query = session.query(DynamicTableRow).filter(DynamicTableRow.table_id == table_id)
    for field in unique_fields:
        if field in data:
            query = query.filter(jsonb_field_equals(DynamicTableRow.data, field, data[field]))

    existing_row: Optional[DynamicTableRow] = query.first()

    if existing_row:
        update_fields = upsert_config.get("update_fields", {})
        # Apply updates
        for field, new_raw in data.items():
            if field in update_fields:
                op = update_fields[field]
                # Coerce current and new values to Decimal where appropriate for arithmetic ops
                if op in ("add", "subtract", "multiply", "divide"):
                    try:
                        cur = to_decimal(existing_row.data.get(field, 0))
                    except ValueError:
                        # If current value is non-numeric, replace rather than attempt arithmetic
                        logger.exception("Current value for %s is non-numeric: %r", field, existing_row.data.get(field))
                        existing_row.data[field] = new_raw
                        continue
                    try:
                        nxt = to_decimal(new_raw)
                    except ValueError:
                        logger.exception("New value for %s is non-numeric: %r", field, new_raw)
                        existing_row.data[field] = new_raw
                        continue

                    if op == "add":
                        result = cur + nxt
                    elif op == "subtract":
                        result = cur - nxt
                    elif op == "multiply":
                        result = cur * nxt
                    elif op == "divide":
                        if nxt == 0:
                            logger.warning("Division by zero when updating %s; preserving current value", field)
                            result = cur
                        else:
                            result = cur / nxt
                    existing_row.data[field] = decimal_to_json_compatible(result)

                elif op in ("max", "min"):
                    # Try numeric compare, fallback to direct compare
                    try:
                        cur = to_decimal(existing_row.data.get(field, 0))
                        nxt = to_decimal(new_raw)
                        result = cur if (op == "max" and cur >= nxt) or (op == "min" and cur <= nxt) else nxt
                        existing_row.data[field] = decimal_to_json_compatible(result)
                    except ValueError:
                        # non-numeric, fallback
                        existing_row.data[field] = max(existing_row.data.get(field), new_raw) if op == "max" else min(existing_row.data.get(field), new_raw)

                elif op == "replace":
                    existing_row.data[field] = new_raw
                else:
                    # Default: replace
                    existing_row.data[field] = new_raw
            else:
                # Not configured -> replace
                existing_row.data[field] = new_raw

        # Update timestamp
        timestamp_field = "updated_at" if "updated_at" in existing_row.data else "last_updated"
        existing_row.data[timestamp_field] = datetime.utcnow().isoformat()

        session.merge(existing_row)
        session.commit()
        return existing_row
    else:
        if upsert_config.get("create_if_missing", True):
            new_row = DynamicTableRow(id=str(uuid.uuid4()), table_id=table_id, data=data, created_at=datetime.utcnow())
            session.add(new_row)
            session.commit()
            return new_row
        else:
            # No action configured
            logger.info("No matching row found and create_if_missing=False for table %s", table_id)
            return None


# -------------------- TableLinker --------------------
@dataclass
class TableLinker:
    session: Any
    commit_on_update: bool = True
    logger: logging.Logger = logger

    def process_linked_updates(self, source_table_id: str, changed_data: Dict[str, Any]) -> None:
        try:
            table_info = get_table_structure(self.session, source_table_id)
        except Exception:
            self.logger.exception("Failed to load table structure for %s", source_table_id)
            return

        if not table_info or not table_info.get("relationships"):
            return

        relationships = table_info["relationships"]
        linked_tables = relationships.get("linked_tables", []) if isinstance(relationships, dict) else []

        for link in linked_tables:
            try:
                self.update_linked_table(link, changed_data, source_table_id)
            except Exception:
                self.logger.exception("Failed to update linked table %s", link.get("table_name", "unknown"))

    def update_linked_table(self, link_config: Dict[str, Any], changed_data: Dict[str, Any], source_table_id: str) -> None:
        target_table_name = link_config.get("table_name")
        if not target_table_name:
            self.logger.warning("Link config missing table_name: %s", link_config)
            return

        target_table = self.session.query(DynamicTable).filter_by(name=target_table_name).first()
        if not target_table:
            self.logger.warning("Target table '%s' not found", target_table_name)
            return

        key_field = link_config.get("key_field")
        source_field = link_config.get("source_field")
        target_field = link_config.get("target_field")
        operation = link_config.get("operation", "subtract")

        if not all([key_field, source_field, target_field]):
            self.logger.warning("Incomplete link configuration for %s: %s", target_table_name, link_config)
            return

        key_value = changed_data.get(key_field)
        raw_change_amount = changed_data.get(source_field)

        if key_value is None or raw_change_amount is None:
            self.logger.debug("No key or change amount present for key_field=%s, source_field=%s", key_field, source_field)
            return

        try:
            change_amount = to_decimal(raw_change_amount)
        except ValueError:
            self.logger.exception("Non-numeric change amount for %s: %r", source_field, raw_change_amount)
            return

        # Optionally use a nested transaction if caller wants isolated commits
        nested = False
        if self.commit_on_update:
            try:
                with self.session.begin_nested():
                    self._apply_link_update(link_config, target_table, key_field, key_value, target_field, operation, change_amount)
                return
            except Exception:
                self.logger.exception("Nested transaction failed for link %s", link_config)
                return
        else:
            # Use outer transaction (caller responsible for commit)
            try:
                self._apply_link_update(link_config, target_table, key_field, key_value, target_field, operation, change_amount)
            except Exception:
                self.logger.exception("Error while applying link update (outer tx) for %s", link_config)
            return

    def _apply_link_update(self, link_config, target_table, key_field, key_value, target_field, operation, change_amount: Decimal):
        # Lock and select the target row
        query = self.session.query(DynamicTableRow).with_for_update().filter(
            DynamicTableRow.table_id == target_table.id,
            jsonb_field_equals(DynamicTableRow.data, key_field, key_value),
        )
        target_row = query.first()

        if target_row is None:
            if link_config.get("create_if_missing", False):
                if operation == "subtract":
                    initial_value = -change_amount
                elif operation == "add":
                    initial_value = change_amount
                else:
                    initial_value = change_amount

                new_row_data = {key_field: key_value, target_field: decimal_to_json_compatible(initial_value)}
                new_row = DynamicTableRow(id=str(uuid.uuid4()), table_id=target_table.id, data=new_row_data, created_at=datetime.utcnow())
                self.session.add(new_row)
                self.logger.info("Created new row in %s for %s=%s", target_table.name, key_field, key_value)
            else:
                self.logger.warning("Target row with %s='%s' not found in %s", key_field, key_value, target_table.name)
            return

        # Compute using Decimal
        current_raw = target_row.data.get(target_field, 0)
        try:
            current_value = to_decimal(current_raw)
        except ValueError:
            self.logger.exception("Stored current value for %s.%s is not numeric: %r", target_table.name, target_field, current_raw)
            return

        if operation == "subtract":
            new_value = current_value - change_amount
        elif operation == "add":
            new_value = current_value + change_amount
        elif operation == "multiply":
            new_value = current_value * change_amount
        elif operation == "divide":
            if change_amount == 0:
                self.logger.warning("Division by zero for link %s; preserving current value", target_table.name)
                new_value = current_value
            else:
                new_value = current_value / change_amount
        elif operation == "replace":
            new_value = change_amount
        else:
            self.logger.warning("Unknown operation '%s' for link; defaulting to subtract", operation)
            new_value = current_value - change_amount

        # Apply min/max constraints
        min_value_raw = link_config.get("min_value")
        max_value_raw = link_config.get("max_value")
        if min_value_raw is not None:
            try:
                min_value = to_decimal(min_value_raw)
                if new_value < min_value:
                    new_value = min_value
            except ValueError:
                self.logger.exception("Invalid min_value in config: %r", min_value_raw)
        if max_value_raw is not None:
            try:
                max_value = to_decimal(max_value_raw)
                if new_value > max_value:
                    new_value = max_value
            except ValueError:
                self.logger.exception("Invalid max_value in config: %r", max_value_raw)

        target_row.data[target_field] = decimal_to_json_compatible(new_value)
        if link_config.get("update_timestamp", True):
            timestamp_field = link_config.get("timestamp_field", "last_updated")
            target_row.data[timestamp_field] = datetime.utcnow().isoformat()

        # Flush to ensure DB sees the change (caller will commit/rollback)
        self.session.flush()


# -------------------- Worker --------------------

def _worker(SessionFactory):
    logger.info("Worker thread started")
    while not _stop_event.is_set():
        try:
            row_in: DynamicTableRowCreate = row_queue.get(timeout=0.5)
        except Exception:
            continue

        try:
            with SessionFactory() as db:
                # Single transaction for upsert + linked updates when possible
                with db.begin():
                    upsert_dynamic_row(db, row_in.table_id, row_in.data, row_in.upsert_config, force_insert=row_in.force_insert)

                    if row_in.process_links:
                        linker = TableLinker(session=db, commit_on_update=False)
                        linker.process_linked_updates(row_in.table_id, row_in.data)

        except Exception as e:
            logger.exception("Failed to process row %s: %s", row_in, e)
        finally:
            row_queue.task_done()

    logger.info("Worker thread stopping")


# -------------------- Public API --------------------

def start_worker(_engine, *, worker_count: int = 1):
    """Initialize SessionLocal and start the background worker thread(s).

    Call this once at application startup.
    """
    global SessionLocal, engine, _worker_thread
    engine = _engine
    SessionLocal = sessionmaker(bind=engine)

    if _worker_thread is not None and _worker_thread.is_alive():
        logger.info("Worker already running")
        return

    _stop_event.clear()
    _worker_thread = threading.Thread(target=_worker, args=(SessionLocal,), daemon=True)
    _worker_thread.start()
    logger.info("Started background worker")


def stop_worker():
    _stop_event.set()
    logger.info("Stop requested for worker thread")


@tool
def enqueue_rows(rows: Union[List[Dict[str, Any]], Dict[str, Any]], upsert_config: Optional[Dict[str, Any]] = None, force_insert: bool = False, process_links: bool = True) -> Dict[str, Any]:
    """Tool wrapper for enqueuing rows for an agent (LangChain/LangGraph).

    Accepts either a single row dict or a list of row dicts. Each row dict must contain:
      - table_id (str)
      - data (dict)

    Returns: {"enqueued": <number>} or {"error": "..."} on failure.
    """
    # Normalize payload to a list
    try:
        payload = rows if isinstance(rows, list) else [rows]
    except Exception as e:
        return {"error": f"invalid rows payload: {e}"}

    count = 0
    for raw in payload:
        # Basic validation so agent mistakes don't crash the worker
        if not isinstance(raw, dict) or "table_id" not in raw or "data" not in raw:
            logger.warning("Skipping invalid row payload: %s", raw)
            continue

        row_data = {
            "table_id": raw["table_id"],
            "data": raw["data"],
            "upsert_config": None if force_insert else upsert_config,
            "process_links": process_links,
            "force_insert": force_insert,
        }

        try:
            row_in = DynamicTableRowCreate(**row_data)
            row_queue.put(row_in)
            count += 1
        except Exception as e:
            logger.exception("Invalid row payload %s: %s", raw, e)
            continue

    return {"enqueued": count}


# Convenience wrappers
def enqueue_sales_transaction(table_id: str, item_name: str, amount: float, quantity: int, process_links: bool = True, **other_fields):
    data = {"item": item_name, "amount": amount, "quantity": quantity, "transaction_date": datetime.utcnow().isoformat(), **other_fields}
    return enqueue_rows(rows={"table_id": table_id, "data": data}, force_insert=True, process_links=process_links)


def enqueue_inventory_item(table_id: str, item_name: str, quantity: int, override_upsert: bool = False):
    if override_upsert:
        return enqueue_rows(rows={"table_id": table_id, "data": {"item": item_name, "quantity": quantity}}, force_insert=True, process_links=False)
    return enqueue_rows(rows={"table_id": table_id, "data": {"item": item_name, "quantity": quantity}})


def enqueue_product_update(table_id: str, product_name: str, price: float, **other_fields):
    data = {"name": product_name, "price": price, **other_fields}
    return enqueue_rows(rows={"table_id": table_id, "data": data})


# -------------------- Example initialization (commented) --------------------
#
# Example (Postgres):
# from sqlalchemy import create_engine
# engine = create_engine("postgresql+psycopg2://user:pass@host/dbname", echo=False, future=True)
# Base.metadata.create_all(engine)  # create tables if not present
# start_worker(engine)
#
# Then use: enqueue_sales_transaction(...)
#
# Call stop_worker() at shutdown if desired.

class State(TypedDict):
    messages: Annotated[List, add_messages]


# ---------- Enhanced Chatbot Node ----------
def chatbot(state: State):
    """Chatbot that uses native tool calling for both regular queries and dashboards"""
    messages = state["messages"]
    
    # Enhanced system prompt that guides the LLM to use tools appropriately
    system_prompt = "You are a helpful assistant that helps in creating dynamic tables for busineeses, you can ask first about the person's business and then suggest column names to the user..also ask the user if they would like to modify or add to the column names...do all of this before using the tool you have acccess to to create the dynamic table.try and make the interaction as short as possible by going straight to the point. The tool has several parameters which are ...\
Name: name of table, description: what the table is used for or supposed to store, columns: list of dictionares which contain the column names and data type in the example shown below. [{'name': 'username', 'type': 'TEXT'},{'name': 'age', 'type': 'INTEGER'}]"



    # Add system message if not present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Bind tools to LLM
    tools = [enqueue_rows]
    llm_with_tools = llm.bind_tools(tools)
    
    # Get LLM response with tool calls
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def create_react_graph(llm_instance):
    """Create and return the compiled ReAct graph with proper tool calling"""
    global llm
    llm = llm_instance
    
    # Initialize the graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    
    # Tool node handles both SQL and dashboard tools
    tools = [enqueue_rows]
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    
    # Tool result processor
    
    # Set entry point
    graph_builder.set_entry_point("chatbot")
    
    # Use LangGraph's built-in tools_condition for routing
    # conditional routing
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
        
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    return graph_builder.compile(checkpointer=memory)


def run_conversation_item(user_message: str, thread_id: str = "default"):
    """Helper function to run a single conversation turn"""
    graph = create_react_graph(llm)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get current state or create new
    try:
        current_state = graph.get_state(config).values
        messages = current_state.get("messages", [])
    except:
        messages = []
    
    # Add user message
    messages.append(HumanMessage(content=user_message))
    
    # Run the graph
    result = graph.invoke({
        "messages": messages,
    }, config)
    
    return result