from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from datetime import datetime
from langchain_core.messages import SystemMessage
import uuid

# %% imports
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain.schema import HumanMessage
from typing import Annotated, TypedDict
from  Database_schema_creation import *

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

key =""
# %% 1) define your tool

# %% 2) initialize LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=key)


class TableColumn(BaseModel):
    name: str
    type: str

class DynamicTableCreate(BaseModel):
    name: str
    description:str
    columns: List[TableColumn]
    upsert_config: Optional[Dict[str, Any]] = None  # NEW: Upsert behavior rules
    relationships: Optional[Dict[str, Any]] = None  # NEW: Relationships with other tables

        
db_uri =  "postgresql+psycopg2://macbookair:@localhost:5432/Geninv"

engine = create_engine(db_uri)
Base.metadata.create_all(engine)
user_id = #user_id gotten from somewhere
# Tool function







_ALLOWED_OPS = {"add", "subtract", "replace", "max", "min", "multiply", "divide"}
def _validate_upsert_config(cfg) -> Optional[str]:
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        return "upsert_config must be an object (dict) or null."

    unique = cfg.get("unique_fields")
    if not isinstance(unique, list) or not unique or not all(isinstance(x, str) for x in unique):
        return "upsert_config.unique_fields must be a non-empty list of strings."

    uf = cfg.get("update_fields", {})
    if not isinstance(uf, dict):
        return "upsert_config.update_fields must be an object mapping field->operation."
    for field, op in uf.items():
        if not isinstance(field, str):
            return "upsert_config.update_fields keys must be strings (field names)."
        if not isinstance(op, str) or op not in _ALLOWED_OPS:
            return f"Invalid op for field '{field}': '{op}'. Allowed: {_ALLOWED_OPS}."

    if "create_if_missing" in cfg and not isinstance(cfg["create_if_missing"], bool):
        return "upsert_config.create_if_missing must be a boolean."

    if "timestamp_field" in cfg and not isinstance(cfg["timestamp_field"], str):
        return "upsert_config.timestamp_field must be a string."

    return None


def _validate_relationships(rel) -> Optional[str]:
    if rel is None:
        return None
    if not isinstance(rel, dict):
        return "relationships must be an object (dict) or null."

    linked = rel.get("linked_tables")
    if linked is None:
        return "relationships.linked_tables is required (can be an empty list)."
    if not isinstance(linked, list):
        return "relationships.linked_tables must be a list."

    for i, link in enumerate(linked):
        if not isinstance(link, dict):
            return f"linked_tables[{i}] must be an object."

        required_keys = ["table_name", "key_field", "source_field", "target_field"]
        for k in required_keys:
            if k not in link or not isinstance(link[k], str) or not link[k]:
                return f"linked_tables[{i}].{k} is required and must be a non-empty string."

        op = link.get("operation", "subtract")
        if not isinstance(op, str) or op not in _ALLOWED_OPS:
            return f"linked_tables[{i}].operation must be one of {_ALLOWED_OPS}."

        if "create_if_missing" in link and not isinstance(link["create_if_missing"], bool):
            return f"linked_tables[{i}].create_if_missing must be boolean."

        for bound in ("min_value", "max_value"):
            if bound in link and link[bound] is not None:
                try:
                    to_decimal(link[bound])
                except Exception:
                    return f"linked_tables[{i}].{bound} must be numeric or null."

        if "update_timestamp" in link and not isinstance(link["update_timestamp"], bool):
            return f"linked_tables[{i}].update_timestamp must be boolean."
        if "timestamp_field" in link and link.get("timestamp_field") is not None and not isinstance(link["timestamp_field"], str):
            return f"linked_tables[{i}].timestamp_field must be a string."

    return None

@tool
def create_dynamic_table(table_in: DynamicTableCreate) -> Dict:
    """
    Creates a dynamic SQL table in PostgreSQL.

    Input format:
    - `name` (str): Name of the table to create.
    - `description` (str): A description of the table.
    - `columns` (List[Dict]): A list of column definitions.
      Each item in the list should be a dictionary with:
        - `name` (str): The name of the column.
        - `type` (str): The SQL type for the column (e.g., TEXT, INTEGER, BOOLEAN).

    Example:
    {
        "name": "users",
        "description": "Stores user info",
        "columns": [
            {"name": "username", "type": "TEXT"},
            {"name": "age", "type": "INTEGER"}
        ]
        "upsert_config": {
            "unique_fields": ["username"],
            "update_fields": {"age": "replace"},
            "create_if_missing": true,
            "timestamp_field": "last_updated"
        },
        "relationships": {
            "linked_tables": [
                {
                    "table_name": "orders",
                    "key_field": "user_id",
                    "source_field": "order_count",
                    "target_field": "total_orders",
                    "operation": "add",
                    "create_if_missing": false,
                    "min_value": 0,
                    "max_value": null,
                    "update_timestamp": true,
                    "timestamp_field": "last_updated"
                }
            ]
        }
    }

    Returns a dictionary with table creation status or error.
    """
    print(f"[TOOL] table creation tool called")

    if not getattr(table_in, "name", None):
        return {"error": "table_in.name is required."}
    cols = []
    if getattr(table_in, "columns", None) is None:
        return {"error": "table_in.columns is required and must be a list of column definitions."}
    for c in table_in.columns:
        if hasattr(c, "model_dump"):
            cols.append(c.model_dump())
        elif isinstance(c, dict):
            cols.append(c)
        else:
            try:
                cols.append(dict(c))
            except Exception:
                return {"error": "Each column must be a dict or pydantic model with model_dump()."}
    upsert_cfg = getattr(table_in, "upsert_config", None)
    err = _validate_upsert_config(upsert_cfg)
    if err:
        return {"error": f"Invalid upsert_config: {err}"}


    rel_cfg = getattr(table_in, "relationships", None)
    err = _validate_relationships(rel_cfg)
    if err:
        return {"error": f"Invalid relationships: {err}"}
    
    with Session(engine) as db:
        table = DynamicTable(
            id=str(uuid.uuid4()),
            name=table_in.name,
            user_id = user_id,
            description = table_in.description,
            columns=[col.model_dump() for col in table_in.columns],
            created_at=datetime.utcnow()
        )
        try:
            db.add(table)
            print(f"Inserting table: {table.name} with columns: {table.columns}")
            db.commit()
            db.refresh(table)
            print("successful!")
            return {"table": table.name, "table_columns":table.columns, "status": "created or exists"}
            
        except Exception as e:
            return {"error": str(e)}
        
class State(TypedDict):
    messages: Annotated[List, add_messages]

current_number_of_tables = 0
# ---------- Enhanced Chatbot Node ----------
def chatbot(state: State):
    """Chatbot that uses native tool calling for both regular queries and dashboards"""
    messages = state["messages"]
    
    # Enhanced system prompt that guides the LLM to use tools appropriately
    system_prompt = f"""
You are a helpful assistant that creates dynamic tables for businesses. Workflow (be brief and go straight to the point):
1) Ask one short question about the user's business/use-case to understand the table goal.
2) Propose a concise list of column names + types (offer 3–8 sensible columns). Ask once if they want to modify/add/remove columns.
3) After confirmation, call the create-table tool (you have access) with the following parameters:
   - name: string
   - description: string (what the table stores)
   - columns: list of dicts e.g. [{{'name':'username','type':'TEXT'}}, {{'name':'age','type':'INTEGER'}}]
   - upsert_config: (optional) object controlling upsert behavior
   - relationships: (optional) object describing linked-table updates

   The current number of tables is {current_number_of_tables}.
   The table_id, table columns and the table_description for the tables that the user has access to is shown below
        {jew_id}/{jew_descript}/{jew_columns}
        {wig_id}/{wig_des}/{wig_column}
   if the current number of tables is greater than 1 then ask to include relationships.

Keep interactions short and only ask follow-ups necessary to build the table schema.

upsert_config (example and rules):
- Purpose: control how insert/update behaves for this table.
for example if we have an inventory we might want to add quantity to existing rows..just use your discretion and choose the rule that makes sense...you can also confrim from the user if you are unsure
- Shape (JSON):
  {{
    "unique_fields": ["item"],                  // fields that identify a unique logical row
    "update_fields": {{                          // per-field rule: add, subtract, replace, max, min, multiply, divide
      "quantity": "add",
      "price": "replace"
    }},
    "create_if_missing": true,                  // create row if none found
    "timestamp_field": "last_updated"           // optional timestamp field to set on updates
  }}
- If you want every row to be an independent record (no upsert), set upsert_config to null.

relationships (example and rules):
- Purpose: trigger updates in other tables when rows in this table change.
- Shape (JSON):
  {{
    "linked_tables": [
      {{
        "table_name": "inventory",
        "key_field": "item",              // field present in both tables to match rows
        "source_field": "quantity",       // field on this (source) table that carries the delta
        "target_field": "quantity",       // field to update on target table
        "operation": "subtract",          // add | subtract | multiply | divide | replace
        "create_if_missing": false,
        "min_value": 0,
        "max_value": null,
        "update_timestamp": true,
        "timestamp_field": "last_updated"
      }}
    ]
  }}

When preparing the tool call, supply the confirmed name, description, columns list, and optional upsert_config and/or relationships. Do not call the tool until the user confirms the schema.
"""

    # Add system message if not present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Bind tools to LLM
    tools = [create_dynamic_table]
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
    tools = [create_dynamic_table]
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


def run_conversation_table(user_message: str, thread_id: str = "default"):
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