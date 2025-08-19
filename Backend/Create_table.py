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

key ="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"

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
        
db_uri =  "postgresql+psycopg2://macbookair:@localhost:5432/Geninv"

engine = create_engine(db_uri)
Base.metadata.create_all(engine)
user_id = #user_id gotten from somewhere
# Tool function
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
    }

    Returns a dictionary with table creation status or error.
    """
    print(f"[TOOL] table creation tool called")
    with Session(engine) as db:
        table = DynamicTable(
            id=str(uuid.uuid4()),
            name=table_in.name,
            user_id = user_id,
            description = table_in.description,
            schema=[col.model_dump() for col in table_in.columns],
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


def run_conversation(user_message: str, thread_id: str = "default"):
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