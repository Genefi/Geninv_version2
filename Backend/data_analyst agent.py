import json
from typing import TypedDict, List, Dict, Annotated, Literal, Optional
import psycopg
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


key =""# %% 1) define your tool

# %% 2) initialize LLM
llm = ChatOpenAI(model="gpt-4.1", api_key=key)

jew_id= df1[0][2]
jew_columns =df1[3][2]
jew_descript = df1[2][2]

wig_id = df1[0][6]
wig_column = df1[3][6]
wig_des = df1[2][6]


from datetime import date
today_str = date.today().strftime("%Y-%m-%d")
print(today_str)


# Global state store for chart data (accessed by tools)
_chart_data_store = []



# ---------- Enhanced SQL Tool with State-Only Storage ----------
@tool
def run_sql(
    query: str, 
    purpose: str = "answer", 
    chart_type: Optional[str] = None,
    chart_title: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None
) -> Dict:
    """
    Runs SQL query against the dynamictablerow table and optionally stores chart data in state.
    
    Args:
        query: SQL query string
        purpose: "answer" for regular queries, "dashboard" for visualization data
        chart_type: Type of chart if purpose is "dashboard" (bar, line, pie, scatter)
        chart_title: Title for the chart if creating visualization
        x_column: Column name for X-axis (optional, will auto-detect if not provided)
        y_column: Column name for Y-axis (optional, will auto-detect if not provided)
        
    Returns:
        For "answer": Dictionary with full query results for LLM analysis
        For "dashboard": Dictionary with summary only - chart data stored in global state
        
    Examples:
        # Regular query
        run_sql("SELECT COUNT(*) as total FROM dynamictablerow WHERE table_id = '123'")
        
        # Dashboard query  
        run_sql(
            query="SELECT data->>'category' as category, COUNT(*) as count FROM dynamictablerow GROUP BY data->>'category'",
            purpose="dashboard",
            chart_type="bar",
            chart_title="Items by Category"
        )
    """
    global _chart_data_store
    
    try:
        conn = psycopg.connect(
            dbname="Geninv",
            user="macbookair", 
            password="",
            host="localhost",
            port="5432"
        )
        
        cur = conn.cursor()
        cur.execute(query)
        
        # Get column names and data
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        
        # Convert to list of dictionaries
        data = [dict(zip(cols, row)) for row in rows] if cols else []
        
        cur.close()
        conn.close()
        
        # For dashboard queries: store data in global state, return summary only
        if purpose == "dashboard" and data and chart_type:
            # Auto-detect columns if not provided
            if not x_column and len(cols) > 0:
                x_column = cols[0]
            if not y_column and len(cols) > 1:
                y_column = cols[1]
            
            # Create full chart config and store in global state
            chart_config = {
                "type": "chart",
                "chart_type": chart_type,
                "title": chart_title or f"{chart_type.title()} Chart",
                "data": data,  # Full data stored here
                "x_column": x_column,
                "y_column": y_column,
                "x_label": x_column.replace('_', ' ').title() if x_column else "X",
                "y_label": y_column.replace('_', ' ').title() if y_column else "Y",
                "source": "run_sql",
                "timestamp": "2025-08-16"
            }
            
            # Store in global state for later retrieval
            _chart_data_store.append(chart_config)
            
            # Return only summary to LLM - NO LARGE DATA
            return {
                "success": True,
                "purpose": "dashboard",
                "chart_created": True,
                "chart_type": chart_type,
                "chart_title": chart_title or f"{chart_type.title()} Chart",
                "data_points": len(data),
                "columns_used": [c for c in [x_column, y_column] if c],
                "summary": f"✅ Created {chart_type} chart '{chart_title or 'Chart'}' with {len(data)} data points using columns: {', '.join([c for c in [x_column, y_column] if c])}"
            }
        
        # For regular answer queries: return full data as LLM needs it for analysis  
        else:
            return {
                "success": True,
                "data": data,
                "row_count": len(data),
                "columns": cols,
                "purpose": "answer",
                "summary": f"Query executed successfully. Returned {len(data)} rows with columns: {', '.join(cols) if cols else 'none'}"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": [],
            "row_count": 0,
            "columns": [],
            "summary": f"❌ Query failed: {str(e)}"
        }

# ---------- Dashboard Creation Tool with State-Only Storage ----------
@tool 
def create_dashboard(queries: List[Dict]) -> Dict:
    """
    Execute multiple SQL queries for dashboard creation and store all chart data in state.
    
    Args:
        queries: List of query configurations, each containing:
            - query: SQL string
            - chart_type: bar, line, pie, scatter
            - title: Chart title
            - x_column: (optional) X-axis column
            - y_column: (optional) Y-axis column
    
    Returns:
        Summary only - all chart data stored in global state
        
    Example:
        create_dashboard([
            {
                "query": "SELECT data->>'category' as category, COUNT(*) as count FROM dynamictablerow GROUP BY data->>'category'",
                "chart_type": "bar",
                "title": "Items by Category"
            },
            {
                "query": "SELECT DATE(created_at) as date, COUNT(*) as daily_count FROM dynamictablerow GROUP BY DATE(created_at)",
                "chart_type": "line", 
                "title": "Daily Activity"
            }
        ])
    """
    global _chart_data_store
    
    chart_summaries = []
    total_data_points = 0
    successful_charts = 0
    
    for i, query_config in enumerate(queries):
        result = run_sql(
            query=query_config["query"],
            purpose="dashboard",
            chart_type=query_config["chart_type"],
            chart_title=query_config["title"],
            x_column=query_config.get("x_column"),
            y_column=query_config.get("y_column")
        )
        
        if result.get("success") and result.get("chart_created"):
            successful_charts += 1
            chart_summaries.append(f"Chart {i+1}: {result.get('summary', 'Chart created')}")
            total_data_points += result.get("data_points", 0)
        else:
            chart_summaries.append(f"Chart {i+1}: ❌ Failed - {result.get('error', 'Unknown error')}")
    
    # Return only summary to LLM - all chart data already stored in global state by run_sql calls
    return {
        "success": True,
        "dashboard_created": True,
        "total_charts_requested": len(queries),
        "successful_charts": successful_charts,
        "total_data_points": total_data_points,
        "chart_summaries": chart_summaries,
        "summary": f"✅ Dashboard created! {successful_charts}/{len(queries)} charts successful with {total_data_points} total data points"
    }

# ---------- State Schema ----------
class State(TypedDict):
    messages: Annotated[List, add_messages]
    graph_list: List[Dict]

# ---------- Enhanced Chatbot Node ----------
def chatbot(state: State):
    """Chatbot that uses native tool calling for both regular queries and dashboards"""
    messages = state["messages"]
    
    # Enhanced system prompt that guides the LLM to use tools appropriately
    system_prompt = f"""You are a data analyst assistant with access to a PostgreSQL database containing a 'dynamictablerow' table you answer questions ans perform your activities by running  jsonb postgressql queries

You have access to these tools:
1. run_sql: For single queries - use purpose="answer" for regular questions, purpose="dashboard" for visualizations
2. create_dashboard: For multiple related charts in one dashboard

The table_id, table columns and the table_description for the tables that the user has access to is shown below
{jew_id}/{jew_descript}/{jew_columns}
{wig_id}/{wig_des}/{wig_column}
note that  today's date {today_str} if the date is not provided by the user\n
note that when a question is asked try to get the unique columns for all categorical or str variable when appropriate so you can write your query effectively
the above means you might need to call the run sql tool you have access to twice in order to get the full context to answer the question

Guidelines:
- For simple questions, use run_sql with purpose="answer"
- For single charts/graphs, use run_sql with purpose="dashboard" and specify chart_type
- For dashboards with multiple charts, use create_dashboard with a list of queries
- Available chart types: bar, line, pie, scatter
- Always provide helpful analysis of the results

The database contains dynamic table rows with JSON data in a 'data' column. Use JSON operators like data->>'field_name' to extract values.

Example table_id: '97dad001-7c7d-433f-97e6-5fa509d2b511'"""

    # Add system message if not present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + messages

    # Bind tools to LLM
    tools = [run_sql, create_dashboard]
    llm_with_tools = llm.bind_tools(tools)
    
    # Get LLM response with tool calls
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# ---------- Tool Result Processor - Retrieves from Global State ----------
def process_tool_results(state: State):
    """Retrieve chart data from global state and update graph_list"""
    global _chart_data_store
    
    # Get current graph_list
    graph_list = list(state.get("graph_list", []))
    
    # Move all charts from global store to state
    if _chart_data_store:
        graph_list.extend(_chart_data_store)
        _chart_data_store.clear()  # Clear global store after moving to state
    
    return {"graph_list": graph_list}

# ---------- Build the Graph ----------
def create_react_graph(llm_instance):
    """Create and return the compiled ReAct graph with proper tool calling"""
    global llm
    llm = llm_instance
    
    # Initialize the graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    
    # Tool node handles both SQL and dashboard tools
    tools = [run_sql, create_dashboard]
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    
    # Tool result processor
    graph_builder.add_node("process_results", process_tool_results)
    
    # Set entry point
    graph_builder.set_entry_point("chatbot")
    
    # Use LangGraph's built-in tools_condition for routing
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,  # This automatically detects tool calls
        {
            "tools": "tools",
            "__end__": "__end__"
        }
    )
    
    # After tools execution, process results then continue conversation
    graph_builder.add_edge("tools", "process_results") 
    graph_builder.add_edge("process_results", "chatbot")
    
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    return graph_builder.compile(checkpointer=memory)

# ---------- Dashboard Export Helper ----------
def export_dashboard_config(state: State, title: str = "Data Dashboard") -> Dict:
    """Export the graph_list as a complete dashboard configuration"""
    charts = state.get("graph_list", [])
    
    return {
        "dashboard_config": {
            "title": title,
            "created_at": "2025-08-16",
            "layout": "responsive",
            "charts": charts,
            "total_charts": len(charts),
            "chart_types": list(set(chart.get("chart_type", "unknown") for chart in charts))
        },
        "ready_for_plotting": True,
        "token_efficient": True  # Large datasets never passed through LLM conversation
    }

# ---------- Helper Function to Clear Chart Store ----------
def clear_chart_store():
    """Clear the global chart data store (useful between sessions)"""
    global _chart_data_store
    _chart_data_store.clear()

# ---------- Conversation Runner ----------
def run_conversation(llm, user_message: str, thread_id: str = "default"):
    """Helper function to run a single conversation turn"""
    graph = create_react_graph(llm)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get current state or create new
    try:
        current_state = graph.get_state(config).values
        messages = current_state.get("messages", [])
        graph_list = current_state.get("graph_list", [])
    except:
        messages = []
        graph_list = []
    
    # Add user message
    messages.append(HumanMessage(content=user_message))
    
    # Run the graph
    result = graph.invoke({
        "messages": messages,
        "graph_list": graph_list
    }, config)
    
    return result