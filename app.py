import os
import re
from decimal import Decimal, ROUND_UP
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
from sqlalchemy import create_engine, text # <--- ADDED FOR DISTRIBUTED DB

# --- AI IMPORTS ---
from langchain_groq import ChatGroq
# Tavily Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage

# --- CRITICAL FIX: Import directly from Pydantic ---
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from typing import TypedDict, Annotated, List
import operator

# --- DATABASE MODELS ---
from models import db, User

# --- SETUP ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# --- CONFIG ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['YOUR_DOMAIN'] = os.environ.get('YOUR_DOMAIN')
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# --- DATABASE STABILITY (User DB) ---
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_size": 2,
    "max_overflow": 0,
    "pool_timeout": 10
}

# Auth0
app.config['AUTH0_CLIENT_ID'] = os.environ.get('AUTH0_CLIENT_ID')
app.config['AUTH0_CLIENT_SECRET'] = os.environ.get('AUTH0_CLIENT_SECRET')
app.config['AUTH0_DOMAIN'] = os.environ.get('AUTH0_DOMAIN')

db.init_app(app)

# --- DISTRIBUTED DB ROUTER (ASSIGNMENT LOGIC) ---
# This sets up the connection to the SECOND database (Node B)
node_b_url = os.environ.get('DATABASE_URL_NODE_B')
engine_node_b = create_engine(node_b_url) if node_b_url else None

def save_distributed_log(user_id, activity):
    """
    Routes data to physically different servers based on User ID.
    This runs BEFORE the AI thinks, ensuring the log is always captured.
    """
    print(f"🔄 [Distributed] Attempting to route User ID: {user_id}")
    try:
        # 1. The Router: Decide destination based on User ID (Even/Odd)
        if user_id % 2 == 0:
            target_engine = db.engine # Node A (Primary)
            server_name = "Node A (Primary - Even ID)"
        else:
            if engine_node_b:
                target_engine = engine_node_b
                server_name = "Node B (Secondary - Odd ID)"
            else:
                target_engine = db.engine
                server_name = "Node A (Fallback - Node B Missing)"

        # 2. The Write Operation
        with target_engine.connect() as conn:
            # Ensure table exists
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS distributed_activity_logs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    activity TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Insert Log
            conn.execute(
                text("INSERT INTO distributed_activity_logs (user_id, activity) VALUES (:uid, :act)"),
                {"uid": user_id, "act": activity}
            )
            conn.commit()
            
        print(f"✅ [Distributed Success] Data written to {server_name}")
        return True

    except Exception as e:
        print(f"❌ [Distributed Fail] Could not write log: {e}")
        return False

# --- LOGIN MANAGER ---
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- OAUTH ---
oauth = OAuth(app)
auth0 = oauth.register(
    'auth0',
    client_id=app.config['AUTH0_CLIENT_ID'],
    client_secret=app.config['AUTH0_CLIENT_SECRET'],
    api_base_url=f"https://{app.config['AUTH0_DOMAIN']}",
    access_token_url=f"https://{app.config['AUTH0_DOMAIN']}/oauth/token",
    authorize_url=f"https://{app.config['AUTH0_DOMAIN']}/authorize",
    client_kwargs={'scope': 'openid profile email'},
    server_metadata_url=f"https://{app.config['AUTH0_DOMAIN']}/.well-known/openid-configuration"
)

# ==============================================================================
# 1. INITIALIZE AI
# ==============================================================================

try:
    if not os.environ.get('GROQ_API_KEY'):
        print("❌ CRITICAL: GROQ_API_KEY is missing")
    
    groq_llm = ChatGroq(
        model_name='llama-3.3-70b-versatile', 
        api_key=os.environ.get('GROQ_API_KEY'),
        temperature=0
    )
    print("✅ Groq AI Client Initialized")

except Exception as e:
    print(f"CRITICAL: AI Client failed to load. {e}")

# ==============================================================================
# 2. DEFINE TOOLS
# ==============================================================================

class CodeInput(BaseModel):
    prompt: str = Field(description="Detailed description of the code to build.")

class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")

@tool(args_schema=CodeInput)
def ui_builder_tool(prompt: str) -> str:
    """Use this tool to generate HTML, CSS, React, or Frontend code."""
    messages = [
        SystemMessage(content="You are an expert Frontend Developer. Generate clean, modern HTML/Tailwind code."),
        HumanMessage(content=prompt)
    ]
    response = groq_llm.invoke(messages)
    return f"[UI Generated by Groq] \n\n {response.content}"

@tool(args_schema=CodeInput)
def backend_builder_tool(prompt: str) -> str:
    """Use this tool to generate Python, Flask, SQL, or complex Backend logic."""
    messages = [
        SystemMessage(content="You are a Senior Backend Engineer. Write secure, production-ready Python code."),
        HumanMessage(content=prompt)
    ]
    response = groq_llm.invoke(messages)
    return f"[Backend Generated by Groq] \n\n {response.content}"

@tool(args_schema=SearchInput)
def web_search_tool(query: str) -> str:
    """
    Use this tool to find real-time information, news, weather, or specific facts.
    """
    print(f"🔎 TAVILY SEARCHING FOR: '{query}'") 
    
    if not os.environ.get('TAVILY_API_KEY'):
        return "Error: TAVILY_API_KEY is missing in Render Environment Variables."

    try:
        search = TavilySearchResults(max_results=5)
        results = search.invoke(query)
        
        if not results:
            return "No search results found."
        
        response_text = f"Search Results for '{query}':\n\n"
        for r in results:
            content = r.get('content', 'No content')
            url = r.get('url', '#')
            response_text += f"Source: {url}\nFact: {content}\n\n"
            
        return response_text

    except Exception as e:
        print(f"❌ SEARCH ERROR: {e}")
        return f"Search Tool Error: {str(e)}"

# ==============================================================================
# 3. BUILD THE AGENT GRAPH
# ==============================================================================

tools = [ui_builder_tool, backend_builder_tool, web_search_tool]
agent_brain = groq_llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def brain_node(state):
    return {"messages": [agent_brain.invoke(state["messages"])]}

def tool_node(state):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    results = []
    
    for t in tool_calls:
        selected_tool = {
            "ui_builder_tool": ui_builder_tool,
            "backend_builder_tool": backend_builder_tool,
            "web_search_tool": web_search_tool
        }.get(t["name"])
        
        if selected_tool:
            try:
                output = selected_tool.invoke(t["args"])
            except Exception as e:
                output = f"Tool Error: {str(e)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(output)))
            
    return {"messages": results}

def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls: return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("brain", brain_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("brain")
workflow.add_conditional_edges("brain", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "brain")

# ==============================================================================
# 4. DATABASE CONNECTION HELPERS
# ==============================================================================

def get_transient_checkpointer():
    """
    Creates a temporary connection pool for ONE request, then closes it.
    This prevents "PoolTimeout" and "SSL Closed" errors.
    """
    db_url = os.environ.get("DATABASE_URL")
    if db_url and db_url.startswith("postgresql"):
        pool = ConnectionPool(
            conninfo=db_url,
            min_size=0,
            max_size=1,
            kwargs={"autocommit": True}
        )
        checkpointer = PostgresSaver(pool)
        return checkpointer, pool
    else:
        # Local fallback
        return SqliteSaver.from_conn_string("checkpoints.sqlite"), None

def init_tables():
    """Runs once on startup to ensure tables exist."""
    saver, pool = get_transient_checkpointer()
    try:
        with app.app_context():
            db.create_all()
            if hasattr(saver, 'setup'):
                saver.setup()
    except Exception as e:
        print(f"Startup Warning: {e}")
    finally:
        if pool: pool.close()

# ==============================================================================
# 5. ROUTING & API
# ==============================================================================

def simple_chat(prompt):
    return groq_llm.invoke([HumanMessage(content=prompt)]).content

@app.route("/api/ask", methods=["POST"])
@login_required
def handle_ask():
    user_prompt = request.json.get("prompt")
    user_id = current_user.auth0_id
    
    # --- STEP 1: DISTRIBUTED LOGGING (The "Log First" Fix) ---
    # We run this immediately so the assignment proof is saved before AI starts.
    save_distributed_log(current_user.id, user_prompt)
    
    # Routing
    try:
        router_sys = "Classify: 1. 'simple' (jokes, greetings). 2. 'complex' (search, code, facts). Return ONLY 'simple' or 'complex'."
        classification = groq_llm.invoke([
            SystemMessage(content=router_sys), 
            HumanMessage(content=user_prompt)
        ]).content.strip().lower()
    except:
        classification = 'complex' 

    final_answer = ""
    cost = 0.5

    if "simple" in classification:
        final_answer = simple_chat(user_prompt)
    else:
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": 50}
        
        agent_sys = SystemMessage(content="""
        You are a helpful AI assistant.
        RULES:
        1. Use 'web_search_tool' for people, news, or facts.
        2. Use 'ui_builder_tool' or 'backend_builder_tool' for coding.
        3. When calling tools, use JSON format.
        """)

        # --- TRANSIENT EXECUTION ---
        pool = None
        try:
            # 1. Open fresh connection
            checkpointer, pool = get_transient_checkpointer()
            
            # 2. Compile graph with this specific connection
            app_with_db = workflow.compile(checkpointer=checkpointer)

            # 3. Run Agent
            result = app_with_db.invoke(
                {"messages": [agent_sys, HumanMessage(content=user_prompt)]}, 
                config=config
            )
            final_answer = result["messages"][-1].content
            
        except Exception as e:
            print(f"⚠️ Agent Error: {e}")
            final_answer = f"I encountered an error: {str(e)}"
        finally:
            # 4. Close connection immediately to free up the slot
            if pool: pool.close()

    # Billing
    try:
        current_user.credits -= Decimal(cost)
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({
        "answer": final_answer,
        "cost": cost,
        "remaining": float(current_user.credits) if current_user.credits else 0.0
    })

# --- ROUTES ---
@app.route('/')
def index():
    if current_user.is_authenticated: return redirect(url_for('chat_interface'))
    return render_template('landing.html')

@app.route('/app')
@login_required
def chat_interface():
    return render_template('ai_assistant.html', user_credits=current_user.get_credit_balance())

@app.route('/login')
def login(): return oauth.auth0.authorize_redirect(redirect_uri=url_for('callback', _external=True))

@app.route('/signup')
def signup(): return oauth.auth0.authorize_redirect(redirect_uri=url_for('callback', _external=True), screen_hint='signup')

@app.route('/callback')
def callback():
    try:
        token = oauth.auth0.authorize_access_token()
        user_info = auth0.get('userinfo').json()
        auth0_id = user_info.get('sub')
        email = user_info.get('email')
        
        if not auth0_id: return redirect(url_for('index'))

        user = User.query.filter_by(auth0_id=auth0_id).first()
        if not user:
            user = User.query.filter_by(email=email).first()
            if user: 
                user.auth0_id = auth0_id
            else:
                user = User(auth0_id=auth0_id, email=email, credits=Decimal('50.0'))
                db.session.add(user)
            db.session.commit()
        
        login_user(user, remember=True)
        return redirect(url_for('chat_interface'))
    except Exception as e:
        db.session.rollback()
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    logout_user()
    return redirect(f"https://{app.config['AUTH0_DOMAIN']}/v2/logout?client_id={app.config['AUTH0_CLIENT_ID']}&returnTo={url_for('index', _external=True)}")

@app.route('/pricing')
@login_required
def pricing_page():
    return render_template('pricing.html')

if __name__ == "__main__":
    init_tables() # Safe table creation
    app.run(host='0.0.0.0', port=5000, debug=True)