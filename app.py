import os
import stripe
import re
import time
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
from decimal import Decimal, ROUND_UP
from dotenv import load_dotenv

# --- NEW OAUTH IMPORT ---
from authlib.integrations.flask_client import OAuth

# --- CLIENTS FOR API PROVIDERS ---
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic

# --- AGENT FRAMEWORK (LANGCHAIN/LANGGRAPH) ---
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.graph import StateGraph, END

# --- Imports for Postgres (production) and SQLite (local) ---
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

from typing import TypedDict, Annotated, List, Union
import operator
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage, BaseMessage

# Import your database models
from models import db, User 

# --- APPLICATION SETUP ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# --- CONFIGURATIONS (from .env file) ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['YOUR_DOMAIN'] = os.environ.get('YOUR_DOMAIN', 'http://127.0.0.1:5000')

app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# --- Auth0 Config ---
app.config['AUTH0_CLIENT_ID'] = os.environ.get('AUTH0_CLIENT_ID')
app.config['AUTH0_CLIENT_SECRET'] = os.environ.get('AUTH0_CLIENT_SECRET')
app.config['AUTH0_DOMAIN'] = os.environ.get('AUTH0_DOMAIN')

db.init_app(app)

# --- LOGIN MANAGER ---
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Setup Authlib (OAuth) ---
oauth = OAuth(app)
auth0 = oauth.register(
    'auth0',
    client_id=app.config['AUTH0_CLIENT_ID'],
    client_secret=app.config['AUTH0_CLIENT_SECRET'],
    api_base_url=f"https://{app.config['AUTH0_DOMAIN']}", 
    access_token_url=f"https://{app.config['AUTH0_DOMAIN']}/oauth/token",
    authorize_url=f"https://{app.config['AUTH0_DOMAIN']}/authorize",
    client_kwargs={
        'scope': 'openid profile email',
    },
    server_metadata_url=f"https://{app.config['AUTH0_DOMAIN']}/.well-known/openid-configuration"
)

# --- API CLIENT INITIALIZATION ---
try:
    llm_groq_llama3 = ChatGroq(model_name='llama3-8b-8192', api_key=os.environ.get('GROQ_API_KEY'))
    llm_openai_mini = ChatOpenAI(model='gpt-4o-mini', api_key=os.environ.get('OPENAI_API_KEY'))
    llm_openai_gpt4o = ChatOpenAI(model='gpt-4o', temperature=0, api_key=os.environ.get('OPENAI_API_KEY'))
except Exception as e:
    print(f"CRITICAL WARNING: Could not initialize all API clients. Missing keys? Error: {e}")

# --- BUSINESS LOGIC: PRICING & CREDITS ---
YOUR_COST_PER_CREDIT_KES = Decimal('2.00')
KES_TO_USD_RATE = Decimal('0.0077') 
API_COSTS_USD = {
    'llama3-8b': {'input': Decimal('0.05'), 'output': Decimal('0.08')},
    'gpt-4o-mini': {'input': Decimal('0.15'), 'output': Decimal('0.60')},
    'deepseek-coder-v2': {'input': Decimal('0.14'), 'output': Decimal('0.28')},
    'gpt-4o': {'input': Decimal('2.50'), 'output': Decimal('10.00')}
}

def calculate_cost_in_kes(model_name, input_tokens, output_tokens):
    costs_usd = API_COSTS_USD.get(model_name)
    if not costs_usd:
        if model_name != 'deepseek-coder-v2':
            print(f"No pricing info for model: {model_name}")
        return Decimal('0.0')
        
    input_cost_usd = (Decimal(str(input_tokens)) / Decimal('1000000')) * costs_usd['input']
    output_cost_usd = (Decimal(str(output_tokens)) / Decimal('1000000')) * costs_usd['output']
    total_cost_usd = input_cost_usd + output_cost_usd
    total_cost_kes = total_cost_usd / KES_TO_USD_RATE
    return total_cost_kes

def calculate_credits_to_deduct(total_cost_kes):
    credit_cost = (total_cost_kes / YOUR_COST_PER_CREDIT_KES).quantize(Decimal('0.0001'), rounding=ROUND_UP)
    return credit_cost

# --- "WORKER" API FUNCTIONS (Called by Agent or Triage) ---

# --- FIX: Removed @tool decorator here so it can be called normally ---
def call_simple_chat(prompt: str, system_prompt: str = "You are a fast and helpful assistant.") -> dict:
    """
    Call this tool for simple, conversational questions. Returns a dict.
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    response = llm_groq_llama3.invoke(messages)
    usage = response.response_metadata.get('token_usage', {})
    return {
        "answer": response.content,
        "usage": {"input_tokens": usage.get('prompt_tokens', 0), "output_tokens": usage.get('completion_tokens', 0)}
    }
# --- END OF FIX ---

@tool
def ui_builder_tool(prompt: str) -> str:
    """
    Call this tool to write high-quality frontend UI code (React, HTML, CSS, JavaScript).
    Use it for all visual components, website pages, and user-facing logic.
    """
    messages = [
        SystemMessage(content="You are an expert UI/UX and frontend developer. You write code for React, Tailwind, HTML/CSS, and JavaScript."),
        HumanMessage(content=prompt)
    ]
    response = llm_openai_mini.invoke(messages)
    usage = response.response_metadata.get('token_usage', {})
    return f"[Code Generated (Model: gpt-4o-mini, Input: {usage.get('prompt_tokens', 0)}, Output: {usage.get('completion_tokens', 0)})]:\n{response.content}"

# --- AGENT FRAMEWORK (LangGraph) ---
agent_tools = [ui_builder_tool]
llm_brain_with_tools = llm_openai_gpt4o.bind_tools(agent_tools)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    total_cost_kes: Annotated[Decimal, operator.add]

def agent_brain_node(state):
    response = llm_brain_with_tools.invoke(state["messages"])
    cost_kes = Decimal('0.0')
    if hasattr(response, 'response_metadata') and response.response_metadata:
        usage = response.response_metadata.get('token_usage', {})
        if usage:
            cost_kes = calculate_cost_in_kes(
                'gpt-4o',
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0)
            )
    return {"messages": [response], "total_cost_kes": cost_kes}

def tool_worker_node(state):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    worker_responses = []
    total_cost_kes = Decimal('0.0')
    for tool_call in tool_calls:
        tool_to_call = None
        if tool_call["name"] == "ui_builder_tool":
            tool_to_call = ui_builder_tool
            
        if tool_to_call:
            observation = tool_to_call.invoke(tool_call["args"]) 
            try:
                model_name_match = re.search(r"Model: (.*?),", observation)
                input_tokens_match = re.search(r"Input: (\d+),", observation)
                output_tokens_match = re.search(r"Output: (\d+)", observation)

                if model_name_match and input_tokens_match and output_tokens_match:
                    model = model_name_match.group(1)
                    input_tokens = int(input_tokens_match.group(1))
                    output_tokens = int(output_tokens_match.group(1))
                    total_cost_kes += calculate_cost_in_kes(model, input_tokens, output_tokens)
                else:
                    print(f"Could not parse token cost from tool message: {observation}")
            except Exception as e:
                print(f"Error parsing token cost: {e}")
            worker_responses.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        else:
            worker_responses.append(ToolMessage(content="Error: Unknown tool.", tool_call_id=tool_call["id"]))
    return {"messages": worker_responses, "total_cost_kes": total_cost_kes}

def router_edge(state):
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "call_tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent_brain", agent_brain_node)
workflow.add_node("call_tools", tool_worker_node)
workflow.set_entry_point("agent_brain")
workflow.add_conditional_edges("agent_brain", router_edge, {"call_tools": "call_tools", END: END})
workflow.add_edge("call_tools", "agent_brain")

# --- Smart Database Checkpointer ---
db_url = os.environ.get("DATABASE_URL")

if db_url and db_url.startswith("postgresql"):
    # PRODUCTION (Neon)
    print("--- Connecting to PostgreSQL for chat history ---")
    
    memory_checkpointer = PostgresSaver.from_conn_string(db_url)
    
    # We must create all tables *within* the app context
    with app.app_context(), memory_checkpointer as memory_saver:
        print("--- Creating 'users' table (if not exists) ---")
        db.create_all() 
        print("--- Creating 'langgraph_checkpoints' table (if not exists) ---")
        memory_saver.setup() 
        
else:
    # LOCAL DEVELOPMENT (SQLite)
    print("--- Using 'checkpoints.sqlite' for local chat history ---")
    memory_checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
    # Also create the 'users' table for local dev
    with app.app_context():
        db.create_all()

agent_app = workflow.compile(checkpointer=memory_checkpointer)


def run_agent_workflow(prompt, user_id):
    config = {"configurable": {"thread_id": str(user_id)}}
    system_message = SystemMessage(content="You are a full-stack expert. Your job is to create a plan and then call your worker tools (`ui_builder_tool`) to execute the plan. When finished, present all the generated code to the user in a single, clean response.")
    final_state = agent_app.invoke(
        {"messages": [system_message, HumanMessage(content=prompt)], "total_cost_kes": Decimal('0.0')}, 
        config=config
    )
    final_answer = final_state["messages"][-1].content
    total_cost_kes = Decimal('0.0')
    run_history = agent_app.get_state_history(config)
    for state in run_history:
        total_cost_kes += state.get('total_cost_kes', Decimal('0.0'))
    return {"answer": final_answer, "cost_kes": total_cost_kes}

# --- TRIAGE ROUTER LOGIC ---
def get_task_classification(prompt):
    system_prompt = """
    You are a high-speed task router. Your job is to classify the user's prompt.
    Does the user want a simple, conversational answer ("simple_chat")?
    Is it a request to build frontend UI, HTML, CSS, or React ("frontend_task")?
    Is it a complex, multi-step goal like "build a full app" ("complex_agent_task")?
    
    Respond with ONLY one phrase: "simple_chat", "frontend_task", or "complex_agent_task".
    """
    response_data = call_simple_chat(prompt, system_prompt=system_prompt)
    answer = response_data["answer"].lower().strip()
    classification = "simple_chat" 
    if "complex_agent_task" in answer:
        classification = "complex_agent_task"
    elif "frontend_task" in answer:
        classification = "frontend_task"
    triage_cost_kes = calculate_cost_in_kes(
        'llama3-8b', 
        response_data['usage']['input_tokens'], 
        response_data['usage']['output_tokens']
    )
    return classification, triage_cost_kes

# --- MAIN API ENDPOINT (USER-FACING) ---
@app.route("/api/ask", methods=["POST"])
@login_required
def handle_ask():
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    user_credits = Decimal(current_user.get_credit_balance())
    total_cost_kes = Decimal('0.0')
    
    try:
        classification, triage_cost_kes = get_task_classification(prompt)
        total_cost_kes += triage_cost_kes
        
        final_answer = ""
        cost_model = ""
        input_tokens = 0
        output_tokens = 0

        if classification == "simple_chat":
            response_data = call_simple_chat(prompt)
            final_answer = response_data["answer"]
            cost_model = 'llama3-8b'
            input_tokens = response_data['usage']['input_tokens']
            output_tokens = response_data['usage']['output_tokens']
            total_cost_kes += calculate_cost_in_kes(cost_model, input_tokens, output_tokens)

        elif classification == "frontend_task":
            final_answer = ui_builder_tool.invoke(prompt)
            cost_model = 'gpt-4o-mini'
            input_tokens_match = re.search(r"Input: (\d+),", final_answer)
            output_tokens_match = re.search(r"Output: (\d+)", final_answer)
            if input_tokens_match and output_tokens_match:
                input_tokens = int(input_tokens_match.group(1))
                output_tokens = int(output_tokens_match.group(1))
                total_cost_kes += calculate_cost_in_kes(cost_model, input_tokens, output_tokens)

        else: # "complex_agent_task"
            agent_response = run_agent_workflow(prompt, current_user.id)
            final_answer = agent_response["answer"]
            total_cost_kes += agent_response["cost_kes"]
            
        credit_cost = calculate_credits_to_deduct(total_cost_kes)
        
        if user_credits < credit_cost:
            return jsonify({
                "error": "Not enough credits for this request.",
                "cost": float(credit_cost),
                "balance": float(user_credits)
            }), 402
            
        current_user.credits = user_credits - credit_cost
        db.session.commit()
        
        return jsonify({
            "answer": final_answer,
            "cost": float(credit_cost),
            "credits_remaining": float(current_user.credits)
        })
    except Exception as e:
        print(f"Error in /api/ask: {e}")
        # Rollback in case of error before commit
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# --- HTML PAGE & USER AUTH ROUTES ---
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat_interface'))
    return render_template('landing.html') 

@app.route('/app')
@login_required
def chat_interface():
    return render_template('ai_assistant.html', user_credits=current_user.get_credit_balance())

# --- NEW/UPDATED AUTH ROUTES ---

@app.route('/login')
def login():
    """ Redirects to Auth0 login page. """
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for('callback', _external=True),
        prompt='login'
    )

@app.route('/signup')
def signup():
    """ Redirects to Auth0 signup page. """
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for('callback', _external=True),
        screen_hint='signup',
        prompt='login'
    )

@app.route('/logout')
def logout():
    """ Logs the user out of *your* app and Auth0. """
    session.clear()
    logout_user() 
    
    # Build the Auth0 logout URL
    auth0_domain = app.config['AUTH0_DOMAIN']
    client_id = app.config['AUTH0_CLIENT_ID']
    return_to = url_for('index', _external=True)
    
    logout_url = f'https://{auth0_domain}/v2/logout?'
    logout_url += f'client_id={client_id}&'
    logout_url += f'returnTo={return_to}'
    
    return redirect(logout_url)

@app.route('/callback')
def callback():
    """ Auth0 sends the user back here after they log in. """
    try:
        token = oauth.auth0.authorize_access_token()
        # Fetch user info from Auth0
        user_info_response = auth0.get('userinfo')
        user_info_response.raise_for_status() # Raise exception for bad responses
        user_info = user_info_response.json()
        
        auth0_id = user_info.get('sub')
        user_email = user_info.get('email')

        if not auth0_id or not user_email:
            flash("Error: Email or Auth0 ID missing from profile.", "danger")
            return redirect(url_for('index'))

        # Find user in our DB
        user = User.query.filter_by(auth0_id=auth0_id).first()
        
        if not user:
            # Check if email is already in use by a *different* auth0_id (edge case)
            existing_email_user = User.query.filter_by(email=user_email).first()
            if existing_email_user:
                flash("This email is already registered. Please log in.", "danger")
                return redirect(url_for('login'))

            # New user! Create them in your database.
            user = User(
                auth0_id=auth0_id,
                email=user_email
                # Credits default to 5.0 in the model
            )
            db.session.add(user)
            db.session.commit()
            
        # Log them into the Flask session
        login_user(user, remember=True)
        return redirect(url_for('chat_interface'))
        
    except Exception as e:
        print(f"Error during callback: {e}")
        flash("An error occurred during login. Please try again.", "danger")
        return redirect(url_for('index'))

# --- NEW PRICING & PAYMENT ROUTES ---
@app.route('/pricing')
@login_required
def pricing_page():
    # This page is now just a static HTML page
    return render_template('pricing.html')
    
# --- (All Stripe/Payment API routes are removed) ---

# --- Create Database and Run App ---
if __name__ == "__main__":
    # db.create_all() has been moved up to the checkpointing logic
    # so it runs in production
    app.run(host='0.0.0.0', port=5000, debug=True)