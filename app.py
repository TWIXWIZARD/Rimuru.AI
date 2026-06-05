# ===================== IMPORTS =====================
import io
import base64
import tempfile
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auth import create_user, login_user
from chat_store import (
    create_session, get_sessions, get_or_create_default_session,
    save_message, load_messages, get_message_count,
    delete_session, rename_session, get_all_user_messages,
    save_graph, load_graphs, load_all_user_graphs, delete_graph,
    save_note, load_notes, delete_note,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
try:
    from langchain.agents import create_react_agent as _create_react_agent
except ImportError:
    _create_react_agent = None
try:
    from langchain_community.agent_toolkits.load_tools import load_tools as _load_tools
except ImportError:
    _load_tools = None
from db import init_db

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="NexusAI — Intelligent Document & Data Platform",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ---- GLOBAL RESET ---- */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050810 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- HIDE DEFAULT ELEMENTS ---- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background: #080c18 !important;
    border-right: 1px solid #1a2040 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ---- SIDEBAR LOGO AREA ---- */
.sidebar-logo {
    background: linear-gradient(135deg, #0d1528 0%, #111a35 100%);
    padding: 24px 20px 20px;
    border-bottom: 1px solid #1a2545;
    margin-bottom: 16px;
}
.sidebar-logo h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.sidebar-logo p {
    font-size: 11px !important;
    color: #4a5568 !important;
    margin: 4px 0 0 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ---- SESSION CARDS ---- */
.session-card {
    background: #0d1528;
    border: 1px solid #1a2545;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 10px;
}
.session-card:hover { border-color: #4f8ef7; background: #111e3a; }
.session-card.active { border-color: #4f8ef7; background: #0f1e40; }
.session-icon { font-size: 16px; }
.session-info { flex: 1; overflow: hidden; }
.session-title { font-size: 13px; font-weight: 500; color: #c8d0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.session-meta { font-size: 11px; color: #3a4560; margin-top: 2px; }

/* ---- MAIN HEADER ---- */
.main-header {
    background: linear-gradient(135deg, #080c18 0%, #0d1528 100%);
    border-bottom: 1px solid #1a2545;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 0;
}
.main-header h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #e8eaf0 !important;
    margin: 0 !important;
}

/* ---- MODE BADGE ---- */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.mode-rag { background: #1a3a5c; color: #60a5fa; border: 1px solid #2a5080; }
.mode-csv { background: #1a3a2a; color: #4ade80; border: 1px solid #2a6040; }
.mode-agent { background: #2a1a3a; color: #c084fc; border: 1px solid #4a2a6a; }

/* ---- CHAT MESSAGES ---- */
.chat-wrapper {
    max-height: calc(100vh - 300px);
    overflow-y: auto;
    padding: 20px 0;
    scroll-behavior: smooth;
}
.msg-human {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
    animation: slideInRight 0.3s ease;
}
.msg-ai {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    animation: slideInLeft 0.3s ease;
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
.bubble-human {
    background: linear-gradient(135deg, #1e3a6e, #2a4f96);
    border: 1px solid #2a4f96;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
    color: #e0eaff;
}
.bubble-ai {
    background: #0d1528;
    border: 1px solid #1a2545;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    color: #c8d4e8;
}
.avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin: 0 8px;
}
.avatar-human { background: linear-gradient(135deg, #1e3a6e, #4f8ef7); }
.avatar-ai { background: linear-gradient(135deg, #2a1a4a, #7c3aed); }
.msg-time { font-size: 10px; color: #2a3550; margin-top: 4px; }

/* ---- STATS CARDS ---- */
.stat-card {
    background: #0d1528;
    border: 1px solid #1a2545;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg, #4f8ef7, #a78bfa));
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 12px; color: #4a5568; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ---- GRAPH GALLERY ---- */
.graph-card {
    background: #0d1528;
    border: 1px solid #1a2545;
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.2s;
}
.graph-card:hover { border-color: #4f8ef7; transform: translateY(-2px); }
.graph-card img { width: 100%; }
.graph-meta { padding: 12px; }
.graph-title { font-size: 13px; font-weight: 600; color: #c8d0e0; }
.graph-q { font-size: 11px; color: #4a5568; margin-top: 4px; }

/* ---- FILE UPLOAD AREA ---- */
[data-testid="stFileUploader"] {
    background: #0d1528 !important;
    border: 1.5px dashed #1a2545 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4f8ef7 !important;
}

/* ---- INPUTS ---- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #0d1528 !important;
    border: 1px solid #1a2545 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.2) !important;
}

/* ---- CHAT INPUT ---- */
[data-testid="stChatInput"] {
    background: #0d1528 !important;
    border: 1px solid #1a2545 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8eaf0 !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background: linear-gradient(135deg, #1e3a6e, #2a4f96) !important;
    color: #e0eaff !important;
    border: 1px solid #2a4f96 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a4f96, #4f8ef7) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,142,247,0.3) !important;
}

/* ---- TABS ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #080c18 !important;
    border-bottom: 1px solid #1a2545 !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5568 !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #0d1528 !important;
    color: #4f8ef7 !important;
    border-bottom: 2px solid #4f8ef7 !important;
}

/* ---- SELECTBOX ---- */
[data-testid="stSelectbox"] > div > div {
    background: #0d1528 !important;
    border: 1px solid #1a2545 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}

/* ---- DATAFRAME ---- */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---- NOTE CARD ---- */
.note-card {
    background: #0d1528;
    border: 1px solid #1a2545;
    border-left: 3px solid #a78bfa;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
}
.note-title { font-size: 14px; font-weight: 600; color: #c8d0e0; }
.note-content { font-size: 13px; color: #6b7a99; margin-top: 6px; }
.note-date { font-size: 11px; color: #2a3550; margin-top: 8px; }

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: #1a2545; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #4f8ef7; }

/* ---- SUCCESS / ERROR ---- */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ---- DIVIDER ---- */
hr { border-color: #1a2545 !important; }

/* ---- SECTION HEADING ---- */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #2a3a5c;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 20px 0 10px;
    padding-left: 4px;
    border-left: 2px solid #4f8ef7;
    padding-left: 8px;
}

/* ---- WELCOME SCREEN ---- */
.welcome-hero {
    text-align: center;
    padding: 80px 40px;
}
.welcome-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
}
.welcome-hero p {
    color: #4a5568;
    font-size: 16px;
    max-width: 500px;
    margin: 0 auto 40px;
    line-height: 1.8;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    max-width: 700px;
    margin: 0 auto;
}
.feature-item {
    background: #0d1528;
    border: 1px solid #1a2545;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.feature-icon { font-size: 28px; margin-bottom: 10px; }
.feature-title { font-size: 13px; font-weight: 600; color: #8a94b4; }
</style>
""", unsafe_allow_html=True)

# ===================== INIT DB =====================
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

# ===================== BEDROCK (CACHED) =====================
@st.cache_resource
def get_embeddings():
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")

@st.cache_resource
def get_llm():
    return ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", region="us-east-1", temperature=0)

@st.cache_resource
def get_agent():
    tools = [DuckDuckGoSearchRun(), WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))]
    if _create_react_agent is not None:
        try:
            from langchain import hub
            prompt = hub.pull("hwchase17/react")
            return _create_react_agent(get_llm(), tools, prompt=prompt)
        except Exception:
            pass
    # Fallback: return tools list so we can call them manually
    return tools

embedding = get_embeddings()
llm = get_llm()
agent = get_agent()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def ingest_document(uploaded_file, ext):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    if ext == "pdf":
        loader = PyPDFLoader(path)
    elif ext == "docx":
        loader = Docx2txtLoader(path)
    elif ext == "pptx":
        loader = UnstructuredPowerPointLoader(path)
    else:
        raise ValueError("Unsupported file")
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, embedding)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_history_messages(history):
    """Convert DB history rows to LangChain message dicts."""
    msgs = []
    for role, content, _ in history:
        if role == "human":
            msgs.append(("human", content))
        elif role == "ai" and not content.startswith("[GRAPH:"):
            msgs.append(("ai", content))
    return msgs

def rag_answer(question, vectordb, history=None):
    retriever = vectordb.as_retriever()
    context_docs = retriever.invoke(question)
    context_text = format_docs(context_docs)

    messages = [("system",
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "Be detailed and structured. Remember what was said earlier in the conversation.")]

    if history:
        messages.extend(build_history_messages(history))

    messages.append(("human", f"<context>{context_text}</context>\nQuestion: {question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

sql_prompt = """
You are an expert Python pandas engineer.
Generate pandas OR matplotlib code that will be EXECUTED automatically.
DataFrame name is `df`. Column names are EXACT and case-sensitive.
RULES:
1. Output ONLY executable Python code
2. Do NOT import anything
3. Use df directly
4. For plots: matplotlib.pyplot is available as plt. End with plt.tight_layout() then plt.show()
5. If numeric → return ONLY expression
6. Do NOT use eval, exec, open, os, sys, __

Question: {question}
"""

def generate_structured_code(question, df, history=None):
    history_text = ""
    if history:
        lines = []
        for role, content, _ in history[-6:]:  # last 6 msgs for context
            if role == "human":
                lines.append(f"User: {content}")
            elif role == "ai" and not content.startswith("[GRAPH:"):
                lines.append(f"Assistant: {content}")
        if lines:
            history_text = "Previous conversation:\n" + "\n".join(lines) + "\n\n"
    return llm.invoke(history_text + sql_prompt.format(question=question)).content.strip()

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#050810", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ===================== SESSION STATE =====================
for k in ["user_id", "active_session_id", "vector_db", "df", "username", "active_tab"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>✦ NexusAI</h1>
        <p>Intelligent Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.user_id:
        # ---- AUTH ----
        auth_tab = st.tabs(["🔑 Login", "✨ Sign Up"])

        with auth_tab[0]:
            le = st.text_input("Email", placeholder="you@example.com", key="login_email")
            lp = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            if st.button("Sign In →", use_container_width=True, key="login_btn"):
                user_id = login_user(le, lp)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.active_session_id = get_or_create_default_session(user_id)
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with auth_tab[1]:
            su = st.text_input("Username", placeholder="yourname", key="signup_user")
            se = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            sp = st.text_input("Password", type="password", placeholder="••••••••", key="signup_pass")
            if st.button("Create Account →", use_container_width=True, key="signup_btn"):
                if create_user(su, se, sp):
                    st.success("Account created! Please login.")
                else:
                    st.error("Email already registered")

    else:
        # ---- NAV ----
        st.markdown('<div class="section-heading">Navigation</div>', unsafe_allow_html=True)
        nav_options = ["💬 Chat", "🎛 Dashboard", "📊 Graph Gallery", "📝 Notes", "📈 Analytics", "⚙️ Settings"]
        nav_choice = st.radio("", nav_options, label_visibility="collapsed", key="nav_radio")
        st.session_state.active_tab = nav_choice

        st.divider()

        # ---- SESSIONS ----
        st.markdown('<div class="section-heading">Sessions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            new_session_name = st.text_input("", placeholder="Session name...", label_visibility="collapsed", key="new_sess_name")
        with col2:
            if st.button("＋", key="create_session_btn"):
                if new_session_name.strip():
                    sid = create_session(st.session_state.user_id, new_session_name.strip())
                    st.session_state.active_session_id = sid
                    st.session_state.vector_db = None
                    st.session_state.df = None
                    st.rerun()

        sessions = get_sessions(st.session_state.user_id)
        mode_icons = {"rag": "📄", "csv": "📊", "general": "🤖"}

        for sid, title, mode, created_at in sessions:
            is_active = sid == st.session_state.active_session_id
            msg_count = get_message_count(sid)
            card_class = "session-card active" if is_active else "session-card"
            icon = mode_icons.get(mode, "🤖")
            date_str = created_at[:10] if created_at else ""

            col_s, col_d = st.columns([4, 1])
            with col_s:
                if st.button(
                    f"{icon} {title}  ({msg_count} msgs)",
                    key=f"sess_{sid}",
                    use_container_width=True,
                ):
                    st.session_state.active_session_id = sid
                    st.session_state.vector_db = None
                    st.session_state.df = None
                    st.rerun()
            with col_d:
                if st.button("🗑", key=f"del_sess_{sid}"):
                    delete_session(sid)
                    if is_active:
                        remaining = get_sessions(st.session_state.user_id)
                        st.session_state.active_session_id = remaining[0][0] if remaining else None
                    st.rerun()

        st.divider()
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ===================== MAIN AREA =====================
if not st.session_state.user_id:
    # ---- WELCOME SCREEN ----
    st.markdown("""
    <div class="welcome-hero">
        <h1>✦ NexusAI</h1>
        <p>An intelligent platform for document analysis, data visualization, and AI-powered insights. Login to begin.</p>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Document RAG<br>PDF · DOCX · PPTX</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div class="feature-title">CSV Analysis<br>Charts & Insights</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI Agent<br>Web · Wikipedia</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🗂</div>
                <div class="feature-title">Chat History<br>Multi-session</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🖼</div>
                <div class="feature-title">Graph Gallery<br>Saved Charts</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📝</div>
                <div class="feature-title">Smart Notes<br>AI-assisted</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

active_tab = st.session_state.active_tab or "💬 Chat"

# ================================================================
# TAB: CHAT
# ================================================================
if active_tab == "💬 Chat":
    if not st.session_state.active_session_id:
        st.warning("Create or select a session from the sidebar to begin.")
        st.stop()

    # ---- FILE UPLOAD ----
    uploaded = st.file_uploader(
        "Upload a document or data file to analyze",
        type=["pdf", "docx", "pptx", "csv"],
        help="PDF/DOCX/PPTX for RAG · CSV for data analysis"
    )

    mode_label = "🤖 AI Agent"
    if uploaded:
        ext = uploaded.name.split(".")[-1].lower()
        if ext in ["pdf", "docx", "pptx"]:
            if st.session_state.vector_db is None:
                with st.spinner("Indexing document…"):
                    st.session_state.vector_db = ingest_document(uploaded, ext)
                    st.session_state.df = None
                st.success(f"✓ {uploaded.name} indexed — ask questions about it!")
            mode_label = "📄 Document RAG"
        elif ext == "csv":
            if st.session_state.df is None:
                st.session_state.df = pd.read_csv(uploaded)
                st.session_state.vector_db = None
            st.success(f"✓ {uploaded.name} loaded — {len(st.session_state.df):,} rows")
            with st.expander("Preview data"):
                st.dataframe(st.session_state.df.head(8), use_container_width=True)
            mode_label = "📊 CSV Analysis"

    # ---- MODE INDICATOR ----
    if st.session_state.df is not None:
        st.markdown('<span class="mode-badge mode-csv">📊 CSV Analysis Mode</span>', unsafe_allow_html=True)
    elif st.session_state.vector_db is not None:
        st.markdown('<span class="mode-badge mode-rag">📄 Document RAG Mode</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="mode-badge mode-agent">🤖 Agent Mode</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ---- CHAT HISTORY DISPLAY ----
    history = load_messages(st.session_state.active_session_id)
    if not history:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0; color: #2a3550;">
            <div style="font-size: 40px; margin-bottom: 16px;">✦</div>
            <div style="font-size: 16px; font-weight: 600; color: #3a4a6a;">Start the conversation</div>
            <div style="font-size: 13px; margin-top: 8px;">Upload a file or ask anything below</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for role, msg, ts in history:
            time_str = ts[11:16] if ts and len(ts) > 15 else ""
            if role == "human":
                st.markdown(f"""
                <div class="msg-human">
                    <div>
                        <div class="bubble-human">{msg}</div>
                        <div class="msg-time" style="text-align:right">{time_str}</div>
                    </div>
                    <div class="avatar avatar-human">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if it's a graph placeholder
                if msg and msg.startswith("[GRAPH:"):
                    gid_str = msg.replace("[GRAPH:", "").replace("]", "").strip()
                    st.markdown(f"""
                    <div class="msg-ai">
                        <div class="avatar avatar-ai">✦</div>
                        <div>
                            <div class="bubble-ai" style="font-style:italic; color:#4a5568;">📊 Chart generated — view in Graph Gallery</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="msg-ai">
                        <div class="avatar avatar-ai">✦</div>
                        <div>
                            <div class="bubble-ai">{msg}</div>
                            <div class="msg-time">{time_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ---- CHAT INPUT ----
    question = st.chat_input("Ask a question, analyze data, or explore your document…")

    if question:
        save_message(st.session_state.active_session_id, "human", question)
        # Reload history AFTER saving the new human message so it's included
        chat_history = load_messages(st.session_state.active_session_id)

        with st.spinner("Thinking…"):
            try:
                if st.session_state.vector_db:
                    answer = rag_answer(question, st.session_state.vector_db, history=chat_history)
                    save_message(st.session_state.active_session_id, "ai", answer)

                elif st.session_state.df is not None:
                    graph_words = ["plot", "graph", "chart", "bar", "line", "pie", "histogram", "scatter", "visualize", "show"]
                    is_graph = any(w in question.lower() for w in graph_words)

                    code = generate_structured_code(question, st.session_state.df, history=chat_history)

                    if is_graph:
                        fig, ax = plt.subplots(facecolor="#0d1528")
                        ax.set_facecolor("#080c18")
                        for spine in ax.spines.values():
                            spine.set_color("#1a2545")
                        ax.tick_params(colors="#6b7a99")
                        ax.xaxis.label.set_color("#6b7a99")
                        ax.yaxis.label.set_color("#6b7a99")
                        ax.title.set_color("#c8d4e8")

                        exec(code, {"__builtins__": {}}, {"df": st.session_state.df, "plt": plt, "ax": ax})

                        # Save graph to DB
                        img_b64 = fig_to_base64(fig)
                        graph_title = question[:50] + "…" if len(question) > 50 else question
                        gid = save_graph(st.session_state.active_session_id, img_b64, question, graph_title)

                        st.pyplot(fig)
                        plt.close(fig)

                        answer = f"[GRAPH:{gid}]"
                    else:
                        result = eval(code, {"__builtins__": {}}, {"df": st.session_state.df})
                        answer = str(result)

                    save_message(st.session_state.active_session_id, "ai", answer)

                else:
                    try:
                        # Build full conversation history for the LLM
                        lc_messages = [
                            ("system",
                             "You are NexusAI, a helpful and intelligent assistant. "
                             "Remember everything said earlier in this conversation and "
                             "use it to give contextual, accurate answers.")
                        ]
                        lc_messages.extend(build_history_messages(chat_history))
                        # The latest human message is already in chat_history; no need to append again.

                        # agent is either a real agent or a list of tools (fallback)
                        if not isinstance(agent, list):
                            try:
                                history_msgs = [HumanMessage(content=m[1]) if m[0]=="human"
                                                else type("AI", (), {"content": m[1]})()
                                                for m in lc_messages[1:]]
                                result = agent.invoke({"messages": history_msgs})
                                answer = result["messages"][-1].content
                                if isinstance(answer, list):
                                    answer = answer[0].get("text", str(answer))
                            except Exception:
                                raise
                        else:
                            raise ValueError("Using LLM with history")
                    except Exception:
                        prompt_with_history = ChatPromptTemplate.from_messages(lc_messages)
                        chain = prompt_with_history | llm | StrOutputParser()
                        answer = chain.invoke({})
                    save_message(st.session_state.active_session_id, "ai", answer)

            except Exception as e:
                answer = f"⚠️ Error: {e}"
                save_message(st.session_state.active_session_id, "ai", answer)

        st.rerun()

# ================================================================
# TAB: GRAPH GALLERY
# ================================================================
elif active_tab == "📊 Graph Gallery":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800; 
             background: linear-gradient(90deg, #4f8ef7, #a78bfa); -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">Graph Gallery</div>
        <div style="font-size: 13px; color: #4a5568; margin-top: 4px;">All charts generated across your sessions</div>
    </div>
    """, unsafe_allow_html=True)

    all_graphs = load_all_user_graphs(st.session_state.user_id)

    if not all_graphs:
        st.markdown("""
        <div style="text-align:center; padding: 80px; color: #2a3550;">
            <div style="font-size: 48px;">📊</div>
            <div style="font-size: 16px; margin-top: 16px; color: #3a4a6a; font-weight: 600;">No charts yet</div>
            <div style="font-size: 13px; margin-top: 8px;">Ask a visualization question in CSV mode to generate charts</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols_per_row = 2
        for i in range(0, len(all_graphs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(all_graphs):
                    gid, title, img_data, question, created_at, session_title = all_graphs[idx]
                    with col:
                        st.markdown(f"""
                        <div class="graph-card">
                            <img src="data:image/png;base64,{img_data}" style="width:100%; border-radius: 8px 8px 0 0;" />
                            <div class="graph-meta">
                                <div class="graph-title">📊 {title}</div>
                                <div class="graph-q">❝ {question[:80]}{"…" if len(question)>80 else ""} ❞</div>
                                <div class="graph-q" style="margin-top:6px;">Session: {session_title} · {created_at[:10] if created_at else ""}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        col_dl, col_del = st.columns(2)
                        with col_dl:
                            img_bytes = base64.b64decode(img_data)
                            st.download_button(
                                "⬇ Download",
                                data=img_bytes,
                                file_name=f"chart_{gid}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"dl_{gid}"
                            )
                        with col_del:
                            if st.button("🗑 Delete", key=f"del_graph_{gid}", use_container_width=True):
                                delete_graph(gid)
                                st.rerun()

# ================================================================
# TAB: NOTES
# ================================================================
elif active_tab == "📝 Notes":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
             background: linear-gradient(90deg, #a78bfa, #f472b6); -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">Smart Notes</div>
        <div style="font-size: 13px; color: #4a5568; margin-top: 4px;">Capture insights and key findings from your sessions</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("＋ Add New Note", expanded=False):
        note_title = st.text_input("Title", placeholder="Key finding, insight…", key="note_title_input")
        note_content = st.text_area("Content", placeholder="Write your note here…", height=120, key="note_content_input")

        col_save, col_ai = st.columns(2)
        with col_save:
            if st.button("💾 Save Note", use_container_width=True, key="save_note_btn"):
                if note_title and note_content:
                    save_note(st.session_state.user_id, note_title, note_content)
                    st.success("Note saved!")
                    st.rerun()
        with col_ai:
            if st.button("✨ Summarize with AI", use_container_width=True, key="summarize_note_btn"):
                if note_content:
                    with st.spinner("Summarizing…"):
                        summary = llm.invoke(f"Summarize this concisely in 2-3 bullet points:\n{note_content}").content
                    st.info(summary)

    notes = load_notes(st.session_state.user_id)
    if not notes:
        st.markdown("""
        <div style="text-align:center; padding:60px; color: #2a3550;">
            <div style="font-size: 40px;">📝</div>
            <div style="font-size: 15px; margin-top: 12px; color: #3a4a6a;">No notes yet</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for nid, ntitle, ncontent, ncreated in notes:
            col_n, col_d = st.columns([5, 1])
            with col_n:
                st.markdown(f"""
                <div class="note-card">
                    <div class="note-title">📌 {ntitle}</div>
                    <div class="note-content">{ncontent[:300]}{"…" if len(ncontent)>300 else ""}</div>
                    <div class="note-date">{ncreated[:16] if ncreated else ""}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_d:
                if st.button("🗑", key=f"del_note_{nid}"):
                    delete_note(nid)
                    st.rerun()

# ================================================================
# TAB: ANALYTICS
# ================================================================
elif active_tab == "📈 Analytics":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
             background: linear-gradient(90deg, #4ade80, #4f8ef7); -webkit-background-clip: text;
             -webkit-text-fill-color: transparent;">Usage Analytics</div>
        <div style="font-size: 13px; color: #4a5568; margin-top: 4px;">Your activity overview across all sessions</div>
    </div>
    """, unsafe_allow_html=True)

    sessions = get_sessions(st.session_state.user_id)
    all_msgs = get_all_user_messages(st.session_state.user_id)
    all_graphs = load_all_user_graphs(st.session_state.user_id)
    all_notes = load_notes(st.session_state.user_id)

    total_sessions = len(sessions)
    total_messages = len(all_msgs)
    human_msgs = sum(1 for r, *_ in all_msgs if r == "human")
    ai_msgs = total_messages - human_msgs
    total_graphs = len(all_graphs)
    total_notes = len(all_notes)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{total_sessions}</div>
            <div class="stat-label">Sessions</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{human_msgs}</div>
            <div class="stat-label">Questions Asked</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{ai_msgs}</div>
            <div class="stat-label">AI Responses</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{total_graphs}</div>
            <div class="stat-label">Charts Saved</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{total_notes}</div>
            <div class="stat-label">Notes</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if all_msgs:
        # Messages per session bar chart
        session_counts = {}
        for _, _, _, stitle in all_msgs:
            session_counts[stitle] = session_counts.get(stitle, 0) + 1

        fig_bar, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1528")
        ax.set_facecolor("#080c18")
        bars = ax.barh(list(session_counts.keys()), list(session_counts.values()),
                      color="#4f8ef7", alpha=0.8)
        ax.set_xlabel("Messages", color="#6b7a99", fontsize=11)
        ax.tick_params(colors="#6b7a99", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#1a2545")
        ax.set_title("Messages per Session", color="#c8d4e8", fontsize=13, pad=12)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Message role distribution
        col_l, col_r = st.columns(2)
        with col_l:
            fig_pie, ax2 = plt.subplots(figsize=(4, 4), facecolor="#0d1528")
            ax2.set_facecolor("#0d1528")
            wedge_props = dict(width=0.5, edgecolor="#0d1528", linewidth=2)
            ax2.pie(
                [human_msgs, ai_msgs],
                labels=["You", "AI"],
                colors=["#4f8ef7", "#a78bfa"],
                autopct="%1.0f%%",
                startangle=90,
                wedgeprops=wedge_props,
                textprops={"color": "#c8d4e8", "fontsize": 12}
            )
            ax2.set_title("Message Distribution", color="#c8d4e8", fontsize=13)
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        with col_r:
            # Session modes
            mode_counts = {}
            for _, _, mode, _ in sessions:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if mode_counts:
                fig_m, ax3 = plt.subplots(figsize=(4, 4), facecolor="#0d1528")
                ax3.set_facecolor("#0d1528")
                ax3.pie(
                    list(mode_counts.values()),
                    labels=list(mode_counts.keys()),
                    colors=["#4ade80", "#f472b6", "#60a5fa"],
                    autopct="%1.0f%%",
                    startangle=90,
                    wedgeprops=dict(width=0.5, edgecolor="#0d1528", linewidth=2),
                    textprops={"color": "#c8d4e8", "fontsize": 12}
                )
                ax3.set_title("Session Types", color="#c8d4e8", fontsize=13)
                st.pyplot(fig_m)
                plt.close(fig_m)
    else:
        st.info("Start chatting to see your analytics!")

# ================================================================
# TAB: CUSTOM DASHBOARD
# ================================================================
elif active_tab == "🎛 Dashboard":
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
             background: linear-gradient(90deg, #f472b6, #fbbf24, #4f8ef7);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Custom Dashboard
        </div>
        <div style="font-size: 13px; color: #4a5568; margin-top: 4px;">
            Upload a CSV and build your own charts — drag, configure, and explore
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- File upload (dedicated for dashboard) ----
    dash_file = st.file_uploader(
        "Upload CSV to build your dashboard",
        type=["csv"],
        key="dashboard_csv_uploader"
    )

    if dash_file:
        st.session_state["dash_df"] = pd.read_csv(dash_file)

    dash_df = st.session_state.get("dash_df", None)

    if dash_df is None:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color: #2a3550;">
            <div style="font-size: 52px;">🎛</div>
            <div style="font-size: 16px; font-weight: 600; color: #3a4a6a; margin-top: 16px;">
                Upload a CSV above to begin
            </div>
            <div style="font-size: 13px; margin-top: 8px; color: #2a3a5a;">
                Then add chart widgets, filter rows, and customise your layout
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ---- Dataset overview ----
    num_cols   = dash_df.select_dtypes(include="number").columns.tolist()
    cat_cols   = dash_df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols   = dash_df.columns.tolist()
    n_rows, n_cols = dash_df.shape

    # Stat row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-card"><div class="stat-value">{n_rows:,}</div>
        <div class="stat-label">Rows</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card"><div class="stat-value">{n_cols}</div>
        <div class="stat-label">Columns</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card"><div class="stat-value">{len(num_cols)}</div>
        <div class="stat-label">Numeric</div></div>""", unsafe_allow_html=True)
    with c4:
        missing = int(dash_df.isnull().sum().sum())
        st.markdown(f"""<div class="stat-card"><div class="stat-value">{missing}</div>
        <div class="stat-label">Missing Values</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Global filter row ----
    with st.expander("🔍 Filter Data", expanded=False):
        filter_col = st.selectbox("Filter by column", ["(none)"] + cat_cols + num_cols, key="dash_filter_col")
        if filter_col != "(none)":
            if filter_col in cat_cols:
                unique_vals = dash_df[filter_col].dropna().unique().tolist()
                chosen_vals = st.multiselect(f"Keep rows where {filter_col} is:", unique_vals, default=unique_vals, key="dash_filter_vals")
                dash_df = dash_df[dash_df[filter_col].isin(chosen_vals)]
            else:
                col_min = float(dash_df[filter_col].min())
                col_max = float(dash_df[filter_col].max())
                lo, hi = st.slider(
                    f"Range for {filter_col}",
                    col_min, col_max, (col_min, col_max),
                    key="dash_range_slider"
                )
                dash_df = dash_df[(dash_df[filter_col] >= lo) & (dash_df[filter_col] <= hi)]
        st.caption(f"Showing {len(dash_df):,} of {n_rows:,} rows after filter")

    st.markdown("---")

    # ---- Widget builder ----
    st.markdown('<div class="section-heading">Add Chart Widget</div>', unsafe_allow_html=True)

    if "dashboard_widgets" not in st.session_state:
        st.session_state["dashboard_widgets"] = []

    CHART_TYPES = ["Bar", "Horizontal Bar", "Line", "Area", "Scatter",
                   "Pie / Donut", "Histogram", "Box Plot", "Heatmap (Correlation)"]

    with st.expander("➕ Configure New Widget", expanded=len(st.session_state["dashboard_widgets"]) == 0):
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            w_title  = st.text_input("Widget title", placeholder="My Chart", key="w_title")
            w_type   = st.selectbox("Chart type", CHART_TYPES, key="w_type")
        with wc2:
            if w_type == "Heatmap (Correlation)":
                w_x = None
                w_y = None
                st.info("Correlation heatmap uses all numeric columns automatically.")
            elif w_type in ["Histogram", "Box Plot"]:
                w_x = st.selectbox("Column", num_cols if num_cols else all_cols, key="w_x_hist")
                w_y = None
            elif w_type == "Pie / Donut":
                w_x = st.selectbox("Labels column", cat_cols + all_cols, key="w_x_pie")
                w_y = st.selectbox("Values column (optional)", ["(count)"] + num_cols, key="w_y_pie")
            else:
                w_x = st.selectbox("X-axis", all_cols, key="w_x")
                w_y = st.selectbox("Y-axis", num_cols if num_cols else all_cols, key="w_y")
        with wc3:
            COLOR_PALETTES = {
                "Blue"  : ["#4f8ef7","#3a7ae0","#2563c4","#1a4fa8","#0d3a8c"],
                "Purple": ["#a78bfa","#8b5cf6","#7c3aed","#6d28d9","#5b21b6"],
                "Teal"  : ["#2dd4bf","#14b8a6","#0d9488","#0f766e","#115e59"],
                "Amber" : ["#fbbf24","#f59e0b","#d97706","#b45309","#92400e"],
                "Pink"  : ["#f472b6","#ec4899","#db2777","#be185d","#9d174d"],
                "Multi" : ["#4f8ef7","#a78bfa","#f472b6","#fbbf24","#4ade80","#2dd4bf"],
            }
            w_palette   = st.selectbox("Color palette", list(COLOR_PALETTES.keys()), key="w_palette")
            w_agg       = st.selectbox("Aggregation (bar/line)", ["sum","mean","count","max","min"], key="w_agg")
            w_top_n     = st.number_input("Limit to top N values (0 = all)", min_value=0, max_value=100, value=0, key="w_topn")
            w_sort      = st.checkbox("Sort descending", value=True, key="w_sort")

        if st.button("＋ Add Widget", key="add_widget_btn", use_container_width=True):
            st.session_state["dashboard_widgets"].append({
                "title"  : w_title or f"{w_type} of {w_y or w_x}",
                "type"   : w_type,
                "x"      : w_x,
                "y"      : w_y,
                "palette": w_palette,
                "agg"    : w_agg,
                "top_n"  : w_top_n,
                "sort"   : w_sort,
            })
            st.success("Widget added!")
            st.rerun()

    # ---- Render widgets in 2-column grid ----
    widgets = st.session_state["dashboard_widgets"]

    if widgets:
        st.markdown('<div class="section-heading">Your Dashboard</div>', unsafe_allow_html=True)

        # Clear all button
        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("🗑 Clear All Widgets", key="clear_widgets_btn"):
                st.session_state["dashboard_widgets"] = []
                st.rerun()

        # Render 2 per row
        for i in range(0, len(widgets), 2):
            row_cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx >= len(widgets):
                    break
                w = widgets[idx]
                palette_colors = COLOR_PALETTES.get(w["palette"], COLOR_PALETTES["Blue"])

                with row_cols[j]:
                    st.markdown(f"""
                    <div style="background:#0d1528; border:1px solid #1a2545; border-radius:12px;
                         padding:16px 18px; margin-bottom:4px;">
                        <div style="font-family:'Syne',sans-serif; font-size:14px; font-weight:700;
                             color:#c8d4e8; margin-bottom:2px;">{w['title']}</div>
                        <div style="font-size:11px; color:#2a3550; text-transform:uppercase;
                             letter-spacing:1px;">{w['type']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    try:
                        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0d1528")
                        ax.set_facecolor("#080c18")
                        for spine in ax.spines.values():
                            spine.set_color("#1a2545")
                        ax.tick_params(colors="#6b7a99", labelsize=9)
                        ax.xaxis.label.set_color("#6b7a99")
                        ax.yaxis.label.set_color("#6b7a99")

                        chart_type = w["type"]
                        x_col = w["x"]
                        y_col = w["y"]
                        agg   = w["agg"]
                        top_n = w["top_n"]
                        sort  = w["sort"]
                        c0    = palette_colors[0]

                        if chart_type == "Heatmap (Correlation)":
                            corr = dash_df[num_cols].corr()
                            im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
                            ax.set_xticks(range(len(num_cols)))
                            ax.set_yticks(range(len(num_cols)))
                            ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8, color="#6b7a99")
                            ax.set_yticklabels(num_cols, fontsize=8, color="#6b7a99")
                            for ri in range(len(num_cols)):
                                for ci in range(len(num_cols)):
                                    ax.text(ci, ri, f"{corr.iloc[ri,ci]:.2f}",
                                            ha="center", va="center", fontsize=7, color="#e8eaf0")
                            plt.colorbar(im, ax=ax)

                        elif chart_type == "Histogram":
                            ax.hist(dash_df[x_col].dropna(), bins=20, color=c0, alpha=0.85, edgecolor="#050810")
                            ax.set_xlabel(x_col)
                            ax.set_ylabel("Count")

                        elif chart_type == "Box Plot":
                            bp = ax.boxplot(dash_df[x_col].dropna(), patch_artist=True, notch=False,
                                            medianprops=dict(color="#fbbf24", linewidth=2),
                                            boxprops=dict(facecolor=c0, alpha=0.6),
                                            whiskerprops=dict(color="#6b7a99"),
                                            capprops=dict(color="#6b7a99"),
                                            flierprops=dict(marker="o", color=c0, alpha=0.5, markersize=4))
                            ax.set_xticklabels([x_col])

                        elif chart_type == "Scatter":
                            ax.scatter(dash_df[x_col], dash_df[y_col], color=c0, alpha=0.6, s=18, edgecolors="none")
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)

                        elif chart_type == "Pie / Donut":
                            if y_col and y_col != "(count)":
                                grp = dash_df.groupby(x_col)[y_col].sum()
                            else:
                                grp = dash_df[x_col].value_counts()
                            if top_n:
                                grp = grp.nlargest(top_n)
                            colors_pie = (palette_colors * 10)[:len(grp)]
                            wedge_props = dict(width=0.55, edgecolor="#050810", linewidth=2)
                            ax.pie(grp.values, labels=grp.index, colors=colors_pie,
                                   autopct="%1.1f%%", startangle=90, wedgeprops=wedge_props,
                                   textprops={"color": "#c8d4e8", "fontsize": 9})

                        else:
                            # Bar, H-Bar, Line, Area all use groupby
                            if y_col and y_col != "(count)":
                                if agg == "sum":
                                    grp = dash_df.groupby(x_col)[y_col].sum()
                                elif agg == "mean":
                                    grp = dash_df.groupby(x_col)[y_col].mean()
                                elif agg == "count":
                                    grp = dash_df.groupby(x_col)[y_col].count()
                                elif agg == "max":
                                    grp = dash_df.groupby(x_col)[y_col].max()
                                else:
                                    grp = dash_df.groupby(x_col)[y_col].min()
                            else:
                                grp = dash_df[x_col].value_counts()

                            if sort:
                                grp = grp.sort_values(ascending=False)
                            if top_n:
                                grp = grp.head(top_n)

                            x_labels = [str(v)[:14] for v in grp.index]
                            colors_bar = (palette_colors * 10)[:len(grp)]

                            if chart_type == "Bar":
                                ax.bar(x_labels, grp.values, color=colors_bar, alpha=0.85, edgecolor="#050810", linewidth=0.5)
                                plt.xticks(rotation=35, ha="right", fontsize=8)
                                ax.set_ylabel(y_col or "Count")
                            elif chart_type == "Horizontal Bar":
                                ax.barh(x_labels[::-1], grp.values[::-1], color=colors_bar[::-1], alpha=0.85, edgecolor="#050810", linewidth=0.5)
                                ax.set_xlabel(y_col or "Count")
                            elif chart_type == "Line":
                                ax.plot(x_labels, grp.values, color=c0, linewidth=2, marker="o", markersize=4, markerfacecolor=palette_colors[1] if len(palette_colors)>1 else c0)
                                ax.fill_between(range(len(x_labels)), grp.values, alpha=0.08, color=c0)
                                plt.xticks(rotation=35, ha="right", fontsize=8)
                                ax.set_ylabel(y_col or "Count")
                            elif chart_type == "Area":
                                ax.fill_between(range(len(x_labels)), grp.values, alpha=0.35, color=c0)
                                ax.plot(range(len(x_labels)), grp.values, color=c0, linewidth=1.5)
                                ax.set_xticks(range(len(x_labels)))
                                ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
                                ax.set_ylabel(y_col or "Count")

                        plt.tight_layout(pad=0.5)
                        st.pyplot(fig)

                        # Save + Download
                        img_b64 = fig_to_base64(fig)
                        plt.close(fig)

                        col_sv, col_dl, col_rm = st.columns(3)
                        with col_sv:
                            if st.button("💾 Save", key=f"save_w_{idx}", use_container_width=True):
                                save_graph(
                                    st.session_state.active_session_id,
                                    img_b64,
                                    f"Dashboard: {w['title']}",
                                    w["title"]
                                )
                                st.success("Saved to Gallery!")
                        with col_dl:
                            st.download_button(
                                "⬇ PNG",
                                data=base64.b64decode(img_b64),
                                file_name=f"{w['title'].replace(' ','_')}.png",
                                mime="image/png",
                                key=f"dl_w_{idx}",
                                use_container_width=True
                            )
                        with col_rm:
                            if st.button("✕ Remove", key=f"rm_w_{idx}", use_container_width=True):
                                st.session_state["dashboard_widgets"].pop(idx)
                                st.rerun()

                    except Exception as e:
                        st.error(f"Chart error: {e}")

    # ---- Data preview + describe ----
    st.markdown("---")
    st.markdown('<div class="section-heading">Data Preview</div>', unsafe_allow_html=True)
    tab_prev, tab_desc, tab_dtypes = st.tabs(["📋 Table", "📊 Statistics", "🔢 Column Types"])

    with tab_prev:
        n_preview = st.slider("Rows to preview", 5, min(200, len(dash_df)), 10, key="preview_slider")
        st.dataframe(dash_df.head(n_preview), use_container_width=True, height=280)

    with tab_desc:
        desc = dash_df.describe().round(3)
        st.dataframe(desc, use_container_width=True, height=280)

    with tab_dtypes:
        dtype_df = pd.DataFrame({
            "Column": dash_df.dtypes.index,
            "Type": dash_df.dtypes.astype(str).values,
            "Non-Null": dash_df.count().values,
            "Null": dash_df.isnull().sum().values,
            "Unique": [dash_df[c].nunique() for c in dash_df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, height=280)


# ================================================================
# TAB: SETTINGS
# ================================================================
elif active_tab == "⚙️ Settings":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800; color: #e8eaf0;">Settings</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🗂 Session Management", expanded=True):
        sessions = get_sessions(st.session_state.user_id)
        if sessions:
            sess_options = {f"{title} (id:{sid})": sid for sid, title, _, _ in sessions}
            rename_target = st.selectbox("Select session to rename", list(sess_options.keys()), key="rename_select")
            new_name = st.text_input("New name", key="rename_input")
            if st.button("✏️ Rename", key="rename_btn"):
                if new_name.strip():
                    rename_session(sess_options[rename_target], new_name.strip())
                    st.success("Renamed!")
                    st.rerun()

    with st.expander("🗑 Danger Zone"):
        st.warning("Deleting a session removes all its messages and charts permanently.")
        sessions = get_sessions(st.session_state.user_id)
        if sessions:
            del_options = {f"{title}": sid for sid, title, _, _ in sessions}
            del_target = st.selectbox("Select session to delete", list(del_options.keys()), key="del_select")
            if st.button("🗑 Delete Session", key="danger_del_btn"):
                delete_session(del_options[del_target])
                st.success("Session deleted.")
                st.session_state.active_session_id = None
                st.rerun()

    with st.expander("📋 Export Chat History"):
        sessions = get_sessions(st.session_state.user_id)
        if sessions:
            exp_options = {f"{title}": sid for sid, title, _, _ in sessions}
            exp_target = st.selectbox("Select session to export", list(exp_options.keys()), key="exp_select")
            if st.button("⬇ Export as CSV", key="export_btn"):
                msgs = load_messages(exp_options[exp_target])
                df_exp = pd.DataFrame(msgs, columns=["Role", "Message", "Timestamp"])
                csv_data = df_exp.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=f"chat_export_{exp_target}.csv",
                    mime="text/csv",
                    key="download_export_btn"
                )
