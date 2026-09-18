import os
import pickle
from collections import Counter

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INDEX_FILE = "faiss_index/index.faiss"
METADATA_FILE = "faiss_index/metadata.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 7
MIN_RELEVANCE = 0.22


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Property Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENTERPRISE UI
# ============================================================

st.markdown("""
<style>
:root {
    --ink: #132238;
    --muted: #66758a;
    --line: rgba(148,163,184,.20);
    --paper: rgba(255,255,255,.82);
    --navy: #0f1f33;
    --gold: #b88a3b;
    --gold-soft: #f4ead8;
    --green: #138a68;
    --red: #c85b5b;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(184,138,59,.11), transparent 27%),
        radial-gradient(circle at 94% 4%, rgba(27,70,105,.10), transparent 30%),
        linear-gradient(180deg, #f7f8fa 0%, #eef2f5 100%);
}

.block-container {
    max-width: 1240px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: rgba(250,251,252,.92);
    border-right: 1px solid var(--line);
}

.brand {
    display:flex;
    align-items:center;
    gap:11px;
    padding: 4px 0 14px;
}

.brand-mark {
    width:42px;
    height:42px;
    border-radius:14px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(145deg,#10233b,#294761);
    color:#e9c982;
    font-size:20px;
    box-shadow:0 12px 28px rgba(15,31,51,.18);
}

.brand-title {
    font-weight:800;
    color:var(--ink);
    letter-spacing:-.025em;
}

.brand-sub {
    color:var(--muted);
    font-size:.72rem;
    margin-top:1px;
}

.hero {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.75);
    border-radius:28px;
    padding:35px 38px;
    margin:8px 0 22px;
    background:
        linear-gradient(135deg, rgba(15,31,51,.98), rgba(32,57,77,.96));
    box-shadow:0 24px 65px rgba(15,31,51,.17);
}

.hero:after {
    content:"";
    position:absolute;
    width:260px;
    height:260px;
    right:-80px;
    top:-100px;
    border-radius:50%;
    background:rgba(218,178,98,.13);
}

.eyebrow {
    color:#e2c17d;
    font-size:.73rem;
    letter-spacing:.16em;
    font-weight:800;
    text-transform:uppercase;
}

.hero h1 {
    position:relative;
    z-index:1;
    color:white;
    font-size:2.55rem;
    letter-spacing:-.055em;
    margin:8px 0 8px;
}

.hero p {
    position:relative;
    z-index:1;
    color:#cbd6e1;
    max-width:760px;
    line-height:1.65;
    margin:0;
}

.pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    margin-top:19px;
    padding:7px 12px;
    border-radius:999px;
    color:#d7f6e9;
    background:rgba(20,138,104,.18);
    border:1px solid rgba(135,230,196,.18);
    font-size:.78rem;
    font-weight:700;
}

.dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:#44d3a4;
}

.metric-card {
    border:1px solid var(--line);
    border-radius:20px;
    background:var(--paper);
    padding:18px 19px;
    box-shadow:0 10px 35px rgba(15,31,51,.055);
}

.metric-label {
    color:var(--muted);
    font-size:.74rem;
    text-transform:uppercase;
    letter-spacing:.09em;
    font-weight:800;
}

.metric-value {
    color:var(--ink);
    font-size:1.55rem;
    font-weight:800;
    margin-top:4px;
}

.section-title {
    color:var(--ink);
    font-size:1.02rem;
    font-weight:800;
    margin:18px 0 8px;
}

.source-card {
    border:1px solid #e1e6eb;
    border-radius:17px;
    padding:15px 17px;
    background:rgba(255,255,255,.92);
    margin:9px 0;
    box-shadow:0 7px 22px rgba(15,31,51,.045);
}

.source-name {
    color:var(--ink);
    font-weight:800;
}

.source-meta {
    color:var(--muted);
    font-size:.78rem;
    margin-top:5px;
}

.risk-card {
    border-left:4px solid #c85b5b;
    border-radius:14px;
    padding:14px 16px;
    background:#fff7f7;
    margin:8px 0;
}

.info-card {
    border-left:4px solid #b88a3b;
    border-radius:14px;
    padding:14px 16px;
    background:#fffaf1;
    margin:8px 0;
}

.stChatMessage {
    border-radius:18px;
}

div[data-testid="stChatInput"] {
    border-radius:18px;
}

button[kind="primary"] {
    background:#132238;
    border-color:#132238;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# RESOURCES
# ============================================================

@st.cache_resource
def load_resources():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("faiss_index/index.faiss is missing.")
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("faiss_index/metadata.pkl is missing.")
    if "GROQ_API_KEY" not in st.secrets:
        raise ValueError("GROQ_API_KEY is missing from Streamlit Secrets.")

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    return index, metadata, model, client


def retrieve(question, index, metadata, model):
    query = model.encode([question], normalize_embeddings=True)
    query = np.asarray(query, dtype="float32")
    scores, positions = index.search(query, TOP_K)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos >= 0 and float(score) >= MIN_RELEVANCE:
            item = metadata[pos].copy()
            item["score"] = float(score)
            results.append(item)

    return results


def context_for_llm(results):
    return "\n\n".join(
        f"""SOURCE {i}
File: {x['source_file']}
Category: {x['category']}
Page: {x['page']}
Version: {x.get('version','Unknown')}
Record Date: {x.get('record_date','Unknown')}
Content:
{x['text']}"""
        for i, x in enumerate(results, 1)
    )


def ask_llm(question, results, client):
    system = """You are a real-estate due-diligence knowledge assistant.
You analyze only the supplied property repository.

Rules:
- Use ONLY the retrieved documents.
- Do not invent legal, financial, title, tax, lease, inspection, or regulatory facts.
- Distinguish documented facts from missing information and unresolved matters.
- Identify outstanding liabilities, liens, mortgages, assessments, unpaid taxes, lease obligations, and other material risk indicators when supported.
- If documents conflict, explicitly identify the conflict.
- If a requested fact is absent, say it is not established by the available documents.
- Cite exact filenames and page numbers in the answer.
- Never present this analysis as legal advice.
- For due-diligence questions, organize the answer clearly with findings, evidence, and open items."""

    user = f"""PROPERTY DUE-DILIGENCE QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context_for_llm(results)}

Produce a concise, source-backed due-diligence response."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user},
        ],
        temperature=0.1,
        max_tokens=1600,
    )
    return response.choices[0].message.content


def render_sources(results):
    for x in results:
        excerpt = " ".join(x["text"].split())
        if len(excerpt) > 320:
            excerpt = excerpt[:320] + "..."
        st.markdown(
            f"""<div class="source-card">
<div class="source-name">📄 {x['source_file']}</div>
<div class="source-meta">{x['category']} · Page {x['page']} · Version {x.get('version','Unknown')} · Relevance {x['score']:.3f}</div>
<div class="source-meta">{x.get('source_path','')}</div>
<div style="margin-top:8px;color:#475569;font-size:.84rem;line-height:1.55">{excerpt}</div>
</div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD
# ============================================================

try:
    index, metadata, embedding_model, groq_client = load_resources()
except Exception as e:
    st.error(f"Knowledge base error: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """<div class="brand">
<div class="brand-mark">◆</div>
<div>
<div class="brand-title">Property Intelligence</div>
<div class="brand-sub">Enterprise Due Diligence</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="metric-label">Repository status</div>', unsafe_allow_html=True)
    st.success("Knowledge base connected")

    categories = sorted({x.get("category", "General") for x in metadata})
    files = sorted({x.get("source_file", "Unknown") for x in metadata})

    st.metric("Documents", len(files))
    st.metric("Indexed chunks", f"{index.ntotal:,}")

    st.markdown('<div class="metric-label">Document categories</div>', unsafe_allow_html=True)
    for c in categories:
        st.caption(f"• {c}")

    st.divider()
    st.caption("Source-backed analysis only. Verify material findings against current originals before a transaction decision.")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """<div class="hero">
<div class="eyebrow">REAL ESTATE · DUE DILIGENCE</div>
<h1>Property Intelligence, grounded in your documents.</h1>
<p>Search sale agreements, title records, tax statements, leases, inspections, approvals and legal disclosures with an AI assistant that keeps every finding tied to its source.</p>
<div class="pill"><span class="dot"></span> Repository online · Source-backed retrieval</div>
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Documents</div><div class="metric-value">{len(files)}</div></div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Indexed chunks</div><div class="metric-value">{index.ntotal:,}</div></div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Categories</div><div class="metric-value">{len(categories)}</div></div>""", unsafe_allow_html=True)

with c4:
    st.markdown("""<div class="metric-card"><div class="metric-label">Analysis mode</div><div class="metric-value">Grounded</div></div>""", unsafe_allow_html=True)


# ============================================================
# QUICK PROMPTS
# ============================================================

st.markdown('<div class="section-title">Due-diligence workspace</div>', unsafe_allow_html=True)

q1, q2, q3, q4 = st.columns(4)

prompts = [
    ("Liabilities", "Are there any documents indicating outstanding liabilities associated with this property?"),
    ("Title & liens", "What liens, mortgages, assessments, or encumbrances are identified?"),
    ("Missing docs", "Which important due-diligence documents are missing or still unresolved?"),
    ("Lease", "What material lease obligations or restrictions should the buyer know about?"),
]

for col, (label, prompt) in zip([q1,q2,q3,q4], prompts):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.pending_question = prompt


# ============================================================
# CHAT
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("Evidence & source documents"):
                render_sources(m["sources"])

question = st.chat_input("Ask a property due-diligence question...")

if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role":"user","content":question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the property repository..."):
            results = retrieve(question, index, metadata, embedding_model)

        if not results:
            answer = "I could not find sufficiently relevant evidence in the indexed property repository."
        else:
            with st.spinner("Preparing source-backed analysis..."):
                answer = ask_llm(question, results, groq_client)

        st.markdown(answer)

        if results:
            with st.expander(f"Evidence & source documents · {len(results)} retrieved"):
                render_sources(results)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer,
        "sources":results
    })
