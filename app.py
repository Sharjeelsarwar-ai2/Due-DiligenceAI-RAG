import os
import pickle

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
    initial_sidebar_state="collapsed",
)


# ============================================================
# ENTERPRISE UI
# ============================================================

st.markdown(
    """
<style>
:root {
    --ink: #10243a;
    --ink-soft: #29425a;
    --muted: #718197;
    --line: rgba(139, 158, 178, .23);
    --paper: rgba(255,255,255,.88);
    --navy: #10243a;
    --navy-2: #1b3b56;
    --gold: #c49a55;
    --gold-light: #eed9a9;
    --green: #1d9b79;
    --canvas: #f3f6f8;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
.stApp {
    background:
        radial-gradient(circle at 4% 0%, rgba(196,154,85,.13), transparent 24%),
        radial-gradient(circle at 100% 9%, rgba(43,101,137,.11), transparent 27%),
        linear-gradient(180deg, #f8fafb 0%, var(--canvas) 100%);
    color: var(--ink);
}
.block-container {
    max-width: 1280px;
    padding: 1.15rem 2.1rem 5.5rem;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }

.topbar {
    display:flex; justify-content:space-between; align-items:center;
    padding: 4px 2px 22px;
}
.brand { display:flex; align-items:center; gap:12px; }
.brand-mark {
    width:42px; height:42px; border-radius:14px; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(145deg, #0e2238, #2b506d); color:var(--gold-light); font-size:18px;
    box-shadow:0 12px 28px rgba(16,36,58,.18);
}
.brand-title { color:var(--ink); font-weight:850; letter-spacing:-.035em; font-size:1rem; }
.brand-sub { color:var(--muted); font-size:.69rem; letter-spacing:.08em; text-transform:uppercase; margin-top:3px; }
.status-chip {
    display:inline-flex; align-items:center; gap:8px; padding:8px 13px; border-radius:999px;
    color:#176b59; background:rgba(29,155,121,.10); border:1px solid rgba(29,155,121,.18);
    font-size:.75rem; font-weight:750;
}
.status-dot { width:7px; height:7px; border-radius:50%; background:#2ac398; box-shadow:0 0 0 4px rgba(42,195,152,.12); }

.hero {
    position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.8); border-radius:30px;
    padding:40px 44px 37px; margin:0 0 22px;
    background:linear-gradient(132deg, rgba(13,31,50,.99), rgba(30,64,87,.97));
    box-shadow:0 26px 70px rgba(16,36,58,.18);
}
.hero:before { content:""; position:absolute; width:330px; height:330px; right:-110px; top:-135px; border-radius:50%; border:1px solid rgba(238,217,169,.20); box-shadow:0 0 0 38px rgba(238,217,169,.035), 0 0 0 76px rgba(238,217,169,.025); }
.hero:after { content:""; position:absolute; width:180px; height:180px; left:-90px; bottom:-115px; border-radius:50%; background:rgba(41,119,143,.19); }
.eyebrow { position:relative; z-index:1; color:var(--gold-light); font-size:.69rem; letter-spacing:.18em; font-weight:850; text-transform:uppercase; }
.hero h1 { position:relative; z-index:1; max-width:760px; color:#fff; font-size:clamp(2rem, 4vw, 3.35rem); line-height:1.06; letter-spacing:-.06em; margin:12px 0 13px; }
.hero p { position:relative; z-index:1; max-width:710px; color:#c9d7e2; line-height:1.7; margin:0; font-size:.96rem; }
.hero-footer { position:relative; z-index:1; display:flex; align-items:center; gap:15px; margin-top:23px; }
.hero-label { color:#aebfcd; font-size:.74rem; }

.workspace-label { color:var(--ink-soft); font-size:.73rem; font-weight:850; letter-spacing:.13em; text-transform:uppercase; margin:27px 0 11px; }
.chat-shell {
    border:1px solid rgba(139,158,178,.23); border-radius:27px; background:rgba(255,255,255,.58);
    padding:20px 22px 9px; min-height:160px; box-shadow:0 14px 40px rgba(16,36,58,.055);
}
[data-testid="stChatMessage"] {
    border:1px solid rgba(139,158,178,.16); border-radius:20px !important; padding:17px 19px !important;
    margin:10px 0 !important; background:rgba(255,255,255,.82); box-shadow:0 7px 22px rgba(16,36,58,.045);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:#2a3e51; line-height:1.68; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:#edf3f7; border-color:#dce7ee; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { border-left:3px solid var(--gold); }
[data-testid="stChatInput"] {
    position:sticky !important; bottom:18px !important; z-index:30 !important;
    width:min(760px, calc(100% - 28px)) !important; margin:22px auto 8px !important;
    border:1px solid rgba(155,207,225,.46) !important; border-radius:22px !important;
    background:linear-gradient(135deg, rgba(10,31,52,.98), rgba(22,70,91,.97) 55%, rgba(31,50,84,.98)) !important;
    box-shadow:0 18px 42px rgba(10,35,57,.25), 0 0 0 5px rgba(255,255,255,.35), inset 0 1px 0 rgba(255,255,255,.16) !important;
    padding:5px !important;
}
[data-testid="stChatInput"] > div {
    border:1px solid rgba(189,226,237,.16) !important; border-radius:16px !important;
    background:linear-gradient(105deg, rgba(255,255,255,.10), rgba(255,255,255,.045)) !important;
}
[data-testid="stChatInput"] textarea { color:#edf8fb !important; caret-color:#91e7da !important; font-size:.90rem !important; }
[data-testid="stChatInput"] textarea::placeholder { color:rgba(224,240,246,.68) !important; }
[data-testid="stChatInput"] button {
    color:#0d3048 !important; background:linear-gradient(145deg, #b7f1df, #78d8d0) !important;
    border:1px solid rgba(255,255,255,.5) !important; border-radius:12px !important;
    box-shadow:0 5px 16px rgba(93,222,199,.23) !important;
}
[data-testid="stChatInput"] button:hover { transform:translateY(-1px); filter:brightness(1.06); }

.quick-grid { display:flex; gap:9px; flex-wrap:wrap; margin:0 0 15px; }
.quick-caption { color:var(--muted); font-size:.77rem; margin:1px 0 9px; }
button[kind="secondary"] { border-radius:12px !important; border-color:#d4dfe6 !important; color:var(--ink-soft) !important; background:rgba(255,255,255,.72) !important; font-size:.78rem !important; }
button[kind="secondary"]:hover { border-color:var(--gold) !important; color:var(--navy) !important; }

.source-card { border:1px solid #e0e7ec; border-radius:17px; padding:15px 17px; background:rgba(255,255,255,.94); margin:9px 0; box-shadow:0 7px 22px rgba(15,31,51,.045); }
.source-name { color:var(--ink); font-weight:800; }
.source-meta { color:var(--muted); font-size:.78rem; margin-top:5px; }
.info-card { border-left:4px solid var(--gold); border-radius:14px; padding:14px 16px; background:#fffaf1; margin:8px 0; }
.stExpander { border-color:rgba(139,158,178,.22) !important; border-radius:15px !important; background:rgba(255,255,255,.43) !important; }
div[data-testid="stPopover"] > button {
    border:1px solid rgba(196,154,85,.42) !important; border-radius:999px !important;
    background:linear-gradient(135deg, rgba(255,250,237,.96), rgba(241,247,249,.96)) !important;
    color:#76582a !important; font-size:.73rem !important; font-weight:800 !important;
    padding:6px 12px !important; min-height:0 !important; margin-top:10px !important;
    box-shadow:0 5px 16px rgba(16,36,58,.07) !important;
}
div[data-testid="stPopover"] > button:hover { border-color:var(--gold) !important; transform:translateY(-1px); }
[data-testid="stPopoverBody"] { border-radius:18px !important; }
.citation-intro { color:var(--muted); font-size:.75rem; line-height:1.5; margin-bottom:10px; }
.citation-index { display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:7px; background:#e8f4f1; color:#137c68; font-size:.72rem; font-weight:850; margin-right:7px; }

.float-rail {
    position:fixed; z-index:20; right:22px; top:44%; transform:translateY(-50%); display:flex; flex-direction:column; gap:8px;
    padding:8px; border:1px solid rgba(255,255,255,.78); border-radius:18px; background:rgba(255,255,255,.72);
    box-shadow:0 16px 34px rgba(16,36,58,.13); backdrop-filter:blur(14px);
}
.float-rail a { width:34px; height:34px; display:flex; align-items:center; justify-content:center; border-radius:11px; color:#648098; text-decoration:none; font-size:15px; transition:.2s ease; }
.float-rail a:hover { color:white; background:var(--navy); transform:translateY(-1px); }
.footer-note { color:#8190a0; text-align:center; font-size:.72rem; margin-top:34px; }
@media (max-width: 760px) {
    .block-container { padding: .8rem 1rem 4.5rem; }
    .hero { padding:28px 24px; border-radius:23px; }
    .hero h1 { font-size:2.15rem; }
    .topbar { padding-bottom:16px; }
    .float-rail { right:10px; top:auto; bottom:17px; transform:none; flex-direction:row; border-radius:16px; }
    .chat-shell { padding:12px 10px 5px; border-radius:21px; }
    [data-testid="stChatInput"] { width:calc(100% - 12px) !important; bottom:10px !important; margin-top:16px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


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
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
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
<div class="source-name">▣ {x['source_file']}</div>
<div class="source-meta">{x['category']} · Page {x['page']} · Version {x.get('version','Unknown')} · Relevance {x['score']:.3f}</div>
<div class="source-meta">{x.get('source_path','')}</div>
<div style="margin-top:8px;color:#475569;font-size:.84rem;line-height:1.55">{excerpt}</div>
</div>""",
            unsafe_allow_html=True,
        )


def render_citation_popover(results):
    """Render a compact, in-context source citation popover for an answer."""
    with st.popover(f"⌁  Sources cited · {len(results)}"):
        st.markdown(
            '<div class="citation-intro">The answer above was generated from these retrieved records. Open each source card to verify the underlying evidence.</div>',
            unsafe_allow_html=True,
        )
        for number, x in enumerate(results, 1):
            excerpt = " ".join(x["text"].split())
            if len(excerpt) > 240:
                excerpt = excerpt[:240] + "..."
            st.markdown(
                f"""<div class="source-card">
<div class="source-name"><span class="citation-index">{number}</span>{x['source_file']}</div>
<div class="source-meta">{x['category']} · Page {x['page']} · Relevance {x['score']:.3f}</div>
<div style="margin-top:7px;color:#475569;font-size:.81rem;line-height:1.5">{excerpt}</div>
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
# APPLICATION SHELL
# ============================================================

st.markdown(
    """<div class="topbar" id="top">
<div class="brand"><div class="brand-mark">◆</div><div><div class="brand-title">Property Intelligence</div><div class="brand-sub">Enterprise Due Diligence</div></div></div>
<div class="status-chip"><span class="status-dot"></span> Knowledge base connected</div>
</div>
<div class="float-rail" aria-label="Quick navigation">
<a href="#top" title="Overview">⌂</a><a href="#workspace" title="Workspace">✦</a><a href="#evidence" title="Evidence">▤</a>
</div>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="hero">
<div class="eyebrow">REAL ESTATE · DUE DILIGENCE</div>
<h1>Property intelligence, grounded in your documents.</h1>
<p>Search sale agreements, title records, tax statements, leases, inspections, approvals and legal disclosures with an AI assistant that keeps every finding tied to its source.</p>
<div class="hero-footer"><span class="status-dot"></span><span class="hero-label">Source-backed retrieval is active</span></div>
</div>""",
    unsafe_allow_html=True,
)

st.markdown('<div class="workspace-label" id="workspace">Due-diligence workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="quick-caption">Start with a focused review or ask your own question below.</div>', unsafe_allow_html=True)

prompts = [
    ("Liabilities", "Are there any documents indicating outstanding liabilities associated with this property?"),
    ("Title & liens", "What liens, mortgages, assessments, or encumbrances are identified?"),
    ("Missing documents", "Which important due-diligence documents are missing or still unresolved?"),
    ("Lease review", "What material lease obligations or restrictions should the buyer know about?"),
]
quick_cols = st.columns(4)
for col, (label, prompt) in zip(quick_cols, prompts):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.pending_question = prompt

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-shell" id="evidence">', unsafe_allow_html=True)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            render_citation_popover(m["sources"])

question = st.chat_input("Ask a property due-diligence question...")

if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
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
            render_citation_popover(results)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-note">Analysis is grounded in the connected property repository. Verify material findings against current originals before a transaction decision.</div>', unsafe_allow_html=True)
