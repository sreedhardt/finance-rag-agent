"""Streamlit chat UI for the Finance RAG Agent."""

import streamlit as st

st.set_page_config(page_title="Finance RAG Agent", page_icon="📊", layout="wide")

SAMPLE_QUESTIONS = [
    "Did FY2025 spend with Helios Foundry meet the minimum purchase commitment in the supply agreement?",
    "What payment terms and liability cap did we agree to with Helios Foundry?",
    "How did Data Center revenue grow across FY2025, and what does the 10-K say drove it?",
    "What is our effective tax rate and why is it below the statutory rate?",
    "What connects Helios Foundry to our SEC filing risk factors?",
]


@st.cache_resource
def get_runtime():
    from finance_rag.agent import create_agent
    from finance_rag.embedder import GeminiEmbedder
    from finance_rag.graph import KnowledgeGraph
    from finance_rag.indexer import VectorIndex
    from finance_rag.tools import AgentTools

    index = VectorIndex(GeminiEmbedder())
    agent = create_agent(AgentTools(index, KnowledgeGraph.load()))
    return index, agent


def md(text: str) -> str:
    """Escape $ so st.markdown doesn't treat dollar amounts as LaTeX math
    delimiters ($1,220 ... $1,200 would otherwise render as a garbled formula)."""
    return text.replace("$", r"\$")


def render_steps(steps):
    if not steps:
        return
    with st.expander(f"🔍 Agent reasoning trace ({len(steps)} tool calls)"):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}. `{step.tool}`** — `{step.args}`")
            st.json(step.result, expanded=False)


st.title("📊 Finance Document Intelligence Agent")
st.caption(
    "Agentic RAG over SEC filings, contracts, and tax reports — with citations, "
    "a knowledge graph, and SQL cross-checks against the structured ledger."
)

index, agent = get_runtime()

with st.sidebar:
    from finance_rag import config
    active_model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.GEMINI_MODEL
    st.caption(f"LLM: **{config.LLM_PROVIDER}** / `{active_model}`")
    st.header("Corpus")
    if st.button("Rebuild index (ingest pipeline)"):
        from finance_rag.pipeline import run_ingestion
        with st.spinner("Running ingestion pipeline..."):
            manifest = run_ingestion()
        st.success(f"Indexed {manifest['chunks_indexed']} chunks "
                   f"from {len(manifest['documents'])} docs")
        st.cache_resource.clear()
        st.rerun()
    for doc in index.list_documents():
        st.markdown(f"- **{doc['title']}** ({doc['doc_type']}, {doc['n_chunks']} chunks)")
    st.divider()
    st.header("Try asking")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=q):
            st.session_state.pending_question = q

if index.count() == 0:
    st.warning("Index is empty — click **Rebuild index** in the sidebar first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(md(msg["content"]))
        if msg.get("steps"):
            render_steps(msg["steps"])

question = st.chat_input("Ask about the documents or financials...")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        with st.spinner("Investigating..."):
            result = agent.ask(question)
        st.markdown(md(result.answer))
        render_steps(result.steps)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "steps": result.steps}
    )
