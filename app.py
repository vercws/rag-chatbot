import streamlit as st
from rag_engine import process_pdfs, build_chain

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Chatbot",
    page_icon="🔬",
    layout="wide"
)

# ── API key ────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("ANTHROPIC_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

# ── Session state defaults ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research RAG")
    st.caption("Upload research papers and ask questions in natural language.")

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once."
    )

    process_btn = st.button(
        "Process Papers",
        type="primary",
        disabled=not uploaded_files
    )

    if process_btn and uploaded_files:
        file_names = [f.name for f in uploaded_files]
        if file_names != st.session_state.processed_files:
            with st.spinner("Reading and indexing papers… this may take a minute."):
                try:
                    vector_store = process_pdfs(uploaded_files)
                    st.session_state.chain = build_chain(vector_store, API_KEY)
                    st.session_state.processed_files = file_names
                    st.session_state.messages = []  # Reset chat on new upload
                    st.success(f"✅ {len(uploaded_files)} paper(s) indexed!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")
        else:
            st.info("These papers are already loaded.")

    if st.session_state.processed_files:
        st.divider()
        st.markdown("**Loaded papers:**")
        for name in st.session_state.processed_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.processed_files = []
        st.rerun()

# ── Main chat area ─────────────────────────────────────────────
st.title("Ask your research papers anything")

if not st.session_state.chain:
    st.info("👈 Upload one or more research PDFs in the sidebar and click **Process Papers** to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.caption(f"**{src['file']}** — page {src['page']}")
                        st.markdown(f"> {src['snippet']}")

    # Chat input
    if prompt := st.chat_input("e.g. What methodology did the authors use?"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer from chain
        with st.chat_message("assistant"):
            with st.spinner("Searching papers…"):
                try:
                    result = st.session_state.chain.invoke({"question": prompt})
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])

                    # Deduplicate sources by (filename, page)
                    seen = set()
                    sources = []
                    for doc in source_docs:
                        key = (doc.metadata.get("source", ""), doc.metadata.get("page", ""))
                        if key not in seen:
                            seen.add(key)
                            sources.append({
                                "file": doc.metadata.get("source", "unknown"),
                                "page": doc.metadata.get("page", "?"),
                                "snippet": doc.page_content[:300] + "…"
                            })

                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Sources"):
                            for src in sources:
                                st.caption(f"**{src['file']}** — page {src['page']}")
                                st.markdown(f"> {src['snippet']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")