# RAG Chatbot for Research Papers

A conversational AI tool that lets you upload PDF research papers and ask questions about them in natural language. Built with LangChain, FAISS, HuggingFace embeddings, and Claude as the LLM — all wrapped in a Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.2-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet-blueviolet)

---

## How it works

1. Upload one or more PDF papers via the sidebar
2. The app chunks the text, generates embeddings (via `all-MiniLM-L6-v2`), and stores them in a FAISS vector index
3. You type a question in the chat — the most relevant chunks are retrieved and passed to Claude Sonnet as context
4. The answer is returned with source citations (file name + page number)

Conversation history is maintained across turns using LangChain's `ConversationBufferMemory`.

---

## Tech stack

| Component | Tool |
|---|---|
| LLM | Anthropic Claude Sonnet (`claude-sonnet-4-6`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, CPU) |
| Vector store | FAISS |
| Orchestration | LangChain `ConversationalRetrievalChain` |
| PDF parsing | LangChain `PyPDFLoader` |
| UI | Streamlit |

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/vercws/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

Create a `.streamlit/secrets.toml` file:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Project structure

```
rag-chatbot/
├── app.py            # Streamlit UI and chat logic
├── rag_engine.py     # PDF processing, embeddings, and chain setup
└── requirements.txt
```

---

## Configuration

Key parameters in `rag_engine.py` you can tune:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 800 | Token size per text chunk |
| `chunk_overlap` | 100 | Overlap between chunks |
| `k` (retriever) | 4 | Number of chunks retrieved per query |
| `temperature` | 0.2 | LLM response randomness |
| `max_tokens` | 1024 | Max tokens in each LLM response |
