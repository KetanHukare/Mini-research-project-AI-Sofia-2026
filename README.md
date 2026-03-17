# MRP-AI — RAG Q&A Bot

A Retrieval-Augmented Generation (RAG) Q&A system built as a Mini Research Problem (MRP) for Sofia University. Drop documents into a folder, build a local vector index, then ask questions — grounded in your own documents, not hallucinated.

---

## How it works

```
docs/ (PDFs, TXT, MD, DOCX)
        │
        ▼
  part1_ingest.py  →  FAISS vector index  (data/faiss_index/)
                                │
                                ▼
                      part2_rag.py  →  terminal Q&A
```

1. **Part 1** scans `docs/`, splits documents into chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and saves a FAISS index to disk.
2. **Part 2** loads that index, retrieves the top-K most relevant chunks for each question, and passes them as context to an LLM.

---

## Quick start

### 1. Clone & create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# then edit .env — see Configuration section below
```

### 3. Add your documents

Drop any `.pdf`, `.txt`, `.md`, or `.docx` files into `docs/`.

### 4. Run Part 1 — build the index

```bash
.venv/bin/python part1_ingest.py
```

Optional flags:
```bash
.venv/bin/python part1_ingest.py --docs ./my_docs --index ./data/faiss_index
```

### 5. Run Part 2 — ask questions

```bash
.venv/bin/python part2_rag.py
```

Optional flags:
```bash
.venv/bin/python part2_rag.py --compare          # RAG vs vanilla LLM side-by-side
.venv/bin/python part2_rag.py --question "..."   # single question, non-interactive
```

---

## LLM Providers

Set `LLM_PROVIDER` in `.env` to one of:

- **`ollama`** — local models via [Ollama](https://ollama.com/download) (no API key needed)
- **`openai`** — OpenAI API (requires `OPENAI_API_KEY`)
- **`huggingface`** — HuggingFace Inference API (requires `HF_API_TOKEN`)

---

## Compare mode

`--compare` sends the same question to both the RAG chain and a vanilla LLM (no documents), then prints both answers side by side.

```bash
.venv/bin/python part2_rag.py --compare
```

This is the core research angle of this MRP: demonstrating that grounding an LLM in a document corpus reduces hallucinations compared to a context-free response.

---

## MRP Experiments

These are the variables you can tune in `.env` to run experiments for your research. Re-run Part 1 after any chunking change (it rebuilds the index), then re-run Part 2 to observe the effect.

### Chunking strategy

| Variable | Try | Effect |
|---|---|---|
| `CHUNK_SIZE` | `200`, `500`, `1000` | Smaller = more precise retrieval but less context per chunk; larger = more context but noisier matches |
| `CHUNK_OVERLAP` | `0`, `50`, `100` | Higher overlap preserves sentence continuity across chunk boundaries |

> Rebuild the index after changing chunk settings: `python part1_ingest.py`

### Retrieval depth

| Variable | Try | Effect |
|---|---|---|
| `RETRIEVAL_TOP_K` | `1`, `2`, `4`, `8` | More chunks = more context for the LLM, but risks diluting relevance with noise |

### LLM model

Swap models without rebuilding the index — only Part 2 is affected.

| Provider | Models to try |
|---|---|
| Ollama | `llama3.2`, `mistral`, `phi3`, `gemma2` |
| OpenAI | `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o` |
| HuggingFace | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `google/flan-t5-xxl` |

### RAG vs vanilla LLM

Use `--compare` mode to ask the same question with and without document context. This is the core research comparison — RAG answers are grounded in your corpus; vanilla answers rely solely on the model's training data.

```bash
.venv/bin/python part2_rag.py --compare
```

### Embedding model

| Value | Notes |
|---|---|
| `sentence-transformers` | Local, no API key, ~90 MB, good general-purpose |
| `openai` | Cloud, requires `OPENAI_API_KEY`, higher quality embeddings |

> Switching embedding models requires a full index rebuild with Part 1.

---

## Configuration reference

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `ollama` / `huggingface` |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `HF_API_TOKEN` | — | HuggingFace token |
| `HF_MODEL` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | HuggingFace model |
| `EMBEDDING_MODEL` | `sentence-transformers` | `sentence-transformers` or `openai` |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per query |
| `DOCS_FOLDER` | `./docs` | Source documents folder |
| `INDEX_SAVE_PATH` | `./data/faiss_index` | FAISS index output path |
| `DISABLE_SSL_VERIFY` | `false` | Set `true` on corporate proxies with self-signed certs |

---

## Project structure

```
MRP-AI/
├── docs/                   # Drop your documents here
├── data/
│   └── faiss_index/        # Generated by part1_ingest.py
│       ├── index.faiss
│       └── index.pkl
├── part1_ingest.py         # Document ingestion & indexing
├── part2_rag.py            # RAG chain & interactive Q&A
├── config.py               # Central config (reads from .env)
├── requirements.txt
├── .env                    # Your local config (never commit this)
└── .env.example            # Template — copy to .env
```

---

## Embedding model

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` (~90 MB), which downloads automatically on first run and caches locally at `~/.cache/huggingface/hub/`. No API key needed.

To pre-cache the model (e.g. before going offline):

```bash
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies
- For Ollama: macOS / Linux / Windows with ~4 GB RAM free

---

## References

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al. (2020), the original RAG paper
2. [LangChain Documentation](https://python.langchain.com/docs/introduction/) — framework used for the RAG chain and document loaders
3. [FAISS — Facebook AI Similarity Search](https://faiss.ai/) — local vector store used for embedding indexing and retrieval
4. [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — embedding model used to encode documents and queries
5. [Ollama](https://ollama.com) — local LLM runner used to serve llama3.2 without an API key
