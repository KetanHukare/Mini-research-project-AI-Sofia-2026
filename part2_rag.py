import os
import sys
import argparse
import textwrap
from pathlib import Path

# Must run before any network-making library is imported.
# Set DISABLE_SSL_VERIFY=true in .env when on a corporate proxy with self-signed certs.
from dotenv import load_dotenv as _load_dotenv_early
_load_dotenv_early()
if os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true":
    import ssl as _ssl
    import requests as _requests
    import urllib3 as _urllib3
    _ssl._create_default_https_context = _ssl._create_unverified_context
    _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)
    _orig_send = _requests.Session.send
    def _patched_send(self, req, **kwargs):
        kwargs["verify"] = False
        return _orig_send(self, req, **kwargs)
    _requests.Session.send = _patched_send

from colorama import Fore, Style, init
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import cfg

init(autoreset=True)


RAG_PROMPT_TEMPLATE = """You are a knowledgeable research assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't have enough information in the provided documents to answer this."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer (be concise and cite the source document name where relevant):"""

RAG_PROMPT = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)


def print_banner():
    print(Fore.MAGENTA + f"""
╔══════════════════════════════════════════════╗
║      MRP-AI  ·  Part 2: RAG Query Engine     ║
║  Provider : {cfg.active_model_name:<33}║
╚══════════════════════════════════════════════╝
""")


def validate_config():
    errors = []
    if cfg.needs_openai_key and not cfg.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is missing — required for "
                      + ("OpenAI LLM" if cfg.LLM_PROVIDER == "openai" else "OpenAI embeddings"))
    if cfg.needs_hf_token and not cfg.HF_API_TOKEN:
        errors.append("HF_API_TOKEN is missing — required for HuggingFace Inference API")
    if errors:
        for e in errors:
            print(Fore.RED + f"[ERROR] {e}")
        print(Fore.YELLOW + "\nCopy .env.example → .env and fill in your credentials.")
        sys.exit(1)


def load_embedding_model():
    """Load the same embedding model used during ingestion (must match Part 1)."""
    if cfg.EMBEDDING_MODEL == "openai":
        return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=cfg.OPENAI_API_KEY)
    elif cfg.EMBEDDING_MODEL == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    else:
        print(Fore.RED + f"[ERROR] Unknown EMBEDDING_MODEL: {cfg.EMBEDDING_MODEL}")
        sys.exit(1)


def load_vectorstore(index_path: str, embeddings):
    index_path = Path(index_path)
    if not index_path.exists():
        print(Fore.RED + f"[ERROR] FAISS index not found at: {index_path.resolve()}")
        print(Fore.YELLOW + "        Run Part 1 first:  python part1_ingest.py")
        answer = input(Fore.CYAN + "\n  Run ingestion now? (y/n): ").strip().lower()
        if answer == "y":
            from part1_ingest import run_ingestion
            vectorstore, _ = run_ingestion()
            return vectorstore
        sys.exit(1)

    print(Fore.GREEN + f"[✓] Loading FAISS index from: {index_path.resolve()}")
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we built this index ourselves
    )
    print(Fore.GREEN + f"[✓] Index loaded  ({vectorstore.index.ntotal} vectors)\n")
    return vectorstore


def build_llm():
    provider = cfg.LLM_PROVIDER

    if provider == "openai":
        llm = ChatOpenAI(model=cfg.LLM_MODEL, temperature=cfg.LLM_TEMPERATURE, openai_api_key=cfg.OPENAI_API_KEY)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            print(Fore.RED + "[ERROR] langchain-ollama is not installed. Run: pip install langchain-ollama")
            sys.exit(1)
        llm = ChatOllama(model=cfg.OLLAMA_MODEL, base_url=cfg.OLLAMA_BASE_URL, temperature=cfg.LLM_TEMPERATURE)

    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEndpoint
        except ImportError:
            print(Fore.RED + "[ERROR] langchain-huggingface is not installed. Run: pip install langchain-huggingface")
            sys.exit(1)
        llm = HuggingFaceEndpoint(repo_id=cfg.HF_MODEL, huggingfacehub_api_token=cfg.HF_API_TOKEN,
                                  temperature=cfg.LLM_TEMPERATURE, max_new_tokens=512)

    else:
        print(Fore.RED + f"[ERROR] Unknown LLM_PROVIDER: '{provider}'. Valid: openai | ollama | huggingface")
        sys.exit(1)

    print(Fore.GREEN + f"[✓] LLM ready    : {cfg.active_model_name}")
    return llm


def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": cfg.RETRIEVAL_TOP_K})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    print(Fore.GREEN + f"[✓] RAG chain ready (top-K={cfg.RETRIEVAL_TOP_K})\n")
    return chain


def ask_rag(chain, question: str) -> dict:
    """Query the RAG chain. Returns answer, source filenames, and raw chunks."""
    result = chain.invoke({"query": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})
    chunks = [doc.page_content for doc in result["source_documents"]]
    return {"answer": result["result"], "sources": sources, "chunks": chunks}


def ask_vanilla_llm(llm, question: str) -> str:
    """Ask the LLM without any retrieval context — used to demonstrate hallucinations."""
    response = llm.invoke(question)
    return response.content if hasattr(response, "content") else str(response)


def print_rag_response(result: dict):
    print("\n" + Fore.MAGENTA + "─" * 60)
    print(Fore.YELLOW + Style.BRIGHT + "  🤖 RAG Answer:")
    print(Fore.WHITE + textwrap.fill(result["answer"], width=70, initial_indent="  ", subsequent_indent="  "))
    print()
    print(Fore.CYAN + "  📚 Retrieved from:")
    for src in result["sources"]:
        print(Fore.CYAN + f"     • {src}")
    print(Fore.MAGENTA + "─" * 60 + "\n")


def print_vanilla_response(answer: str):
    print("\n" + Fore.RED + "─" * 60)
    print(Fore.RED + Style.BRIGHT + "  🤔 Vanilla LLM Answer (no documents):")
    print(Fore.WHITE + textwrap.fill(answer, width=70, initial_indent="  ", subsequent_indent="  "))
    print(Fore.RED + "─" * 60 + "\n")


def interactive_loop(chain, llm=None, compare_mode: bool = False):
    if compare_mode:
        print(Fore.YELLOW + """
  ┌─────────────────────────────────────────────────────┐
  │  COMPARISON MODE: RAG vs Vanilla LLM                 │
  │  Watch how RAG grounds answers in YOUR documents!    │
  └─────────────────────────────────────────────────────┘""")

    print(Fore.CYAN + "  Type your question and press Enter.")
    print(Fore.CYAN + "  Commands: 'quit' or 'exit' to stop | 'clear' to clear screen\n")

    while True:
        try:
            question = input(Fore.GREEN + "❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(Fore.YELLOW + "\n\n  Goodbye! 👋\n")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(Fore.YELLOW + "\n  Goodbye! 👋\n")
            break
        if question.lower() == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        print(Fore.CYAN + "\n  Searching documents and generating answer...")
        print_rag_response(ask_rag(chain, question))

        if compare_mode and llm:
            print(Fore.CYAN + "  Asking vanilla LLM (no documents)...")
            print_vanilla_response(ask_vanilla_llm(llm, question))


def initialise_rag_system(index_path: str = None):
    """Bootstrap the full RAG system. Returns (chain, llm) — importable by Part 3."""
    index_path = index_path or cfg.INDEX_SAVE_PATH
    validate_config()
    embeddings = load_embedding_model()
    vectorstore = load_vectorstore(index_path, embeddings)
    llm = build_llm()
    chain = build_rag_chain(vectorstore, llm)
    return chain, llm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRP-AI Part 2 — Interactive RAG Q&A over your documents")
    parser.add_argument("--compare", action="store_true",
                        help="Show vanilla LLM answer alongside RAG answer")
    parser.add_argument("--index", type=str, default=cfg.INDEX_SAVE_PATH,
                        help=f"Path to FAISS index (default: {cfg.INDEX_SAVE_PATH})")
    parser.add_argument("--question", type=str, default=None,
                        help="Ask a single question non-interactively")
    args = parser.parse_args()

    print_banner()
    chain, llm = initialise_rag_system(index_path=args.index)

    if args.question:
        print_rag_response(ask_rag(chain, args.question))
        if args.compare:
            print_vanilla_response(ask_vanilla_llm(llm, args.question))
    else:
        interactive_loop(chain, llm=llm, compare_mode=args.compare)
