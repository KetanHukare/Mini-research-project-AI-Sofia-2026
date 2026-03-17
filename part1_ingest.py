"""Part 1 — Document ingestion and FAISS index builder. See README.md for usage."""

import os
import sys
import argparse
import time
from pathlib import Path

# Must run before any network-making library is imported.
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
    print("[SSL] Certificate verification disabled (DISABLE_SSL_VERIFY=true)")

from tqdm import tqdm
from colorama import Fore, Style, init
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import cfg

init(autoreset=True)


def print_banner():
    print(Fore.CYAN + """
╔══════════════════════════════════════════════╗
║      MRP-AI  ·  Part 1: Document Ingestion   ║
╚══════════════════════════════════════════════╝
""")


def get_loader(file_path: str):
    """Return the appropriate LangChain loader for a given file extension."""
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".md":   UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }
    loader_cls = loaders.get(ext)
    return loader_cls(file_path) if loader_cls else None


def scan_docs_folder(folder: str) -> list[str]:
    """Recursively find all supported documents in the given folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        print(Fore.RED + f"[ERROR] Docs folder not found: {folder}")
        print(Fore.YELLOW + f"        Create it and drop your files in: {folder_path.resolve()}")
        sys.exit(1)

    found_files, skipped_files = [], []
    for file_path in sorted(folder_path.rglob("*")):
        if file_path.is_file():
            if file_path.suffix.lower() in cfg.SUPPORTED_EXTENSIONS:
                found_files.append(str(file_path))
            else:
                skipped_files.append(file_path.name)

    print(Fore.GREEN + f"[✓] Docs folder : {folder_path.resolve()}")
    print(Fore.GREEN + f"[✓] Files found : {len(found_files)}")
    if skipped_files:
        print(Fore.YELLOW + f"[!] Skipped     : {len(skipped_files)} unsupported file(s): "
              + ", ".join(skipped_files))

    if not found_files:
        print(Fore.RED + "\n[ERROR] No supported documents found.")
        print(Fore.YELLOW + f"        Supported: {cfg.SUPPORTED_EXTENSIONS}")
        sys.exit(1)

    print()
    for f in found_files:
        print(f"    📄 {Path(f).name}")
    print()
    return found_files


def load_documents(file_paths: list[str]) -> list:
    """Load all documents and return a flat list of LangChain Document objects."""
    all_docs = []
    print(Fore.CYAN + "── Step 1/4 · Loading documents ─────────────────────")

    for file_path in tqdm(file_paths, desc="Loading", unit="file"):
        loader = get_loader(file_path)
        if loader is None:
            print(Fore.YELLOW + f"  [!] Skipping unsupported: {file_path}")
            continue
        try:
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = Path(file_path).name
            all_docs.extend(docs)
            print(Fore.GREEN + f"  [✓] Loaded {len(docs):>3} page(s) from: {Path(file_path).name}")
        except Exception as e:
            print(Fore.RED + f"  [✗] Failed to load {file_path}: {e}")

    print(f"\n    Total raw document pages loaded: {Fore.YELLOW}{len(all_docs)}\n")
    return all_docs


def split_into_chunks(documents: list) -> list:
    """Split documents into overlapping chunks.
    RecursiveCharacterTextSplitter tries paragraph → sentence → word boundaries
    so chunks respect semantic structure rather than cutting mid-sentence.
    """
    print(Fore.CYAN + "── Step 2/4 · Splitting into chunks ─────────────────")
    print(f"    Chunk size   : {cfg.CHUNK_SIZE} tokens")
    print(f"    Chunk overlap: {cfg.CHUNK_OVERLAP} tokens\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"    Raw pages   → {len(documents)}")
    print(f"    Chunks made → {Fore.YELLOW}{len(chunks)}")
    print(f"    Avg chunk size: ~{sum(len(c.page_content) for c in chunks) // len(chunks)} chars\n")
    return chunks


def build_embedding_model():
    """Build the embedding model. Options: 'openai' or 'sentence-transformers'."""
    print(Fore.CYAN + "── Step 3/4 · Building embedding model ──────────────")

    if cfg.EMBEDDING_MODEL == "openai":
        if not cfg.OPENAI_API_KEY:
            print(Fore.RED + "[ERROR] OPENAI_API_KEY is not set in .env")
            print(Fore.YELLOW + "        Set EMBEDDING_MODEL=sentence-transformers for a free local option")
            sys.exit(1)
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=cfg.OPENAI_API_KEY)
        print(Fore.GREEN + "    [✓] Using OpenAI text-embedding-ada-002\n")

    elif cfg.EMBEDDING_MODEL == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings
        print("    Loading local model: all-MiniLM-L6-v2...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        print(Fore.GREEN + "    [✓] Using sentence-transformers/all-MiniLM-L6-v2 (local, free)\n")

    else:
        print(Fore.RED + f"[ERROR] Unknown EMBEDDING_MODEL: '{cfg.EMBEDDING_MODEL}'")
        print(Fore.YELLOW + "        Valid options: 'openai' or 'sentence-transformers'")
        sys.exit(1)

    return embeddings


def build_and_save_index(chunks: list, embeddings, save_path: str):
    """Embed all chunks and persist the FAISS index to disk."""
    print(Fore.CYAN + "── Step 4/4 · Building FAISS vector index ───────────")
    print(f"    Embedding {len(chunks)} chunks...")

    start = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start

    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)

    print(Fore.GREEN + f"    [✓] Index built in {elapsed:.1f}s")
    print(Fore.GREEN + f"    [✓] Saved to: {Path(save_path).resolve()}\n")
    return vectorstore


def run_ingestion(docs_folder: str = None, index_save_path: str = None):
    """Full ingestion pipeline. Importable by part2_rag.py."""
    docs_folder = docs_folder or cfg.DOCS_FOLDER
    index_save_path = index_save_path or cfg.INDEX_SAVE_PATH

    print_banner()
    file_paths = scan_docs_folder(docs_folder)
    documents = load_documents(file_paths)
    chunks = split_into_chunks(documents)
    embeddings = build_embedding_model()
    vectorstore = build_and_save_index(chunks, embeddings, index_save_path)

    print(Fore.GREEN + Style.BRIGHT + "✅  Ingestion complete! You can now run part2_rag.py\n")
    return vectorstore, embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRP-AI Part 1 — Ingest documents into a FAISS vector index")
    parser.add_argument("--docs", type=str, default=cfg.DOCS_FOLDER,
                        help=f"Path to docs folder (default: {cfg.DOCS_FOLDER})")
    parser.add_argument("--index", type=str, default=cfg.INDEX_SAVE_PATH,
                        help=f"Where to save the FAISS index (default: {cfg.INDEX_SAVE_PATH})")
    args = parser.parse_args()
    run_ingestion(docs_folder=args.docs, index_save_path=args.index)
