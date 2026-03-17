import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_MODEL: str = os.getenv("HF_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

    LLM_TEMPERATURE: float = 0.0  # 0 = deterministic, best for Q&A

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 4))

    DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "./docs")
    INDEX_SAVE_PATH: str = os.getenv("INDEX_SAVE_PATH", "./data/faiss_index")

    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".txt", ".md", ".docx")

    @property
    def needs_openai_key(self) -> bool:
        """True when an OpenAI API key is required for this configuration."""
        return self.LLM_PROVIDER == "openai" or self.EMBEDDING_MODEL == "openai"

    @property
    def needs_hf_token(self) -> bool:
        """True when a HuggingFace API token is required."""
        return self.LLM_PROVIDER == "huggingface"

    @property
    def active_model_name(self) -> str:
        """Human-readable active LLM name, for display purposes."""
        if self.LLM_PROVIDER == "ollama":
            return f"Ollama · {self.OLLAMA_MODEL}"
        if self.LLM_PROVIDER == "huggingface":
            return f"HuggingFace · {self.HF_MODEL}"
        return f"OpenAI · {self.LLM_MODEL}"


cfg = Config()
