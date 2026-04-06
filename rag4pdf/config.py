from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    index_dir: Path = Path(".rag_index")
    embedding_model_name: str = "all-MiniLM-L6-v2"
    ollama_api_url: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
    ollama_timeout_sec: int = int(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))
    chunk_size: int = 500
    chunk_overlap: int = 50

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "chunks.json"