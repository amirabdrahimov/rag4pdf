from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .config import Settings


@dataclass
class ChunkRecord:
    id: int
    text: str
    source: str
    page: int | None


class PdfRagAssistant:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.model = SentenceTransformer(self.settings.embedding_model_name)
        self.chunk_records: list[ChunkRecord] = []
        self.index: faiss.Index | None = None

    def initialize(self) -> None:
        docs = self._load_documents(self.settings.data_dir)
        chunks = self._split_documents(docs)
        self.chunk_records = self._build_chunk_records(chunks)
        self.index = self._load_or_build_index(self.chunk_records)

    def _load_documents(self, data_dir: Path) -> list[Any]:
        pdf_paths = sorted(data_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")

        docs: list[Any] = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["source"] = str(pdf_path)
            docs.extend(file_docs)

        print(f"Loaded {len(pdf_paths)} PDFs and {len(docs)} pages.")
        return docs

    def _split_documents(self, docs: list[Any]) -> list[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks.")
        return chunks

    def _build_chunk_records(self, chunks: list[Any]) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for idx, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    id=idx,
                    text=chunk.page_content,
                    source=chunk.metadata.get("source", "unknown"),
                    page=chunk.metadata.get("page", None),
                )
            )
        return records

    def _load_or_build_index(self, records: list[ChunkRecord]) -> faiss.Index:
        if self.settings.index_path.exists() and self.settings.metadata_path.exists():
            index = faiss.read_index(str(self.settings.index_path))
            with open(self.settings.metadata_path, "r", encoding="utf-8") as fh:
                persisted = json.load(fh)
            self.chunk_records = [
                ChunkRecord(
                    id=item["id"],
                    text=item["text"],
                    source=item.get("source", "unknown"),
                    page=item.get("page", None),
                )
                for item in persisted
            ]
            print(
                f"Loaded persisted index from {self.settings.index_path} with {index.ntotal} vectors."
            )
            return index

        texts = [record.text for record in records]
        embeddings = self.model.encode(texts, normalize_embeddings=True).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.settings.index_path))
        with open(self.settings.metadata_path, "w", encoding="utf-8") as fh:
            json.dump([record.__dict__ for record in records], fh, ensure_ascii=True)

        print(f"Built new index with {index.ntotal} vectors and saved to {self.settings.index_dir}/")
        return index

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Assistant is not initialized. Call initialize() first.")

        query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunk_records):
                continue
            record = self.chunk_records[idx]
            results.append(
                {
                    "text": record.text,
                    "source": record.source,
                    "page": record.page,
                    "score": float(score),
                }
            )
        return results

    def _ollama_generate(self, prompt: str, model_name: str | None = None) -> str:
        payload = {
            "model": model_name or self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.settings.ollama_api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.ollama_timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))

        if body.get("error"):
            raise RuntimeError(body["error"])

        text = (body.get("response") or "").strip()
        thinking = (body.get("thinking") or "").strip()
        if not text and thinking:
            raise RuntimeError(
                "Model returned thinking content but no final response. Try a different model or shorter context."
            )
        if not text:
            raise RuntimeError(f"Empty Ollama response. Raw keys: {list(body.keys())}")
        return text

    def answer(self, query: str, k: int = 4, llm_model: str | None = None) -> dict[str, Any]:
        retrieved = self.retrieve(query, k=k)
        if not retrieved:
            return {
                "answer": "I could not find relevant context in the indexed PDFs.",
                "sources": [],
            }

        context = "\n\n".join(
            [
                f"[{i + 1}] Source: {r['source']} | Page: {r['page']}\n{r['text']}"
                for i, r in enumerate(retrieved)
            ]
        )

        prompt = f"""You are a scientific assistant. Use only the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Return:
1) A concise answer
2) Citations like [source, page] for key claims
"""

        try:
            response_text = self._ollama_generate(prompt, model_name=llm_model)
        except Exception as exc:
            return {
                "answer": (
                    "Ollama request failed. Verify Ollama is running, the model exists, and the context is not too large. "
                    f"Details: {exc}"
                ),
                "sources": [
                    {
                        "source": r["source"],
                        "page": r["page"],
                        "score": r["score"],
                    }
                    for r in retrieved
                ],
            }

        return {
            "answer": response_text,
            "sources": [
                {
                    "source": r["source"],
                    "page": r["page"],
                    "score": r["score"],
                }
                for r in retrieved
            ],
        }