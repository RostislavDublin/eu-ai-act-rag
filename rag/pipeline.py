"""
RAG pipeline: DIAL client, chunker, vector store, BM25 index, hybrid retriever, generator.
"""
import hashlib
import os
import re
from typing import Optional

import chromadb
import requests
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# DIAL API client
# ---------------------------------------------------------------------------

class DIALClient:
    """LLM client supporting EPAM DIAL and direct OpenAI API."""

    def __init__(self, base_url: str, token: str, chat_model: str, embed_model: str, provider: str = "dial"):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.token = token
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.provider = provider  # "dial" | "openai"
        if provider == "openai":
            self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        else:
            self._headers = {"Api-Key": token, "Content-Type": "application/json"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            url = "https://api.openai.com/v1/embeddings"
            payload = {"model": self.embed_model, "input": texts}
        else:
            url = f"{self.base_url}/openai/deployments/{self.embed_model}/embeddings"
            payload = {"input": texts}
        resp = requests.post(url, headers=self._headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        if self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            payload = {"model": self.chat_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        else:
            url = f"{self.base_url}/openai/deployments/{self.chat_model}/chat/completions"
            payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        resp = requests.post(url, headers=self._headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Vector store (ChromaDB in-memory with clearance metadata)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, client: DIALClient):
        self._dial = client
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name="rag_store",
            metadata={"hnsw:space": "cosine"},
        )

    def _existing_hashes(self) -> set[str]:
        result = self._collection.get(include=["metadatas"])
        if not result["metadatas"]:
            return set()
        return {m["source_hash"] for m in result["metadatas"]}

    def add_document(
        self,
        text: str,
        source_name: str,
        clearance: str = "public",
    ) -> tuple[int, int]:
        """
        Add a document to the vector store.
        Returns (added_chunks, skipped_chunks).
        Skips entirely if source_hash already exists (dedup by content).
        """
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        if file_hash in self._existing_hashes():
            chunks = chunk_text(text)
            return 0, len(chunks)

        chunks = chunk_text(text)
        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source_name,
                "source_hash": file_hash,
                "clearance": clearance,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Embed in batches of 100 (DIAL limit)
        embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            embeddings.extend(self._dial.embed(chunks[i : i + batch_size]))

        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        return len(chunks), 0

    def query_semantic(
        self, query_embedding: list[float], clearance: str, n_results: int = 10
    ) -> list[dict]:
        allowed = ["public", "restricted"] if clearance == "restricted" else ["public"]
        where = {"clearance": {"$in": allowed}}
        # Clamp to actual collection size
        total = self._collection.count()
        if total == 0:
            return []
        n_results = min(n_results, total)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "clearance": meta["clearance"],
                    "distance": dist,
                }
            )
        return chunks

    def get_all_chunks(self, clearance: str) -> list[dict]:
        allowed = ["public", "restricted"] if clearance == "restricted" else ["public"]
        where = {"clearance": {"$in": allowed}}
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            chunks.append(
                {"text": doc, "source": meta["source"], "clearance": meta["clearance"]}
            )
        return chunks

    def corpus_summary(self) -> list[dict]:
        """Return per-document stats: name, clearance, chunk count."""
        result = self._collection.get(include=["metadatas"])
        stats: dict[str, dict] = {}
        for meta in result["metadatas"]:
            key = meta["source"]
            if key not in stats:
                stats[key] = {"source": key, "clearance": meta["clearance"], "chunks": 0}
            stats[key]["chunks"] += 1
        return list(stats.values())

    def total_chunks(self) -> int:
        return self._collection.count()

    def get_document_chunks(self, source_name: str, clearance: str) -> list[dict]:
        """Return all chunks for a given source document, sorted by chunk_index."""
        allowed = ["public", "restricted"] if clearance == "restricted" else ["public"]
        result = self._collection.get(
            where={"$and": [{"source": {"$eq": source_name}}, {"clearance": {"$in": allowed}}]},
            include=["documents", "metadatas"],
        )
        chunks = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "clearance": meta["clearance"],
                "chunk_index": meta.get("chunk_index", 0),
            })
        chunks.sort(key=lambda c: c["chunk_index"])
        return chunks


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def query(self, query_text: str, n_results: int = 10) -> list[tuple[int, float]]:
        """Returns list of (original_index, score) sorted by score descending.
        Chunks scoring below BM25_MIN_SCORE_FRACTION * top_score are excluded,
        preventing weakly-matching chunks from inflating their RRF rank.
        """
        if self._bm25 is None or not self._chunks:
            return []
        tokens = self._tokenize(query_text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] <= 0.0:
            return []  # no keyword overlap at all
        min_score = ranked[0][1] * BM25_MIN_SCORE_FRACTION
        ranked = [(idx, s) for idx, s in ranked if s >= min_score]
        return ranked[:n_results]


# ---------------------------------------------------------------------------
# Hybrid retriever with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

RRF_K = 60
# BM25 relative threshold: chunk must score >= top_score * fraction to enter RRF.
# Prevents weakly-matching chunks (1-2 incidental keywords) from getting rank=1.
BM25_MIN_SCORE_FRACTION = 0.25
# Cosine distance >= this means semantically too far - exclude from fusion
SEM_MAX_DISTANCE = 0.85


def reciprocal_rank_fusion(
    semantic_chunks: list[dict],
    bm25_results: list[tuple[int, float]],
    all_chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Fuse semantic (vector) and BM25 (keyword) rankings via RRF.

    semantic_chunks - ordered by vector similarity (rank = position in list)
    bm25_results    - list of (index_into_all_chunks, bm25_score) ordered by score
    all_chunks      - full chunk list used to build BM25 index (for lookup)

    Each returned chunk has extra keys:
      rrf_score    - final fused score (higher = better)
      sem_rank     - 1-based rank from semantic search (None if not in semantic results)
      bm25_rank    - 1-based rank from BM25 search (None if not in BM25 results)
      sem_distance - cosine distance from ChromaDB (lower = more similar; None if absent)
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    sem_rank_map: dict[str, int] = {}
    bm25_rank_map: dict[str, int] = {}

    for rank, chunk in enumerate(semantic_chunks):
        key = chunk["text"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_map[key] = chunk
        sem_rank_map[key] = rank + 1

    bm25_score_map: dict[str, float] = {}
    for rank, (idx, bm25_score) in enumerate(bm25_results):
        c = all_chunks[idx]
        key = c["text"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        bm25_rank_map[key] = rank + 1
        bm25_score_map[key] = float(bm25_score)
        if key not in chunk_map:
            chunk_map[key] = c

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for k, score in ranked[:top_n]:
        chunk = dict(chunk_map[k])  # shallow copy to avoid mutating cached data
        chunk["rrf_score"] = round(score, 5)
        chunk["sem_rank"] = sem_rank_map.get(k)            # None if BM25-only
        chunk["bm25_rank"] = bm25_rank_map.get(k)          # None if semantic-only
        chunk["bm25_score"] = bm25_score_map.get(k)        # None if semantic-only
        chunk["sem_distance"] = chunk.get("distance")      # from query_semantic
        result.append(chunk)
    return result


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, dial_client: DIALClient):
        self._vs = vector_store
        self._dial = dial_client
        self._bm25 = BM25Index()

    def rebuild_bm25(self, clearance: str) -> None:
        """Rebuild BM25 index from current corpus (call after adding documents)."""
        chunks = self._vs.get_all_chunks(clearance)
        self._bm25.build(chunks)

    def retrieve(self, query: str, clearance: str, top_n: int = 5) -> list[dict]:
        if self._vs.total_chunks() == 0:
            return []
        query_embedding = self._dial.embed([query])[0]
        semantic = self._vs.query_semantic(query_embedding, clearance, n_results=top_n * 2)
        # Filter semantically distant chunks before RRF - they are ranked but not relevant
        semantic = [c for c in semantic if c.get("distance", 1.0) < SEM_MAX_DISTANCE]
        all_chunks = self._vs.get_all_chunks(clearance)
        bm25_results = self._bm25.query(query, n_results=top_n * 2)
        return reciprocal_rank_fusion(semantic, bm25_results, all_chunks, top_n=top_n)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise Q&A assistant for the EU AI Act. "
    "Answer ONLY based on the provided context. "
    "If the context does not contain enough information to answer, say so explicitly. "
    "Do NOT add a '(Source: ...)' line at the end - sources are shown separately in the UI."
)


class Generator:
    def __init__(self, dial_client: DIALClient):
        self._dial = dial_client

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, str]:
        """
        Generate an answer grounded in the retrieved chunks.
        Returns (answer, context_text).
        """
        if not chunks:
            return "No relevant context found in the corpus.", ""

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        answer = self._dial.chat(messages, temperature=0.0)
        return answer, context


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def build_pipeline_from_env(
    base_url: str, token: str, chat_model: str = "gpt-4o", embed_model: str = "text-embedding-005",
    provider: str = "dial",
) -> tuple["DIALClient", "VectorStore", "HybridRetriever", "Generator"]:
    """Construct pipeline components from explicit credentials."""
    client = DIALClient(base_url, token, chat_model, embed_model, provider=provider)
    store = VectorStore(client)
    retriever = HybridRetriever(store, client)
    generator = Generator(client)
    return client, store, retriever, generator
