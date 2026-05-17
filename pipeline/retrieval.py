"""
retrieval.py
Hybrid search: BM25 + FAISS dense retrieval, fused with Reciprocal Rank Fusion,
then re-ranked with a cross-encoder for precision.
"""

from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_DENSE  = 10  # candidates from FAISS
TOP_K_BM25   = 10  # candidates from BM25
TOP_K_FINAL  = 4   # chunks returned after reranking


def _load_rerank_model() -> CrossEncoder:
    local_path = MODELS_DIR / "ms-marco-MiniLM-L-6-v2"
    if local_path.exists():
        return CrossEncoder(str(local_path))
    print(f"[retrieval] Downloading reranker → {local_path}")
    model = CrossEncoder(RERANK_MODEL_NAME)
    model.save(str(local_path))
    return model


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk indices."""
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    """
    Public API:
        retriever = HybridRetriever()
        retriever.load(chunks, faiss_index, bm25_index, embed_model)
        results = retriever.retrieve(query)   # list[str]
    """

    def __init__(self):
        self._reranker: CrossEncoder | None = None
        self.chunks: list[str] = []
        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self.embed_model: SentenceTransformer | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = _load_rerank_model()
        return self._reranker

    def load(
        self,
        chunks: list[str],
        index: faiss.Index,
        bm25: BM25Okapi,
        embed_model: SentenceTransformer,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.bm25 = bm25
        self.embed_model = embed_model

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[str]:
        if not self.chunks:
            raise RuntimeError("Call load() before retrieve().")

        # ── Dense retrieval ───────────────────────────────────────────────────
        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        _, dense_ids = self.index.search(q_emb, TOP_K_DENSE)
        dense_ranked = dense_ids[0].tolist()

        # ── BM25 retrieval ────────────────────────────────────────────────────
        token_q = query.lower().split()
        bm25_scores = self.bm25.get_scores(token_q)
        bm25_ranked = np.argsort(bm25_scores)[::-1][:TOP_K_BM25].tolist()

        # ── Fuse ──────────────────────────────────────────────────────────────
        fused = _rrf([dense_ranked, bm25_ranked])
        candidates = sorted(fused, key=fused.get, reverse=True)[: max(TOP_K_DENSE, TOP_K_BM25)]

        # ── Rerank ────────────────────────────────────────────────────────────
        pairs = [(query, self.chunks[i]) for i in candidates]
        rerank_scores = self.reranker.predict(pairs)
        scored = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in scored[:top_k]]
        return [self.chunks[i] for i in top_indices]
