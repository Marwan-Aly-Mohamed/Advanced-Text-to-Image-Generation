"""
ingestion.py
Handles PDF → chunks → embeddings → FAISS + BM25 indexes.
All artifacts are cached under cache/<pdf_stem>/ so re-runs skip processing.
"""

import os
import pickle
import hashlib
from pathlib import Path

import fitz  # pymupdf
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
CACHE_DIR = Path("cache")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks

MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def _load_embed_model() -> SentenceTransformer:
    """Load embedding model from local cache; download once if missing."""
    local_path = MODELS_DIR / "all-MiniLM-L6-v2"
    if local_path.exists():
        return SentenceTransformer(str(local_path))
    print(f"[ingestion] Downloading embedding model → {local_path}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model.save(str(local_path))
    return model


def _pdf_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:12]


def _extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 30]   # drop tiny tail chunks


class PDFIngester:
    """
    Public API:
        ingester = PDFIngester()
        chunks, faiss_index, bm25 = ingester.ingest(pdf_path)
    """

    def __init__(self):
        self._embed_model: SentenceTransformer | None = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = _load_embed_model()
        return self._embed_model

    def ingest(self, pdf_path: str) -> tuple[list[str], faiss.Index, BM25Okapi]:
        """
        Returns (chunks, faiss_index, bm25_index).
        Results are cached; subsequent calls with the same PDF are instant.
        """
        pdf_path = str(pdf_path)
        key = _pdf_hash(pdf_path)
        cache_dir = CACHE_DIR / key
        chunks_file = cache_dir / "chunks.pkl"
        faiss_file  = cache_dir / "index.faiss"
        bm25_file   = cache_dir / "bm25.pkl"

        if chunks_file.exists() and faiss_file.exists() and bm25_file.exists():
            print(f"[ingestion] Cache hit for {Path(pdf_path).name}")
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            index = faiss.read_index(str(faiss_file))
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
            return chunks, index, bm25

        # ── Process ───────────────────────────────────────────────────────────
        print(f"[ingestion] Processing {Path(pdf_path).name} …")
        text = _extract_text(pdf_path)
        chunks = _chunk_text(text)
        print(f"[ingestion] {len(chunks)} chunks created")

        # Embeddings
        print("[ingestion] Embedding chunks …")
        embeddings = self.embed_model.encode(
            chunks, show_progress_bar=True, batch_size=64, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype="float32")

        # FAISS index (inner-product on normalised vectors = cosine similarity)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # BM25
        tokenized = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)

        # Cache to disk
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        faiss.write_index(index, str(faiss_file))
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)

        print("[ingestion] Done — indexes saved to cache.")
        return chunks, index, bm25
