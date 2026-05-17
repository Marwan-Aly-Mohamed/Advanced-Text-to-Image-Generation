# 📖 Visual Book QA

A local RAG (Retrieval-Augmented Generation) app that lets you upload any PDF book, ask questions about it, and receive answers as beautifully rendered visual slides — all running entirely on your machine.

---

## ✨ How It Works

1. **Upload a PDF** — the book is extracted, chunked, and indexed.
2. **Ask a question** — hybrid search finds the most relevant passages.
3. **Get a visual slide** — a local LLM builds a structured answer and renders it as a slide image.

```
PDF → Chunks → FAISS + BM25 indexes
                    ↓
     Query → Hybrid Retrieval → Reranker
                    ↓
          Ollama (qwen2.5:7b) → JSON
                    ↓
          HTML Slide → Playwright → PNG
```

---

## 🗂️ Project Structure

```
project/
├── app.py                  # Gradio UI entry point
├── pipeline/
│   ├── ingestion.py        # PDF extraction, chunking, FAISS + BM25 indexing
│   ├── retrieval.py        # Hybrid search (dense + BM25 + RRF + cross-encoder rerank)
│   ├── generation.py       # Ollama prompting + SVG slide builder
│   └── renderer.py         # HTML → PNG via Playwright
├── models/                 # Auto-downloaded embedding & reranker models
├── cache/                  # Per-PDF index cache (auto-created)
└── outputs/                # Generated slide PNGs (auto-created)
```

---

## ⚙️ Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| [Ollama](https://ollama.com/download) | Latest |
| Chromium (via Playwright) | Auto-installed |

---

## 🚀 Installation

**1. Clone and set up a virtual environment**

```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**2. Install Python dependencies**

```bash
pip install gradio pymupdf faiss-cpu numpy rank-bm25 sentence-transformers playwright pillow
```

**3. Install Playwright's Chromium browser**

```bash
playwright install chromium
```

**4. Install and start Ollama**

Download from [ollama.com/download](https://ollama.com/download), then pull the model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before launching the app:

```bash
ollama serve
```

---

## ▶️ Running the App

```bash
python app.py
```

Then open your browser at **http://127.0.0.1:7860**

---

## 🧩 Pipeline Details

### `ingestion.py` — PDF Ingester

- Extracts text from PDF pages using **PyMuPDF**
- Splits text into overlapping chunks (`400` chars, `80` overlap)
- Encodes chunks with **`all-MiniLM-L6-v2`** (sentence-transformers)
- Builds a **FAISS** inner-product index (cosine similarity on normalised vectors)
- Builds a **BM25Okapi** index for keyword search
- Caches everything under `cache/<md5_hash>/` — re-runs are instant

### `retrieval.py` — Hybrid Retriever

- **Dense retrieval**: top-20 candidates from FAISS
- **Sparse retrieval**: top-20 candidates from BM25
- **Fusion**: Reciprocal Rank Fusion (RRF) merges both ranked lists
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all candidates
- Returns the **top 7** most relevant chunks

### `generation.py` — Answer Generator

- Sends the query + top-5 chunks to **Ollama (`qwen2.5:3b`)** via its local REST API
- Prompts the model to return a strict **JSON schema** (title, bullets, diagram type, nodes, etc.)
- Builds a self-contained **HTML/SVG slide** from the JSON
- Supports 7 diagram types: `layers`, `flowchart`, `cycle`, `split`, `network`, `tree`, `timeline`

### `renderer.py` — HTML Renderer

- Saves the HTML to a temp file
- Opens it in a headless **Playwright Chromium** browser (960×640 viewport)
- Screenshots it to a timestamped **PNG** under `outputs/`
- Also saves `outputs/last_debug.html` for inspection

---

## 🛠️ Configuration

Key constants you can tweak:

| File | Constant | Default | Description |
|---|---|---|---|
| `ingestion.py` | `CHUNK_SIZE` | `400` | Characters per chunk |
| `ingestion.py` | `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `ingestion.py` | `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model |
| `retrieval.py` | `TOP_K_DENSE` | `20` | FAISS candidates |
| `retrieval.py` | `TOP_K_BM25` | `20` | BM25 candidates |
| `retrieval.py` | `TOP_K_FINAL` | `7` | Chunks after reranking |
| `generation.py` | `OLLAMA_MODEL` | `qwen2.5:3b` | LLM used for generation |
| `renderer.py` | `VIEWPORT_W/H` | `960×640` | Slide render dimensions |

---

## 🐛 Troubleshooting

**`Ollama is not running`**
Make sure `ollama serve` is running in a separate terminal before starting the app.

**`Playwright not found`**
Run `playwright install chromium` and ensure you're inside the correct virtual environment.

**`FAISS` install issues on Apple Silicon**
Use `pip install faiss-cpu` — the CPU version works on all platforms.

**Slow first run**
The embedding model (`~90MB`) and reranker model (`~70MB`) are downloaded once to `models/` on first use. Subsequent runs load them from disk.

**JSON parse errors from the LLM**
The generator includes a fallback slide when the model returns malformed JSON. Try rephrasing the question or using a larger Ollama model (e.g. `qwen2.5:7b`).

---

## 📄 Future work


