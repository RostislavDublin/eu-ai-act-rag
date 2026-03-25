# Module 3.2 - RAG (Practical Task)

**Author:** Rostislav Dublin
**Submission level:** Best (Streamlit app)
**Deadline:** Mar 26, 2026 18:59

## Task Summary

Build a RAG application that analyzes documents, answers questions, extracts data, and visualizes quality metrics. Deliverable: interactive Streamlit app with annotated source code.

## Domain

**EU AI Act Q&A Assistant** - takes user questions about the EU Artificial Intelligence Act and produces grounded answers with source citations.

Chosen because:
- Specialized legal document - LLMs have limited training coverage (reduces pre-existing knowledge contamination, per trainer feedback).
- Questions have verifiable answers in the source text (enables faithful evaluation).
- Domain is relevant to AI architects (regulatory compliance).
- Full text is publicly available.

## Approach

### Architecture: Streamlit app with in-memory RAG pipeline

Interactive app. No external services beyond DIAL API. In-memory vector store (ChromaDB). Source code is modular but compact (4 files).

```
PDF/text -> chunking -> embeddings (DIAL) -> ChromaDB (in-memory)
                                                    |
user question -> embedding -> [semantic search top-K]  ]
             -> tokenize  -> [BM25 keyword search top-K] ] -> RRF fusion -> top-K chunks
                                                    |
                                          prompt + context -> DIAL LLM -> answer
                                                    |
                                          evaluation (RAG Triad) -> radar chart
```

### App pages / tabs

| Tab | Purpose |
|-----|---------|
| Chat | Ask questions, see answers with source chunks highlighted |
| Corpus | View loaded documents, chunk count, coverage stats |
| Evaluation | Run eval dataset, see per-question scores |
| Quality Dashboard | RAGAS radar chart + per-question breakdown bars |

### Evaluation: RAG Triad

Three metrics measured per question, visualized as a radar (spider) chart:

- **Faithfulness** - is the answer grounded in the retrieved context? (no hallucination)
- **Answer Relevancy** - does the answer address the question? (no off-topic drift)
- **Context Precision** - did the retriever pull the right chunks? (retrieval quality)

Implementation: LLM-as-judge via DIAL (GPT-4o evaluates each dimension with a structured rubric).

### Visualization

- RAGAS Quality Radar Chart: spider chart with 3 axes, one polygon per question or averaged.
- Per-question grouped bar chart showing individual metric scores.
- Plotly for interactive charts inside Streamlit.

## What we borrow from rag-lab (ideas only, fresh code)

| Component | rag-lab approach | POC simplification |
|-----------|-----------------|-------------------|
| Embedding | Google GenAI SDK, text-embedding-005 | DIAL `text-embedding-005`, 768-dim, same model via OpenAI-compatible API |
| Vector store | PostgreSQL + pgvector | ChromaDB in-memory (zero setup) |
| Chunking | 2000 chars, 200 overlap | Same strategy, simpler code |
| Hybrid retrieval | N/A | Semantic (ChromaDB cosine) + BM25 keyword, fused via RRF |
| LLM | Google GenAI SDK | DIAL API chat completions (generation only) |
| Evaluation | E2E tests | LLM-as-judge RAG Triad |

## Hosting

**Production (submission):** Streamlit Community Cloud (cloud.streamlit.io)
- Deploys from GitHub repo, free tier.
- `DIAL_TOKEN` and other credentials stored in Streamlit Secrets (UI, not committed to git).
- Reviewer gets a public URL, no local setup needed.
- After submission: rotate DIAL token or delete the app.

**Local run (development / fallback):**

```bash
cd module-3.2-rag
cp .env-template .env        # fill in DIAL credentials
pip install -r requirements.txt
streamlit run app.py
```

The app loads dataset from `data/` on startup and builds the vector index automatically.

## Dataset

EU AI Act full text. Options:
- Download from EUR-Lex (official source)
- Use a pre-processed plain-text version in `data/`

## File structure

```
module-3.2-rag/
  .env-template          # credential template (committed)
  .env                   # real credentials (gitignored)
  README.md              # this file
  requirements.txt       # Python dependencies
  app.py                 # Streamlit entry point (UI, tabs, session state)
  rag/
    __init__.py
    pipeline.py          # chunking, embedding, retrieval, generation
    evaluation.py        # RAG Triad LLM-as-judge scoring
    visualization.py     # radar chart, bar charts (plotly)
  data/                  # dataset files (EU AI Act text)
  eval_questions.json    # evaluation Q&A pairs with expected answers
```

## Action items

- [x] Verify DIAL API supports embedding endpoint (`text-embedding-005`, 768-dim, confirmed)
- [x] Download EU AI Act text (public), find/create one restricted document
- [x] Create rag/pipeline.py:
  - chunking (2000 chars, 200 overlap)
  - embedding via DIAL `text-embedding-005`
  - hybrid retrieval: semantic (ChromaDB) + BM25 keyword, fused via RRF
  - clearance metadata filter applied before both search legs
  - generation via DIAL chat completions
- [x] Create rag/evaluation.py (RAG Triad LLM-as-judge: faithfulness, answer relevancy, context precision)
- [x] Create rag/visualization.py (radar chart, average radar, bar chart, plotly)
- [x] Create app.py (Streamlit UI: Chat / Corpus / Evaluation / Quality Dashboard)
  - Corpus tab: upload arbitrary .txt/.pdf file, incremental index (SHA-256 dedup)
  - Corpus tab: clearance badge per document, chunk count
- [x] Create eval_questions.json (12 Q&A pairs: 10 public + 2 restricted)
- [x] Update requirements.txt (pypdf, rank-bm25)
- [ ] End-to-end test: upload a new document, verify it appears in corpus without full rebuild
- [ ] End-to-end test: public user (no token) cannot see restricted doc answers
- [ ] End-to-end test: authenticated user sees all answers
- [ ] End-to-end test: eval runs, radar chart renders
- [ ] Review README, verify clean start from scratch works
- [ ] Deploy to Streamlit Community Cloud, add DIAL_TOKEN to Streamlit Secrets
- [ ] Verify live URL works end-to-end before submission
- [ ] Push to git, prepare submission link (Streamlit app URL + GitLab PR)

## Ninja challenges (planned)

### 1. Corpus update w/o rebuild (incremental indexing)

Upload button on the Corpus tab. On upload: compute SHA-256 of the file, check if any chunk with that `source_hash` already exists in ChromaDB, skip if yes, add only new chunks if no. No full rebuild, no duplicates.

### 2. Access control aware RAG

Two access tiers enforced at retrieval time via ChromaDB metadata filtering:

- Each chunk is stored with a `clearance` field: `"public"` or `"restricted"`.
- The EU AI Act base text is loaded as `public`.
- One additional document (e.g., an internal compliance commentary) is loaded as `restricted`.
- Session state holds the current `user_clearance`, determined at startup:
  - If `DIAL_TOKEN` is set in `.env` (i.e., the user has authenticated credentials) → `"restricted"`
  - If not set → `"public"`
- Every `collection.query()` call includes `where={"clearance": {"$in": allowed_tiers}}` where `allowed_tiers = ["public", "restricted"]` for authenticated users and `["public"]` for anonymous.
- The Corpus tab shows the clearance badge next to each loaded document.

This mirrors a real SSO pattern stripped to essentials: token presence = identity claim, metadata filter = enforcement point.

### 3. Full RAG Triad + precision/recall metrics

Context Precision and Context Recall added to evaluation alongside Faithfulness and Answer Relevancy. All four axes on the radar chart.

### Optional (if time allows)

- [ ] Reranking step (two-stage retrieval)
- [ ] Multi-modal RAG (extract tables/diagrams from PDF via GPT-4o vision)
